from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
from datetime import date
from database import get_supabase
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
import os
from dotenv import load_dotenv

load_dotenv()

router = APIRouter(prefix="/follow-up", tags=["follow-up"])

# Initialize LLM
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
llm = ChatGoogleGenerativeAI(
    google_api_key=GEMINI_API_KEY,
    model="gemini-2.5-flash",
    temperature=0.4, # Lower temperature for more concise/consistent outputs
)


@router.get("/medicine-status/{user_id}")
async def check_medicine_status(user_id: int):
    """
    Returns personalized, short notification text based on adherence.
    """
    try:
        supabase = get_supabase()

        # Fetch all medicines for this user
        meds_response = (
            supabase
            .table("medicine_data")
            .select("*")
            .eq("user_id", user_id)
            .execute()
        )

        if not hasattr(meds_response, "data"):
            raise HTTPException(500, "Invalid Supabase response structure.")

        medicines = meds_response.data or []

        if len(medicines) == 0:
            return {
                "user_id": user_id,
                "notifications": [],
                "notification_text": "No medications found.",
                "adherence_score": None
            }

        # Fetch daily routine records for today
        today = date.today().isoformat()
        routine_response = (
            supabase
            .table("daily_routine")
            .select("*")
            .eq("user_id", user_id)
            .eq("date_filled", today)
            .execute()
        )

        daily_records = routine_response.data or []
        routine_map = {record["medicine_id"]: record for record in daily_records}

        notifications: List[Dict[str, Any]] = []
        
        # Track names for better LLM personalization
        pending_med_names = [] 
        
        adherence_data = {
            "total_meds": len(medicines),
            "taken": 0,
            "missed": 0,
            "late": 0,
            "pending": 0
        }

        for med in medicines:
            med_id = med.get("id")
            name = med.get("name", "Medicine")
            dose = med.get("dose", "")
            
            routine = routine_map.get(med_id)

            if not routine:
                # No record -> Pending
                adherence_data["pending"] += 1
                pending_med_names.append(name)
                notifications.append({
                    "type": "reminder",
                    "medicine_id": med_id,
                    "medicine_name": name,
                    "status": "pending"
                })
            else:
                taken = routine.get("taken", False)
                not_taken = routine.get("not_taken", False)
                late_taken = routine.get("late_taken")

                if taken and not late_taken:
                    adherence_data["taken"] += 1
                    notifications.append({"type": "success", "medicine_id": med_id, "medicine_name": name, "status": "taken_on_time"})
                elif taken and late_taken:
                    adherence_data["late"] += 1
                    notifications.append({"type": "late", "medicine_id": med_id, "medicine_name": name, "status": "taken_late"})
                elif not_taken:
                    adherence_data["missed"] += 1
                    notifications.append({"type": "missed", "medicine_id": med_id, "medicine_name": name, "status": "missed"})
                else:
                    # Record exists but no status -> Pending
                    adherence_data["pending"] += 1
                    pending_med_names.append(name)
                    notifications.append({"type": "reminder", "medicine_id": med_id, "medicine_name": name, "status": "pending"})

        # Calculate adherence score
        total = adherence_data["total_meds"]
        adherence_score = round(
            ((adherence_data["taken"] + adherence_data["late"] * 0.5) / total * 100) if total > 0 else 0,
            1
        )

        # --- UPDATED PROMPT LOGIC ---
        
        # Format pending names for the prompt
        pending_list_str = ", ".join(pending_med_names) if pending_med_names else "None"

        system_prompt = """You are a mobile health assistant.
        Generate a **single, short, urgent but friendly push notification** (max 15-20 words).
        
        Rules:
        1. If medicines are PENDING: Focus strictly on reminding the user to take them. Mention the medicine name if there is only 1 or 2.
        2. If medicines are MISSED: Express gentle concern and ask if they want to reschedule.
        3. If all TAKEN: A quick "Great job keeping up with your meds!" style message.
        4. No "Hello", no "Subject", no markdown. Just the raw notification text.
        """

        user_prompt = f"""
        Status:
        - Pending: {adherence_data['pending']} (Names: {pending_list_str})
        - Taken: {adherence_data['taken']}
        - Missed: {adherence_data['missed']}
        
        Write the notification:
        """

        try:
            llm_response = await llm.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            notification_text = llm_response.content.strip()
        except Exception:
            # Fallback
            if adherence_data['pending'] > 0:
                notification_text = f"Reminder: You have {adherence_data['pending']} medicines left to take today."
            else:
                notification_text = "All caught up! Great job taking your medications today."

        return {
            "user_id": user_id,
            "date": today,
            "adherence_score": adherence_score,
            "adherence_data": adherence_data,
            "notifications": notifications,
            "notification_text": notification_text  # Renamed from 'summary' to be clear
        }

    except Exception as e:
        raise HTTPException(500, f"Follow-up agent error: {e}")