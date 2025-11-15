from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
from datetime import date
from database import get_supabase
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()

router = APIRouter(prefix="/follow-up", tags=["follow-up"])

# Initialize LLM
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
llm = ChatGoogleGenerativeAI(
    google_api_key=GEMINI_API_KEY,
    model="gemini-2.5-flash",
    temperature=0.7,
    convert_system_message_to_human=True,
)


@router.get("/medicine-status/{user_id}")
async def check_medicine_status(user_id: int):
    """
    Reads medicine_data and daily_routine tables for a user and returns 
    personalized, LLM-generated notifications based on adherence behavior.
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
                "summary": "No medications found for this user.",
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

        # Build a map of medicine_id -> daily_routine record
        routine_map = {record["medicine_id"]: record for record in daily_records}

        # Analyze adherence
        notifications: List[Dict[str, Any]] = []
        adherence_data = {
            "total_meds": len(medicines),
            "taken": 0,
            "missed": 0,
            "late": 0,
            "pending": 0
        }

        for med in medicines:
            med_id = med.get("id")
            name = med.get("name", "Unknown medicine")
            dose = med.get("dose", "")
            frequency = med.get("frequency", "")
            
            routine = routine_map.get(med_id)

            if not routine:
                # No record for today → pending reminder
                adherence_data["pending"] += 1
                notifications.append({
                    "type": "reminder",
                    "medicine_id": med_id,
                    "medicine_name": name,
                    "dose": dose,
                    "frequency": frequency,
                    "status": "pending"
                })
            else:
                taken = routine.get("taken", False)
                not_taken = routine.get("not_taken", False)
                late_taken = routine.get("late_taken")

                if taken and not late_taken:
                    # Taken on time
                    adherence_data["taken"] += 1
                    notifications.append({
                        "type": "success",
                        "medicine_id": med_id,
                        "medicine_name": name,
                        "status": "taken_on_time"
                    })
                elif taken and late_taken:
                    # Taken late
                    adherence_data["late"] += 1
                    notifications.append({
                        "type": "late",
                        "medicine_id": med_id,
                        "medicine_name": name,
                        "late_time": str(late_taken),
                        "status": "taken_late"
                    })
                elif not_taken:
                    # Explicitly missed
                    adherence_data["missed"] += 1
                    notifications.append({
                        "type": "missed",
                        "medicine_id": med_id,
                        "medicine_name": name,
                        "status": "missed"
                    })
                else:
                    # Pending (record exists but no action)
                    adherence_data["pending"] += 1
                    notifications.append({
                        "type": "reminder",
                        "medicine_id": med_id,
                        "medicine_name": name,
                        "dose": dose,
                        "status": "pending"
                    })

        # Calculate adherence score
        total = adherence_data["total_meds"]
        adherence_score = round(
            ((adherence_data["taken"] + adherence_data["late"] * 0.5) / total * 100) if total > 0 else 0,
            1
        )

        # Generate personalized summary using LLM
        system_prompt = """You are a compassionate healthcare assistant helping patients with medication adherence.
Generate a brief, encouraging summary (2-3 sentences) based on the patient's medication adherence data.
Be supportive and motivating. If adherence is good, praise them. If there are issues, gently encourage improvement.
Keep it warm, personal, and actionable."""

        user_prompt = f"""Patient medication adherence today:
- Total medications: {adherence_data['total_meds']}
- Taken on time: {adherence_data['taken']}
- Taken late: {adherence_data['late']}
- Missed: {adherence_data['missed']}
- Pending: {adherence_data['pending']}
- Adherence score: {adherence_score}%

Generate a personalized, encouraging summary for the patient."""

        try:
            llm_response = await llm.ainvoke([
                ("system", system_prompt),
                ("human", user_prompt)
            ])
            summary = llm_response.content.strip()
        except Exception as e:
            # Fallback if LLM fails
            summary = f"Your adherence score today is {adherence_score}%. Keep up the good work!"

        return {
            "user_id": user_id,
            "date": today,
            "adherence_score": adherence_score,
            "adherence_data": adherence_data,
            "notifications": notifications,
            "summary": summary
        }

    except Exception as e:
        raise HTTPException(500, f"Follow-up agent error: {e}")
