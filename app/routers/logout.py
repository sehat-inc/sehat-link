from typing import Annotated, Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Depends
import os
import redis.asyncio as redis

from auth_utils import get_current_user_id
from database import get_supabase

from core.logging import get_logger

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

logger = get_logger("PATIENT LOGOUT ROUTER")

router = APIRouter(prefix="/user",tags=["user"])

async def get_redis_state_for_user(user_id: int) -> Optional[Dict[str, Any]]:
    """
    Get the current MedicalAgentState from Redis for a user.
    Returns None if no state exists.
    """
    redis_client = await redis.from_url(REDIS_URL, decode_responses=False)
    
    try:
        # LangGraph stores checkpoints with pattern: checkpoint:user_{id}:*
        # We need to find the latest checkpoint
        pattern = f"checkpoint:user_{user_id}:*"
        keys = []
        
        async for key in redis_client.scan_iter(match=pattern):
            keys.append(key)
        
        if not keys:
            logger.info(f"No Redis state found for user {user_id}")
            return None
        
        logger.info(f"Found {len(keys)} checkpoint keys for user {user_id}")
        
        # Try to get a checkpoint key (not checkpoint_write)
        for key in keys:
            key_str = key.decode() if isinstance(key, bytes) else key
            if "__empty__" in key_str and "checkpoint_write" not in key_str:
                data = await redis_client.get(key)
                if data:
                    try:
                        import pickle
                        state_data = pickle.loads(data)
                        
                        # LangGraph checkpoint structure
                        if isinstance(state_data, dict):
                            if 'values' in state_data:
                                logger.info(f"✅ Successfully extracted state from {key_str}")
                                return state_data['values']
                            elif 'channel_values' in state_data:
                                logger.info(f"✅ Successfully extracted state from {key_str}")
                                return state_data['channel_values']
                        elif hasattr(state_data, '__dict__'):
                            if hasattr(state_data, 'values') and not callable(state_data.values):
                                logger.info(f"✅ Successfully extracted state from {key_str}")
                                return state_data.values
                    except Exception as e:
                        logger.warning(f"Failed to parse key {key_str}: {e}")
                        continue
        
        logger.warning(f"Could not parse Redis state for user {user_id}")
        return None
        
    except Exception as e:
        logger.error(f"Error getting Redis state: {e}")
        return None
    finally:
        await redis_client.aclose()


async def save_state_to_longterm_session(user_id: int, state: Dict[str, Any]):
    """
    Save MedicalAgentState to longterm_session table in Supabase.
    Updates existing record or creates new one.
    """
    supabase = get_supabase()
    
    # Prepare data for longterm_session table
    session_data = {
        "user_id": user_id,
        "user_name": state.get("user_name", ""),
        "user_age": state.get("user_age"),
        "user_gender": state.get("user_gender"),
        "user_phone": state.get("user_phone"),
        "preferred_language": state.get("preferred_language", "en"),
        "user_location": state.get("user_location"),
        "user_domicile_location": state.get("user_domicile_location"),
        
        # Medical history
        "chronic_conditions": state.get("chronic_conditions", []),
        "allergies": state.get("allergies", []),
        "current_medications": state.get("current_medications", []),
        
        # Detected context
        "detected_urgency": state.get("detected_urgency"),
        "detected_problem_type": state.get("detected_problem_type"),
        
        # Symptoms
        "symptoms_collected": state.get("symptoms_collected", []),
        "symptoms_summary": state.get("symptoms_summary"),
        
        # Research results
        "symptom_research_result": state.get("symptom_research_result", []),
        "similar_cases": state.get("similar_cases", []),
        
        # Shared knowledge
        "shared_facts": state.get("shared_facts", []),
        "shared_warnings": state.get("shared_warnings", []),
        "red_flags": state.get("red_flags", []),
        
        # Prescription
        "prescription_data": state.get("prescription_data")
    }
    
    # Remove None values to avoid overwriting with null
    session_data = {k: v for k, v in session_data.items() if v is not None}
    
    try:
        # Check if record exists
        existing = supabase.table("longterm_session").select("user_id").eq("user_id", user_id).execute()
        
        if existing.data:
            # Update existing record
            response = supabase.table("longterm_session").update(session_data).eq("user_id", user_id).execute()
            logger.info(f"✅ Updated longterm_session for user {user_id}")
        else:
            # Insert new record
            response = supabase.table("longterm_session").insert(session_data).execute()
            logger.info(f"✅ Inserted longterm_session for user {user_id}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error saving to longterm_session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save session data: {str(e)}")


async def delete_redis_keys_for_user(user_id: int) -> int:
    """
    Delete all Redis keys for a user.
    Returns count of deleted keys.
    """
    redis_client = await redis.from_url(REDIS_URL, decode_responses=False)
    
    try:
        # Find all keys for this user - use broader pattern
        pattern = f"*user_{user_id}*"
        keys = []
        
        async for key in redis_client.scan_iter(match=pattern):
            keys.append(key)
        
        if not keys:
            logger.info(f"No Redis keys found for user {user_id}")
            return 0
        
        logger.info(f"Found {len(keys)} Redis keys to delete for user {user_id}")
        
        # Delete all keys at once
        deleted_count = await redis_client.delete(*keys)
        logger.info(f"✅ Deleted {deleted_count} Redis keys for user {user_id}")
        
        return deleted_count
        
    except Exception as e:
        logger.error(f"Error deleting Redis keys: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear session: {str(e)}")
    finally:
        await redis_client.aclose()


@router.post("/logout")
async def logout(current_user_id: Annotated[str, Depends(get_current_user_id)]):
    """
    Logout endpoint that:
    1. Gets current MedicalAgentState from Redis
    2. Saves it to longterm_session table in Supabase
    3. Deletes all Redis keys for the user
    4. Returns success with cleanup stats
    """
    user_id = int(current_user_id)
    
    logger.info(f"🚪 Logout initiated for user {user_id}")
    
    try:
        # Step 1: Get current state from Redis
        redis_state = await get_redis_state_for_user(user_id)
        
        if redis_state:
            logger.info(f"📊 Found Redis state for user {user_id}")
            
            # Step 2: Save to supabase table
            await save_state_to_longterm_session(user_id, redis_state)
        else:
            logger.info(f"ℹ️  No Redis state found for user {user_id}, skipping save")
        
        # Step 3: Delete all Redis keys
        deleted_count = await delete_redis_keys_for_user(user_id)
        
        logger.info(f"✅ Logout completed for user {user_id}: {deleted_count} keys deleted")
        
        return {
            "success": True,
            "message": "Logged out successfully",
            "user_id": user_id,
            "state_saved": redis_state is not None,
            "redis_keys_deleted": deleted_count
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Logout error for user {user_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Logout failed: {str(e)}"
        )

@router.post("/new-chat")
async def newchat(current_user_id: Annotated[str, Depends(get_current_user_id)]):
    """
    Newchat endpoint that:
    1. Deletes all Redis keys for the user
    2. Returns success with cleanup stats
    """
    user_id = int(current_user_id)
    
    logger.info(f"🚪 New chat initiated for user {user_id}")
    
    try:
        # Step 1: Get current state from Redis
        redis_state = await get_redis_state_for_user(user_id)
        
        if redis_state:
            logger.info(f"📊 Found Redis state for user {user_id}")
        else:
            logger.info(f"ℹ️  No Redis state found for user {user_id}, skipping save")
        
        # Step 2: Delete all Redis keys
        deleted_count = await delete_redis_keys_for_user(user_id)
        
        logger.info(f"✅ New-chat transition completed for user {user_id}: {deleted_count} keys deleted")
        
        return {
            "success": True,
            "message": "New-chat Transition successfull",
            "user_id": user_id,
            "state_saved": redis_state is not None,
            "redis_keys_deleted": deleted_count
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Newchat transition error for user {user_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start new chat :{str(e)}"
        )
