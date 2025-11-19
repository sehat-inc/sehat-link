from typing import Optional
import json

NODE_STREAMING_MODE = {
    "frontend_agent": True, 
    # All agents that talk with user added here
}

def safe_str(x):
    if isinstance(x, str): 
        return x
    if hasattr(x, "content"):
        return str(x.content)
    elif hasattr(x, "text"):
        return str(x.text)
    else:
        return str(x)

def safe_int(value, default=None):
    if value is None:
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

def safe_list(value, default=None):
    if value is None:
        return default or []
    if isinstance(value, str):
        try:
            return json.loads(value)
        except:
            return default or []
    elif isinstance(value, list):
        return value
    else:
        return default or []

def safe_bool(value, default=False):
    if value is None:
        return default
    return bool(value)

# TODO: Add proper geolocation - i remember doing this before but now forgot
_CITY_TO_PROVINCE = {
    "karachi": "Sindh",
    "lahore": "Punjab",
    "faisalabad": "Punjab",
    "rawalpindi": "Punjab",
    "multan": "Punjab",
    "peshawar": "Khyber Pakhtunkhwa",
    "quetta": "Balochistan",
    "islamabad": "Islamabad Capital Territory",
}

def infer_province_from_city(city: Optional[str]) -> Optional[str]:
    if not city:
        return None
    key = city.strip().lower()
    return _CITY_TO_PROVINCE.get(key)


