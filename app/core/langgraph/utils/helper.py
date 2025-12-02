from typing import Optional
import json
import re
from langchain_core.messages import HumanMessage, ToolMessage

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


def extract_llm_content(llm_response) -> str:
    """
    Extract only the text content from an LLM response, removing metadata, signatures, and extras.
    This cleans up the raw LLM response object to just get the useful text.
    """
    if not hasattr(llm_response, 'content'):
        return str(llm_response)
    
    content = llm_response.content
    
    if isinstance(content, str):
        return content
    
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                # Extract text, ignore 'extras' with signatures
                if 'text' in item:
                    text_parts.append(item['text'])
            elif isinstance(item, str):
                text_parts.append(item)
        return ''.join(text_parts)
    
    return str(content)


def extract_user_message(messages: list) -> str:
    """
    Extract the actual user message from message history.
    If the last message is a ToolMessage (tool output), find the actual last human message.
    If it's a tool output JSON, return "[Tool was called]" to avoid bloating the prompt.
    """
    if not messages:
        return ""
    
    last_msg = messages[-1]
    
    # If it's a ToolMessage, find the last actual HumanMessage
    if isinstance(last_msg, ToolMessage):
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                content = msg.content
                if isinstance(content, str):
                    # Check if it looks like tool output
                    if _is_tool_output(content):
                        return "[Tool was called]"
                    return content
                elif isinstance(content, list):
                    texts = []
                    for item in content:
                        if isinstance(item, dict) and 'text' in item:
                            texts.append(item['text'])
                        elif isinstance(item, str):
                            texts.append(item)
                    result = ''.join(texts)
                    if _is_tool_output(result):
                        return "[Tool was called]"
                    return result
        return "[Tool was called]"
    
    # Regular message extraction
    content = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
    
    if isinstance(content, str):
        if _is_tool_output(content):
            return "[Tool was called]"
        return content
    
    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict) and 'text' in item:
                texts.append(item['text'])
            elif isinstance(item, str):
                texts.append(item)
        result = ''.join(texts)
        if _is_tool_output(result):
            return "[Tool was called]"
        return result
    
    return str(content)


def _is_tool_output(content: str) -> bool:
    """
    Check if content looks like a tool output JSON (strategy/results/sub_queries pattern).
    """
    if not content:
        return False
    stripped = content.strip()
    if stripped.startswith('{'):
        # Common tool output patterns
        tool_indicators = ['"strategy"', '"results"', '"sub_queries"', '"original_question"']
        return any(indicator in content for indicator in tool_indicators)
    return False

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


