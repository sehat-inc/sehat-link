import os
import hashlib
import bcrypt
from datetime import datetime, timedelta, timezone
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer
from jose import JWTError, jwt
from dotenv import load_dotenv

from models import TokenData

load_dotenv()

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "YOUR_SUPER_SECURE_SECRET_KEY_HERE")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 1 day

# --- Password Hashing ---
def hash_password(password: str) -> str:
    # Bcrypt has a 72-byte limit, use SHA-256 for longer passwords
    if len(password.encode('utf-8')) > 72:
        password = hashlib.sha256(password.encode('utf-8')).hexdigest()
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

def verify_password(plain: str, hashed: str) -> bool:
    # Bcrypt has a 72-byte limit, use SHA-256 for longer passwords
    if len(plain.encode('utf-8')) > 72:
        plain = hashlib.sha256(plain.encode('utf-8')).hexdigest()
    return bcrypt.checkpw(plain.encode('utf-8'), hashed.encode('utf-8'))

# --- JWT Handling ---
oauth2_scheme = HTTPBearer()  # use this for any user type

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire.timestamp()})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


# --- Token Validation ---
async def get_current_user_data(token: Annotated[str, Depends(oauth2_scheme)]) -> TokenData:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = str(payload.get("user_id"))
        user_type = str(payload.get("user_type"))
        if not user_id or not user_type:
            raise credentials_exception
        return TokenData(user_id=user_id, user_type=user_type)
    except JWTError:
        raise credentials_exception


# --- Role-Specific Helpers ---
def get_current_user_id(token_data: Annotated[TokenData, Depends(get_current_user_data)]) -> str:
    """Generic: works for any authenticated user."""
    return token_data.user_id

def get_current_doctor_id(token_data: Annotated[TokenData, Depends(get_current_user_data)]) -> str:
    """Restricts to doctors only."""
    if token_data.user_type != "doctor":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only doctors can access this endpoint.")
    return token_data.user_id

def get_current_patient_id(token_data: Annotated[TokenData, Depends(get_current_user_data)]) -> str:
    """Restricts to patients only."""
    if token_data.user_type != "patient":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only patients can access this endpoint.")
    return token_data.user_id
