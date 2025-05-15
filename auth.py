from fastapi import Header, HTTPException, status
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OCR_API_KEY")

async def validate_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Invalid API key")
