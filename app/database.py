import os
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

supabase: Client | None = None

def get_supabase():
    global supabase
    if supabase is None:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        supabase = create_client(url, key)
    return supabase
