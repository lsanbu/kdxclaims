from supabase import create_client
from dotenv import load_dotenv
import os

load_dotenv()  # 👈 this line loads .env into environment variables

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

print("URL:", url[:30], "...")  # quick sanity check
client = create_client(url, key)

print(client.table("app_users").select("*").limit(1).execute())
