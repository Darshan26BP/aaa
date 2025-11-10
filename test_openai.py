from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    organization=os.getenv("OPENAI_ORG"),
    project=os.getenv("OPENAI_PROJECT")
)

try:
    models = client.models.list()
    print("✅ Connection successful!")
    print("Available models:", [m.id for m in models.data[:5]])
except Exception as e:
    print("❌ Connection failed:", e)
