from dotenv import load_dotenv
import os

load_dotenv()

print("OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY"))
print("OPENAI_PROJECT:", os.getenv("OPENAI_PROJECT"))
print("OPENAI_ORG:", os.getenv("OPENAI_ORG"))
