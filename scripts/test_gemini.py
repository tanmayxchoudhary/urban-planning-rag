from google import genai
from google.genai import types
from dotenv import load_dotenv
import os

# Load API key
load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')

if not api_key:
    raise ValueError("❌ GEMINI_API_KEY not found in .env file")

# Initialize client
client = genai.Client(api_key=api_key)

# List available models
print("Available Gemini models:")
for model in client.models.list():
    print(f"  ✅ {model.name}")

# Test with gemini-3-flash-preview
print("\n🧪 Testing gemini-3-flash-preview...")

response = client.models.generate_content(
    model='gemini-3-flash-preview',
    contents='Explain urban planning in 2 sentences.'
)

print("\n📝 Response:")
print(response.text)
print("\n✅ Gemini API working with new SDK")
