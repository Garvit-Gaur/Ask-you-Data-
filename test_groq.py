"""Standalone Groq connectivity test - run by itself, no Streamlit needed.
Usage:  python test_groq.py
"""
import os, time
from dotenv import load_dotenv
load_dotenv()

key = os.getenv("GROQ_API_KEY")
if not key or "paste_your" in key:
    print("GROQ_API_KEY is missing or still the placeholder in your .env file.")
    raise SystemExit(1)

print(f"Key starts with: {key[:8]}...  (should start with gsk_)")

from openai import OpenAI
client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=key)

# Step 1: ask Groq which models THIS key can actually see
print("\nFetching the list of models your key has access to...")
try:
    models = [m.id for m in client.models.list().data]
    print(f"Found {len(models)} accessible models:")
    for m in models:
        print(" -", m)
except Exception as e:
    print("Could not list models:", e)
    raise SystemExit(1)

if not models:
    print("\nYour key has access to ZERO models. The key or account has a problem - "
          "generate a fresh key at console.groq.com/keys and try again.")
    raise SystemExit(1)

# Step 2: try a real call using an actual model from that list
chat_models = [m for m in models if "whisper" not in m and "tts" not in m and "guard" not in m]
model = chat_models[0] if chat_models else models[0]
print(f"\nTesting a real request with: {model}")

start = time.time()
try:
    resp = client.chat.completions.create(
        model=model,
        max_tokens=20,
        messages=[{"role": "user", "content": "Say hello and nothing else."}],
    )
    print(f"SUCCESS in {time.time()-start:.1f}s")
    print("Response:", resp.choices[0].message.content)
    print(f"\n>>> Put this in your .env file:  MODEL_NAME={model}")
except Exception as e:
    print(f"FAILED after {time.time()-start:.1f}s")
    print("Error:", e)
