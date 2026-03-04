import json
import time
import os
import tempfile
import requests
from google import genai

api_key = os.environ.get("GOOGLE_API_KEY")
client = genai.Client(api_key=api_key)

req = {
    "request": {
        "contents": [{"role": "user", "parts": [{"text": "Hello, world!"}]}],
        "model": "models/gemini-3.1-pro-preview"
    }
}

with tempfile.NamedTemporaryFile('w', suffix='.jsonl', delete=False) as f:
    f.write(json.dumps(req) + "\n")
    path = f.name

f_obj = client.files.upload(file=path, config={'mime_type': 'application/jsonl'})
print("Uploaded file:", f_obj.uri)

job = client.batches.create(model="models/gemini-3.1-pro-preview", src=f_obj.name)
print("Job created:", job.name)

while True:
    job = client.batches.get(name=job.name)
    print("State:", job.state)
    if job.state in ['JOB_STATE_SUCCEEDED', 'JOB_STATE_FAILED', 'SUCCEEDED', 'FAILED']:
        break
    time.sleep(5)

print("Job done. Dest:", job.dest)
if job.dest and hasattr(job.dest, 'file_name'):
    dl_url = f"https://generativelanguage.googleapis.com/v1beta/{job.dest.file_name}?alt=media&key={api_key}"
    print("Download URL:", dl_url)
    res = requests.get(dl_url)
    print("Download status:", res.status_code)
    print("Content preview:", res.text[:200])

