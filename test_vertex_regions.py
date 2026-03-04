import os
from google import genai

project = os.environ.get('GOOGLE_PROJECT_ID', 'gen-lang-client-0380290667')

for location in ['us-central1', 'us-east5', 'us-east1', 'europe-west1', 'europe-west4']:
    print(f"\nTrying location: {location}")
    try:
        client = genai.Client(vertexai=True, project=project, location=location)
        response = client.models.generate_content(
            model='gemini-3.1-pro-preview',
            contents='Hi'
        )
        print("Success! Response:", response.text)
    except Exception as e:
        print("Error:", str(e))
