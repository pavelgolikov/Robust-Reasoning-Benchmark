#!/usr/bin/env python3
"""Delete a Google context cache by name.

Usage:
  python delete_cache.py projects/1077084806582/locations/global/cachedContents/4628864994258190336
"""
import os
import sys
from dotenv import load_dotenv
from google import genai

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))

if len(sys.argv) < 2:
    print("Usage: python delete_cache.py <cache_name>")
    sys.exit(1)

cache_name = sys.argv[1]

project = os.environ.get("GOOGLE_PROJECT_ID")
location = os.environ.get("GOOGLE_LOCATION", "global")
api_key = os.environ.get("GOOGLE_API_KEY")

if project:
    client = genai.Client(vertexai=True, project=project, location=location)
elif api_key:
    client = genai.Client(api_key=api_key)
else:
    print("Error: set GOOGLE_PROJECT_ID or GOOGLE_API_KEY")
    sys.exit(1)

client.caches.delete(name=cache_name)
print("Deleted:", cache_name)