
import os
import time
import logging
from abc import ABC, abstractmethod
from dotenv import load_dotenv

# Load .env from project root (parent directory of experiments/)
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(base_dir, '.env'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LLMProvider(ABC):
    @abstractmethod
    def generate(self, messages, model_name, temperature=0.7, max_tokens=None):
        pass

class GoogleProvider(LLMProvider):
    def __init__(self):
        from google import genai
        api_key = os.environ.get("GOOGLE_API_KEY")
        project = os.environ.get("GOOGLE_PROJECT_ID")
        location = os.environ.get("GOOGLE_LOCATION", "global")
        
        if project:
            # Vertex AI Mode
            self.client = genai.Client(vertexai=True, project=project, location=location)
            print(f"Initialized GoogleProvider in Vertex AI mode (project={project}, location={location})")
        else:
            # Gemini API (AI Studio) Mode
            if not api_key:
                raise ValueError("Neither GOOGLE_PROJECT_ID nor GOOGLE_API_KEY environment variable set.")
            self.client = genai.Client(api_key=api_key)

    def generate(self, messages, model_name, temperature=0.7, max_tokens=None, cached_content=None):
        if max_tokens is None:
            raise ValueError("max_tokens must be explicitly provided.")

        system_instruction = None
        contents = []
        
        for msg in messages:
            if msg['role'] == 'system':
                system_instruction = msg['content']
            elif msg['role'] == 'user':
                contents.append({'role': 'user', 'parts': [{'text': msg['content']}]})
            elif msg['role'] == 'assistant':
                contents.append({'role': 'model', 'parts': [{'text': msg['content']}]})
        
        from google.genai import types
        config_kwargs = dict(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        if system_instruction and not cached_content:
            config_kwargs['system_instruction'] = system_instruction
        if cached_content:
            config_kwargs['cached_content'] = cached_content
        config = types.GenerateContentConfig(**config_kwargs)
        
        if hasattr(self, 'client') and getattr(self.client, 'vertexai', False):
             full_model_name = model_name
        else:
             full_model_name = f"models/{model_name}" if not model_name.startswith("models/") else model_name

        response = self.client.models.generate_content(
            model=full_model_name,
            contents=contents,
            config=config
        )
        return response.text

class OpenAIProvider(LLMProvider):
    def __init__(self):
        from openai import OpenAI
        # OpenAI client automatically reads OPENAI_API_KEY and OPENAI_BASE_URL
        self.client = OpenAI()

    def generate(self, messages, model_name, temperature=0.7, max_tokens=None):
        kwargs = {
            "model": model_name,
            "messages": messages,
        }
        
        # Check if model supports standard parameters or requires new o1-style params
        is_o1 = "o1" in model_name.lower()
        
        if is_o1:
            # o1 models (beta) use 'max_completion_tokens' instead of 'max_tokens'
            # and may not support 'temperature' (fixed at 1.0 often)
            if max_tokens:
                kwargs["max_completion_tokens"] = max_tokens
            # Omit temperature for o1 unless supported in future
        else:
            # Standard ChatCompletion models (gpt-4o, gpt-3.5, etc.) and compatible APIs
            if max_tokens:
                kwargs["max_tokens"] = max_tokens
            kwargs["temperature"] = temperature
        
        response = self.client.chat.completions.create(**kwargs)
        
        message = response.choices[0].message
        content = message.content
        
        if not content:
            try:
                # OpenAI python lib use Pydantic
                if hasattr(message, 'model_dump'):
                    msg_dump = message.model_dump()
                elif hasattr(message, 'to_dict'):
                    msg_dump = message.to_dict()
                else:
                    msg_dump = str(message)
                
                # Check for DeepSeek reasoning (often in 'reasoning_content' or 'reasoning')
                reasoning = msg_dump.get('reasoning_content') or msg_dump.get('reasoning')
                if reasoning:
                     return f"<think>\n{reasoning}\n</think>\n[NO FINAL ANSWER GENERATED]"
            except Exception:
                pass
            return ""
        return content

class AnthropicProvider(LLMProvider):
    def __init__(self):
        from anthropic import Anthropic
        # Anthropic client automatically reads ANTHROPIC_API_KEY
        self.client = Anthropic()

    def generate(self, messages, model_name, temperature=0.7, max_tokens=None, cached_context_messages=None):
        system_prompt = ""
        anthro_messages = []
        
        for msg in messages:
            if msg['role'] == 'system':
                system_prompt = msg['content']
            else:
                anthro_messages.append(msg)

        # Prepend cached context if provided (already formatted with cache_control markers)
        if cached_context_messages:
            anthro_messages = cached_context_messages + anthro_messages
        
        response = self.client.messages.create(
            model=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=anthro_messages
        )
        return response.content[0].text

# Registry
PROVIDER_REGISTRY = {
    "google": GoogleProvider,
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "openai_compatible": OpenAIProvider  # Alias for generic use
}

def get_provider(provider_name):
    if provider_name not in PROVIDER_REGISTRY:
        raise ValueError(f"Unknown provider: {provider_name}. Available: {list(PROVIDER_REGISTRY.keys())}")
    return PROVIDER_REGISTRY[provider_name]()

def infer_provider(model_name):
    if "gemini" in model_name.lower():
        return "google"
    elif "claude" in model_name.lower():
        return "anthropic"
    elif "gpt" in model_name.lower() or "o1" in model_name.lower():
        return "openai"
    else:
        # Default to generic openai compatible if unknown? 
        # Or error? Let's check environment variables for hints? 
        # No, just return None and force user to specify or error.
        print(f"Could not infer provider from model name '{model_name}'. Please specify --provider.")
        return None

def generate_response(messages, model_name, provider=None, temperature=0.7, max_tokens=None, context_cache=None):
    """
    Generates a response from the specified model using the appropriate provider.
    context_cache: optional dict with keys 'type' ('google'|'anthropic') and 'ref' (cache name or messages list).
    """
    if max_tokens is None:
        raise ValueError("max_tokens must be explicitly provided.")
        
    # Normalize provider
    if provider:
        provider = provider.lower().strip()
    else:
        provider = infer_provider(model_name)
    
    if not provider:
        raise ValueError(f"Could not infer provider from model name '{model_name}'. Please specify --provider.")

    try:
        llm_provider = get_provider(provider)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize provider '{provider}': {e}")
    
    retries = 5
    base_delay = 2

    for attempt in range(retries):
        try:
            # Pass context cache to the appropriate provider method
            extra_kwargs = {}
            if context_cache:
                if context_cache['type'] == 'google' and provider == 'google':
                    extra_kwargs['cached_content'] = context_cache['ref']
                elif context_cache['type'] == 'anthropic' and provider == 'anthropic':
                    extra_kwargs['cached_context_messages'] = context_cache['ref']
            return llm_provider.generate(messages, model_name, temperature, max_tokens, **extra_kwargs)
        except Exception as e:
            logging.warning(f"Attempt {attempt + 1}/{retries} failed for {provider}/{model_name}: {e}")
            if hasattr(e, 'status_code') and e.status_code == 429: # Rate limit
                time.sleep(base_delay * (2 ** attempt))
            elif "429" in str(e):
                 time.sleep(base_delay * (2 ** attempt))
            else:
                time.sleep(base_delay)
    
    raise RuntimeError(f"Failed to generate response after {retries} retries.")


# =====================================================================
# CONTEXT CACHING UTILITIES
# =====================================================================

def load_context_messages(context_file_path):
    """
    Loads a context JSON file. Returns (system_prompt_str, user_assistant_messages_list).
    The system message is extracted and returned separately; the rest are user/assistant pairs.
    """
    with open(context_file_path, 'r') as f:
        messages = json.load(f)
    system_prompt = None
    conversation = []
    for msg in messages:
        if msg['role'] == 'system':
            system_prompt = msg['content']
        else:
            conversation.append(msg)
    return system_prompt, conversation


def create_google_context_cache(context_file_path, model_name, ttl_seconds=3600):
    """
    Creates an explicit Google AI Studio context cache from a context file.
    Returns the cache name string (e.g. 'cachedContents/abc123').
    Requires GOOGLE_API_KEY to be set (AI Studio mode).
    """
    from google import genai
    from google.genai import types
    api_key = os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError('GOOGLE_API_KEY must be set to use Google context caching (AI Studio mode).')
    client = genai.Client(api_key=api_key)

    system_prompt, conversation = load_context_messages(context_file_path)

    # Convert to Gemini Content format
    contents = []
    for msg in conversation:
        role = 'model' if msg['role'] == 'assistant' else 'user'
        contents.append(types.Content(role=role, parts=[types.Part(text=msg['content'])]))

    full_model_name = f"models/{model_name}" if not model_name.startswith("models/") else model_name

    create_config = types.CreateCachedContentConfig(
        contents=contents,
        ttl=f"{ttl_seconds}s",
    )
    if system_prompt:
        create_config = types.CreateCachedContentConfig(
            contents=contents,
            system_instruction=system_prompt,
            ttl=f"{ttl_seconds}s",
        )

    cache = client.caches.create(model=full_model_name, config=create_config)
    print(f"Created Google context cache: {cache.name} (model={model_name}, ttl={ttl_seconds}s, messages={len(contents)})")
    return cache.name


def prepare_anthropic_cached_messages(context_file_path):
    """
    Loads a context JSON file and returns user/assistant messages formatted for Anthropic
    with cache_control markers on the last message (marking the entire prefix as cached).
    """
    _, conversation = load_context_messages(context_file_path)

    if not conversation:
        return []

    formatted = []
    for i, msg in enumerate(conversation):
        is_last = (i == len(conversation) - 1)
        content_block = {"type": "text", "text": msg['content']}
        if is_last:
            content_block["cache_control"] = {"type": "ephemeral"}
        formatted.append({"role": msg['role'], "content": [content_block]})

    return formatted


def create_google_context_cache_from_messages(messages, model_name, ttl_seconds=3600):
    """
    Like create_google_context_cache but accepts a pre-loaded list of {role, content} messages
    (already trimmed/processed). The first 'system' message is used as system_instruction.
    """
    from google import genai
    from google.genai import types
    api_key = os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError('GOOGLE_API_KEY must be set to use Google context caching (AI Studio mode).')
    client = genai.Client(api_key=api_key)

    system_prompt = None
    conversation = []
    for msg in messages:
        if msg['role'] == 'system':
            system_prompt = msg['content']
        else:
            conversation.append(msg)

    contents = []
    for msg in conversation:
        role = 'model' if msg['role'] == 'assistant' else 'user'
        contents.append(types.Content(role=role, parts=[types.Part(text=msg['content'])]))

    full_model_name = f"models/{model_name}" if not model_name.startswith("models/") else model_name

    config_kwargs = dict(contents=contents, ttl=f"{ttl_seconds}s")
    if system_prompt:
        config_kwargs['system_instruction'] = system_prompt
    cache = client.caches.create(model=full_model_name, config=types.CreateCachedContentConfig(**config_kwargs))
    print(f"Created Google context cache: {cache.name} (model={model_name}, ttl={ttl_seconds}s, {len(contents)} messages)")
    return cache.name


def prepare_anthropic_cached_messages_from_list(messages):
    """
    Like prepare_anthropic_cached_messages but accepts a pre-loaded list of {role, content} messages.
    System messages are skipped (pass separately as the system param). cache_control is set on the last message.
    """
    conversation = [m for m in messages if m['role'] != 'system']
    if not conversation:
        return []
    formatted = []
    for i, msg in enumerate(conversation):
        is_last = (i == len(conversation) - 1)
        content_block = {"type": "text", "text": msg['content']}
        if is_last:
            content_block["cache_control"] = {"type": "ephemeral"}
        formatted.append({"role": msg['role'], "content": [content_block]})
    return formatted


# --- BATCH PROVIDERS ---

import json
import requests
import tempfile

class BatchProvider(ABC):
    @abstractmethod
    def create_batch(self, jobs, model_name, max_tokens=None, temperature=0.7):
        """
        Takes a list of job dicts `[{id, messages}, ...]`,
        formats them for the specific provider, uploads, creates batch,
        and returns a dict: {"batch_id": ..., "status": ...}
        """
        pass

class OpenAIBatchProvider(BatchProvider):
    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI()

    def create_batch(self, jobs, model_name, max_tokens=None, temperature=0.7):
        # 1. Create JSONL
        is_o1 = "o1" in model_name.lower()
        jsonl_lines = []
        for job in jobs:
            body = {
                "model": model_name,
                "messages": job['messages'],
            }
            if is_o1:
                if max_tokens: body["max_completion_tokens"] = max_tokens
            else:
                if max_tokens: body["max_tokens"] = max_tokens
                body["temperature"] = temperature
                
            req = {
                "custom_id": str(job['id']) + "_" + str(job.get('sample_idx', 0)),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body
            }
            jsonl_lines.append(json.dumps(req))
            
        # 2. Upload file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('\n'.join(jsonl_lines))
            tmp_path = f.name
            
        print(f"Uploading {len(jobs)} jobs to OpenAI...")
        with open(tmp_path, "rb") as f:
            file_obj = self.client.files.create(file=f, purpose="batch")
            
        os.remove(tmp_path)
        
        # 3. Create batch
        print(f"Creating OpenAI batch with file_id {file_obj.id}...")
        batch = self.client.batches.create(
            input_file_id=file_obj.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        return {"batch_id": batch.id, "file_id": file_obj.id, "status": batch.status, "provider": "openai"}

class AnthropicBatchProvider(BatchProvider):
    def __init__(self):
        from anthropic import Anthropic
        self.client = Anthropic()

    def create_batch(self, jobs, model_name, max_tokens=None, temperature=0.7, context_cache=None):
        requests_list = []
        for job in jobs:
            system_prompt = ""
            anthro_messages = []
            for msg in job['messages']:
                if msg['role'] == 'system':
                    system_prompt = msg['content']
                else:
                    anthro_messages.append(msg)

            # Prepend cached context messages if provided
            if context_cache and context_cache.get('type') == 'anthropic':
                anthro_messages = context_cache['ref'] + anthro_messages
                    
            req = {
                "custom_id": str(job['id']) + "_" + str(job.get('sample_idx', 0)),
                "params": {
                    "model": model_name,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "system": system_prompt,
                    "messages": anthro_messages
                }
            }
            requests_list.append(req)
            
        print(f"Creating Anthropic message batch with {len(jobs)} jobs...")
        batch = self.client.messages.batches.create(
            requests=requests_list
        )
        return {"batch_id": batch.id, "status": batch.processing_status, "provider": "anthropic"}

class GoogleBatchProvider(BatchProvider):
    def __init__(self):
        from google import genai
        self.api_key = os.environ.get("GOOGLE_API_KEY")
        self.project = os.environ.get("GOOGLE_PROJECT_ID")
        self.location = os.environ.get("GOOGLE_LOCATION", "global")
        self.gcs_bucket = os.environ.get("GOOGLE_GCS_BUCKET")
        
        # We delay client initialization until create_batch because the model might dictate the mode
        self.client = None

    def _init_client(self, model_name):
        from google import genai
        
        if self.project:
            # Vertex AI Mode
            self.client = genai.Client(vertexai=True, project=self.project, location=self.location)
            self.is_vertex = True
            print(f"Initialized GoogleBatchProvider in Vertex AI mode (project={self.project}, location={self.location})")
        else:
            # Gemini API (AI Studio) Mode
            if not self.api_key:
                raise ValueError("Neither GOOGLE_PROJECT_ID nor GOOGLE_API_KEY environment variable set.")
            self.client = genai.Client(api_key=self.api_key)
            self.is_vertex = False
            print("Initialized GoogleBatchProvider in AI Studio mode")

    def create_batch(self, jobs, model_name, max_tokens=None, temperature=0.7):
        if self.client is None:
            self._init_client(model_name)
            
        if max_tokens is None:
            raise ValueError("max_tokens must be explicitly provided.")

        import tempfile
        import json
        
        jsonl_lines = []
        for job in jobs:
            system_instruction = None
            contents = []
            for msg in job['messages']:
                if msg['role'] == 'system':
                    system_instruction = msg['content']
                elif msg['role'] == 'user':
                    contents.append({'role': 'user', 'parts': [{'text': msg['content']}]})
                elif msg['role'] == 'assistant':
                    contents.append({'role': 'model', 'parts': [{'text': msg['content']}]})
            
            # Formatting for Google Batch API
            body = {
                "contents": contents,
                "generation_config": {
                    "temperature": temperature,
                    "max_output_tokens": max_tokens
                }
            }
            if system_instruction:
                body["system_instruction"] = {"parts": [{"text": system_instruction}]}
                
            req = {
                "request": body,
                "id": str(job['id']) + "_" + str(job.get('sample_idx', 0))
            }
            jsonl_lines.append(json.dumps(req))
            
        print(f"Preparing batch file...")
        
        # Vertex AI mode needs the source file to be in GCS
        if self.is_vertex:
            if not self.gcs_bucket:
                raise ValueError("GOOGLE_GCS_BUCKET must be set when using Vertex AI mode (GOOGLE_PROJECT_ID).")
            
            from google.cloud import storage
            storage_client = storage.Client()
            bucket_name = self.gcs_bucket.replace("gs://", "").strip("/")
            bucket = storage_client.bucket(bucket_name)
            
            # Use a unique name for the input file in GCS
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            blob_name = f"input/batch_{timestamp}.jsonl"
            blob = bucket.blob(blob_name)
            
            print(f"Uploading batch file to GCS: gs://{bucket_name}/{blob_name}")
            blob.upload_from_string('\n'.join(jsonl_lines), content_type='application/jsonl')
            src_uri = f"gs://{bucket_name}/{blob_name}"
            
        else:
            # Gemini API (AI Studio) mode uses the files API
            with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
                f.write('\n'.join(jsonl_lines))
                tmp_path = f.name
            
            print(f"Uploading batch file to Google...")
            try:
                file_obj = self.client.files.upload(file=tmp_path, config={'mime_type': 'application/jsonl'})
                src_uri = file_obj.name
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        
        print(f"Creating Google batch job using new SDK...")
        if self.is_vertex:
            # Vertex AI batch prediction requires the publishers/google/models/ prefix
            if not model_name.startswith("publishers/") and not model_name.startswith("projects/"):
                full_model_name = f"publishers/google/models/{model_name}"
            else:
                full_model_name = model_name
        else:
            # AI Studio likes the models/ prefix
            full_model_name = f"models/{model_name}" if not model_name.startswith("models/") else model_name
        
        create_kwargs = {
            "model": full_model_name,
            "src": src_uri
        }
        
        if self.is_vertex and self.gcs_bucket:
            # Ensure bucket name doesn't have gs:// prefix for the config if the user added it by mistake
            bucket_path = self.gcs_bucket.replace("gs://", "").strip("/")
            # Use a unique subfolder for each job to avoid collisions
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            unique_dest = f"gs://{bucket_path}/results_{timestamp}/"
            from google.genai import types
            create_kwargs["config"] = types.CreateBatchJobConfig(
                dest=unique_dest
            )
            print(f"Using GCS destination: {unique_dest}")

        batch_job = self.client.batches.create(**create_kwargs)
        
        return {
            "batch_id": batch_job.name, 
            "file_uri": src_uri, 
            "status": str(batch_job.state), 
            "provider": "google"
        }

BATCH_PROVIDER_REGISTRY = {
    "google": GoogleBatchProvider,
    "openai": OpenAIBatchProvider,
    "anthropic": AnthropicBatchProvider,
    "openai_compatible": OpenAIBatchProvider
}

# Separate registry for providers that support context caching via AI Studio.
# Used when context_cache is provided for Google.
CACHED_BATCH_PROVIDER_REGISTRY = {
    "google": "GoogleAIStudioBatchProvider",  # resolved below after class definition
    "anthropic": AnthropicBatchProvider,
}

class GoogleAIStudioBatchProvider(BatchProvider):
    """
    Google Batch Provider using AI Studio (Gemini API / API key) mode.
    Supports explicit context caching via cachedContent in request bodies.
    Results are retrieved via the SDK files API (no GCS).
    """
    def __init__(self):
        from google import genai
        self.api_key = os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY must be set to use Google AI Studio batch (context caching path).")
        self.client = genai.Client(api_key=self.api_key)

    def create_batch(self, jobs, model_name, max_tokens=None, temperature=0.7, context_cache=None):
        if max_tokens is None:
            raise ValueError("max_tokens must be explicitly provided.")

        cached_content_name = None
        if context_cache and context_cache.get('type') == 'google':
            cached_content_name = context_cache['ref']
            print(f"Using Google context cache: {cached_content_name}")

        full_model_name = f"models/{model_name}" if not model_name.startswith("models/") else model_name

        jsonl_lines = []
        for job in jobs:
            system_instruction = None
            contents = []
            for msg in job['messages']:
                if msg['role'] == 'system':
                    system_instruction = msg['content']
                elif msg['role'] == 'user':
                    contents.append({'role': 'user', 'parts': [{'text': msg['content']}]})
                elif msg['role'] == 'assistant':
                    contents.append({'role': 'model', 'parts': [{'text': msg['content']}]})

            body = {
                "contents": contents,
                "generation_config": {
                    "temperature": temperature,
                    "max_output_tokens": max_tokens
                }
            }
            # If using a cache, the system prompt is embedded in the cache; skip it here.
            if cached_content_name:
                body["cached_content"] = cached_content_name
            elif system_instruction:
                body["system_instruction"] = {"parts": [{"text": system_instruction}]}

            req = {
                "request": body,
                "id": str(job['id']) + "_" + str(job.get('sample_idx', 0))
            }
            jsonl_lines.append(json.dumps(req))

        print(f"Preparing AI Studio batch file ({len(jobs)} jobs)...")
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('\n'.join(jsonl_lines))
            tmp_path = f.name

        try:
            file_obj = self.client.files.upload(file=tmp_path, config={'mime_type': 'application/jsonl'})
            src_uri = file_obj.name
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        print(f"Creating Google AI Studio batch job...")
        batch_job = self.client.batches.create(model=full_model_name, src=src_uri)

        return {
            "batch_id": batch_job.name,
            "file_uri": src_uri,
            "status": str(batch_job.state),
            "provider": "google",
            "google_mode": "ai_studio",
        }


# Update the cached batch registry now that GoogleAIStudioBatchProvider is defined
CACHED_BATCH_PROVIDER_REGISTRY["google"] = GoogleAIStudioBatchProvider


def submit_batch(jobs, model_name, provider=None, temperature=0.7, max_tokens=None, context_cache=None):
    if max_tokens is None:
        raise ValueError("max_tokens must be explicitly provided.")
        
    if provider:
        provider = provider.lower().strip()
    else:
        provider = infer_provider(model_name)
        
    if not provider:
        raise ValueError(f"Could not infer provider from model name '{model_name}'.")

    # Choose the right registry: if a context cache is requested and this provider
    # supports a cached mode, use the cached batch provider.
    if context_cache and provider in CACHED_BATCH_PROVIDER_REGISTRY:
        registry = CACHED_BATCH_PROVIDER_REGISTRY
    else:
        registry = BATCH_PROVIDER_REGISTRY
        
    if provider not in registry:
        raise ValueError(f"Batch not supported for provider: {provider}")
        
    batch_provider = registry[provider]()
    if context_cache is not None and registry is CACHED_BATCH_PROVIDER_REGISTRY:
        return batch_provider.create_batch(jobs, model_name, max_tokens, temperature, context_cache=context_cache)
    else:
        return batch_provider.create_batch(jobs, model_name, max_tokens, temperature)
