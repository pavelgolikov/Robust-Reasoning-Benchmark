
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
    def generate(self, messages, model_name, temperature=0.7, max_tokens=1000):
        pass

class GoogleProvider(LLMProvider):
    def __init__(self):
        import google.generativeai as genai
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set.")
        genai.configure(api_key=api_key)
        self.genai = genai

    def generate(self, messages, model_name, temperature=0.7, max_tokens=1000):
        system_instruction = None
        contents = []
        
        for msg in messages:
            if msg['role'] == 'system':
                system_instruction = msg['content']
            elif msg['role'] == 'user':
                contents.append({'role': 'user', 'parts': [msg['content']]})
            elif msg['role'] == 'assistant':
                contents.append({'role': 'model', 'parts': [msg['content']]})
        
        model = self.genai.GenerativeModel(model_name, system_instruction=system_instruction)
        
        response = model.generate_content(
            contents,
            generation_config=self.genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens
            )
        )
        return response.text

class OpenAIProvider(LLMProvider):
    def __init__(self):
        from openai import OpenAI
        # OpenAI client automatically reads OPENAI_API_KEY and OPENAI_BASE_URL
        self.client = OpenAI()

    def generate(self, messages, model_name, temperature=0.7, max_tokens=1000):
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

    def generate(self, messages, model_name, temperature=0.7, max_tokens=1000):
        system_prompt = ""
        anthro_messages = []
        
        for msg in messages:
            if msg['role'] == 'system':
                system_prompt = msg['content']
            else:
                anthro_messages.append(msg)
        
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

def generate_response(messages, model_name, provider=None, temperature=0.7, max_tokens=1000):
    """
    Generates a response from the specified model using the appropriate provider.
    """
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
            return llm_provider.generate(messages, model_name, temperature, max_tokens)
        except Exception as e:
            logging.warning(f"Attempt {attempt + 1}/{retries} failed for {provider}/{model_name}: {e}")
            if hasattr(e, 'status_code') and e.status_code == 429: # Rate limit
                time.sleep(base_delay * (2 ** attempt))
            elif "429" in str(e):
                 time.sleep(base_delay * (2 ** attempt))
            else:
                time.sleep(base_delay)
    
    raise RuntimeError(f"Failed to generate response after {retries} retries.")
