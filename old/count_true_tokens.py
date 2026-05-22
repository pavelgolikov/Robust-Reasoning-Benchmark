import os
import json
import asyncio
from anthropic import AsyncAnthropic
from google import genai
from openai import AsyncOpenAI
import tiktoken
from dotenv import load_dotenv
import sys

load_dotenv()

async def measure_tokens(filepath):
    print(f"\\n{'='*50}")
    print(f"Token Counting: {os.path.basename(filepath)}")
    print(f"{'='*50}")

    with open(filepath, 'r') as f:
        messages = json.load(f)
        
    raw_text = "".join([m['content'] for m in messages])

    # 1. OpenAI (Newest Model Tokenizer)
    try:
        enc = tiktoken.get_encoding("o200k_base")
        print(f"OpenAI (o200k_base): {len(enc.encode(raw_text)):,} tokens")
    except Exception as e:
        print(f"OpenAI error: {e}")

    # 2. Anthropic (Claude Opus 4.6 native API)
    try:
        client = AsyncAnthropic(default_headers={"anthropic-beta": "context-1m-2025-08-07"})
        anthro_msgs = []
        for m in messages:
            role = m["role"] if m["role"] != "system" else "user"
            content = m["content"].rstrip() if m["role"] == "assistant" else m["content"]
            anthro_msgs.append({"role": role, "content": content})
        
        resp = await client.beta.messages.count_tokens(
            betas=["token-counting-2024-11-01"],
            model="claude-opus-4-6",
            messages=anthro_msgs
        )
        print(f"Anthropic (claude-opus-4-6): {resp.input_tokens:,} tokens")
    except Exception as e:
        print(f"Anthropic API error: {e}")

    # 3. Google (Gemini 3.1 Pro Preview native API)
    try:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            print("Google API error: GOOGLE_API_KEY not set")
        else:
            client = genai.Client(api_key=api_key)
            contents = []
            for m in messages:
                role = "model" if m["role"] == "assistant" else "user"
                contents.append({"role": role, "parts": [{"text": m["content"]}]})
            
            resp = client.models.count_tokens(
                model='gemini-3.1-pro-preview',
                contents=contents,
            )
            print(f"Google (gemini-3.1-pro-preview): {resp.total_tokens:,} tokens")
    except Exception as e:
        print(f"Google API error: {e}")

async def measure_tokens_native_check(provider, model_name, raw_messages):
    """
    Programmatic entrypoint for evaluate_context_api.py to check true token counts dynamically.
    Returns the exact int token count for the given provider and model, or -1 if auth fails.
    """
    try:
        if provider == "openai":
            text = "".join(m['content'] for m in raw_messages)
            try:
                enc = tiktoken.encoding_for_model(model_name)
            except Exception:
                # If the exact model name is unrecognized by tiktoken (e.g. gpt-5.4),
                # fallback to the latest standard encoding.
                enc = tiktoken.get_encoding("o200k_base")
            return len(enc.encode(text))
            
        elif provider == "anthropic":
            client = AsyncAnthropic(default_headers={"anthropic-beta": "context-1m-2025-08-07"})
            formatted = []
            for m in raw_messages:
                role = m["role"] if m["role"] != "system" else "user"
                content = m["content"].rstrip() if m["role"] == "assistant" else m["content"]
                formatted.append({"role": role, "content": content})
            resp = await client.beta.messages.count_tokens(
                betas=["token-counting-2024-11-01"],
                model=model_name,
                messages=formatted
            )
            return resp.input_tokens
            
        elif provider == "google":
            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                return -1
            client = genai.Client(api_key=api_key)
            contents = []
            for m in raw_messages:
                role = "model" if m["role"] == "assistant" else "user"
                contents.append({"role": role, "parts": [{"text": m["content"]}]})
            resp = client.models.count_tokens(
                model=model_name,
                contents=contents,
            )
            return resp.total_tokens
    except Exception as e:
        import sys
        sys.stderr.write(f"Token count error native: {e}\n")
        return -1
    return -1

async def main(filename):
    # base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # math_1m = os.path.join(base_dir, filename)
    if os.path.exists(filename):
        await measure_tokens(filename)
        
    # # text_1m = os.path.join(base_dir, filename)
    # if os.path.exists(filename):
    #     await measure_tokens(text_1m)

if __name__ == "__main__":
    filename = sys.argv[1]
    asyncio.run(main(filename))
