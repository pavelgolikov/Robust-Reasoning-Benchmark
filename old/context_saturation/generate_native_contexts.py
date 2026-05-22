import os
import sys
import json
import asyncio
import time
from anthropic import AsyncAnthropic
from google import genai
from openai import AsyncOpenAI
import tiktoken
from dotenv import load_dotenv

load_dotenv()

# We need the same tag stripper used in the actual API requests to ensure our target build 
# matches the payload size exactly at runtime.
import re as _re
def strip_thinking_tags(text):
    return _re.sub(r'</?(think|thinking|reasoning|reflection|scratchpad)[^>]*>', '', text, flags=_re.IGNORECASE).strip()

async def get_native_token_count(provider, model_name, messages):
    try:
        if provider == "openai":
            text = "".join(m['content'] for m in messages)
            try:
                enc = tiktoken.encoding_for_model(model_name)
            except KeyError:
                # Fallback to standard OpenAI tokenizer if exact model isn't mapped in tiktoken yet
                enc = tiktoken.get_encoding('o200k_base')
            return len(enc.encode(text))
            
        elif provider == "anthropic":
            client = AsyncAnthropic(default_headers={"anthropic-beta": "context-1m-2025-08-07"})
            formatted = []
            for m in messages:
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
            for m in messages:
                role = "model" if m["role"] == "assistant" else "user"
                contents.append({"role": role, "parts": [{"text": m["content"]}]})
            resp = client.models.count_tokens(
                model=model_name,
                contents=contents,
            )
            return resp.total_tokens
    except Exception as e:
        print(f"Token counting failed: {e}")
        return -1
    return -1

async def build_provider_context(base_file, provider, model_name, target_tokens, tolerance_pct=0.01):
    print(f"\n--- Building context for {provider} ({model_name}) ---")
    
    with open(base_file, 'r') as f:
        master_messages = json.load(f)
        
    print(f"Loaded master file ({len(master_messages)} messages). Stripping <think> tags...")
    for msg in master_messages:
        msg['content'] = strip_thinking_tags(msg['content'])

    # Binary search to find the correct number of messages to hit target_tokens
    low = 1
    high = len(master_messages)
    best_k = 1
    best_count = 0
    margin = target_tokens * tolerance_pct
    
    print(f"Executing binary search via API calls for {target_tokens:,} tokens...")
    
    while low <= high:
        mid_k = (low + high) // 2
        test_history = master_messages[:mid_k]
        
        count = await get_native_token_count(provider, model_name, test_history)
        if count == -1:
            print(f"Failed to get token count for mid={mid_k}. Aborting.")
            return None
            
        print(f"  Tested K={mid_k}: {count:,} tokens")
        
        # If we hit it perfectly within margin, stop immediately 
        if abs(count - target_tokens) <= margin:
            best_k = mid_k
            best_count = count
            break
            
        if count < target_tokens:
            best_k = mid_k
            best_count = count
            low = mid_k + 1
        else:
            high = mid_k - 1
            
        # Respect rate limits for cloud API counters
        if provider != "openai":
            await asyncio.sleep(1)

    print(f"Found ideal K={best_k} with {best_count:,} tokens.")
    
    if abs(best_count - target_tokens) > margin:
        print(f"WARNING: Best match {best_count:,} is outside 1% margin ({int(margin):,}).")

    # Save specific file
    filename = os.path.basename(base_file)
    context_type = filename.split('_')[1] # 'math' or 'text'
    size_str = f"{target_tokens // 1000}K" if target_tokens >= 1000 else str(target_tokens)
    out_name = f"experiments/context_{context_type}_{size_str}_{provider}.json"
    
    print(f"Saving exactly {best_k} messages to {out_name}...")
    with open(out_name, 'w') as f:
        json.dump(master_messages[:best_k], f, indent=2)
        
    return out_name

async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate native token-counted contexts.")
    parser.add_argument("--model", type=str, required=True, help="API model name")
    parser.add_argument("--provider", type=str, required=False, help="Explicit provider (inferred if None)")
    parser.add_argument("--context_size", type=int, required=True, help="Target precise token size")
    parser.add_argument("--context_type", type=str, required=True, help="math or text")
    
    args = parser.parse_args()
    
    provider = args.provider
    if not provider:
        if "gemini" in args.model.lower(): provider = "google"
        elif "claude" in args.model.lower(): provider = "anthropic"
        elif "gpt" in args.model.lower() or "o1" in args.model.lower(): provider = "openai"
        else: raise ValueError("Could not infer provider")

    # Select base master file based on context_type
    if args.context_type == "math":
        base_file = "experiments/context_math_1M.json"
    elif args.context_type == "text":
        base_file = "experiments/context_text_1M.json"
    else:
        raise ValueError(f"Unknown context type: {args.context_type}")
        
    if not os.path.exists(base_file):
        raise FileNotFoundError(f"Missing master file: {base_file}")
        
    await build_provider_context(base_file, provider, args.model, args.context_size)

if __name__ == "__main__":
    asyncio.run(main())
