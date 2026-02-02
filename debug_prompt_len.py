import sys
import os
import json
from transformers import AutoTokenizer

def main():
    print("Initializing Tokenizer (Falcon-H1R-7B)...")
    # Use the exact model if possible, or fallback to 7B base compatible
    tokenizer = AutoTokenizer.from_pretrained("tiiuae/Falcon-7B", trust_remote_code=True)
    
    with open('history_dump.txt', 'r') as f:
        history = json.load(f) # List of content strings
        
    # Reconstruct history list of dicts for template
    # Since history_dump only saved content, we must infer roles based on index
    # 0: System
    # 1: User
    # 2: Asst
    # 3: User
    # 4: Asst
    # 5: User
    
    roles = ["system", "user", "assistant", "user", "assistant", "user"]
    chat_history = []
    
    full_text_manual = ""
    
    for i, content in enumerate(history):
        role = roles[i] if i < len(roles) else "user"
        chat_history.append({"role": role, "content": content})
        full_text_manual += content + " "

    print(f"Loaded {len(chat_history)} messages.")

    # 1. Measure Manual Concat
    tokens_manual = len(tokenizer.encode(full_text_manual))
    print(f"Manual Concat Tokens: {tokens_manual}")
    
    # 2. Measure Chat Template (if available)
    try:
        if tokenizer.chat_template:
            templated = tokenizer.apply_chat_template(chat_history, tokenize=False)
            tokens_template = len(tokenizer.encode(templated))
            print(f"Chat Template Tokens: {tokens_template}")
        else:
            print("No default chat template found.")
            # construct default stack
            # Falcon usually uses: 
            # System: ...
            # User: ...
            # Falcon: ...
            # User: ...
            prompt = ""
            for msg in chat_history:
                if msg['role'] == 'system':
                    prompt += msg['content'] + "\n"
                elif msg['role'] == 'user':
                    prompt += "User: " + msg['content'] + "\n"
                elif msg['role'] == 'assistant':
                    prompt += "Falcon: " + msg['content'] + "\n"
            
            tokens_naive = len(tokenizer.encode(prompt))
            print(f"Naive Format Tokens: {tokens_naive}")

    except Exception as e:
        print(f"Template Error: {e}")

if __name__ == "__main__":
    main()
