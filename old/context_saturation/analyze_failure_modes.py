import json
import re

file_path = 'experiments/context_saturation/results/GAIR_LIMO-v2_context_saturation_s42_20260115_045322.json'

def detect_repetition(text, window_size=50, threshold=3):
    """Detects if a substring of window_size appears threshold times."""
    if len(text) < window_size:
        return False
    # Check for direct repetition of phrases
    seen = {}
    tokens = text.split()
    # Use a simple n-gram approach with words for speed/robustness
    n = 10 # 10-word phrases
    if len(tokens) < n:
        return False
        
    phrases = {}
    for i in range(len(tokens) - n + 1):
        phrase = " ".join(tokens[i:i+n])
        phrases[phrase] = phrases.get(phrase, 0) + 1
        if phrases[phrase] >= threshold:
            return True
            
    # Also check for character-level large block repetition
    # (e.g. entire paragraphs verified)
    return False

def analyze_failures():
    with open(file_path, 'r') as f:
        data = json.load(f)

    failures = [d for d in data if not d['correct']]
    
    circular_count = 0
    failed_conclude_indices = []
    wrong_answer_indices = []
    
    print(f"Analyzing {len(failures)} failures...")

    for fail in failures:
        output = fail['output']
        original_input = fail['original']
        
        # 1. Extract Real Problem Keywords (Last 500 chars of input usually contains the real problem)
        # The distractors are usually formatted as "Statement N" or "Problem N". 
        # The real problem is at the end.
        real_problem_text = original_input[-1000:]
        
        # Simple heuristic: Does the output contain ANY significant unique tokens from the last 200 chars?
        # Let's pick 4-character words from the very end of the prompt (likely the question).
        
        # Better Heuristic: 
        # If the output starts with "Statement" or "Problem" followed by a number that matches a distractor, it's distraction.
        # But we are looking for "Context Length" issues (Truncation). 
        # If truncated, the model won't see the end. 
        # If it doesn't see the end, it might hallucinate or discuss an earlier distractor.
        
        # Let's perform a "Relevance Check":
        # Does the output look like it's addressing the real problem?
        # We can simulate this by checking if the output contains a substantial bigram overlap with the *end* of the input vs the *beginning*.
        
        # Strategy:
        # 1. Identify "Looping": 
        #    - Check for repeated 10-word phrases (>3 times).
        # 2. Identify "Context Loss/Truncation":
        #    - If NOT looping, check if it discusses the wrong problem.
        #    - We'll look for key numbers/nouns from the last sentence of the input.
        
        # Get identifying tokens from the real problem (last sentence)
        last_chunk = original_input.strip().split('\n')[-1]
        if len(last_chunk) < 50: # If last line is short, take last few
             last_chunk = original_input[-300:]
             
        # Extract unique alphanumeric tokens > 3 chars
        keywords = set(re.findall(r'[a-zA-Z0-9]{4,}', last_chunk))
        
        # Count keyword hits in output
        hits = 0
        if len(keywords) > 0:
            for kw in keywords:
                if kw in output:
                    hits += 1
            relevance_score = hits / len(keywords)
        else:
            relevance_score = 1.0 # Can't judge
            
        is_looping = detect_repetition(output)
        
        if is_looping:
            circular_count += 1
        elif "\\boxed{" in output:
            # If it has a boxed answer but correct=False, it's a Wrong Answer
            wrong_answer_indices.append(fail['id'])
        else:
            # No boxed answer and no loop -> Failed to Conclude (Truncation/Giving Up)
            failed_conclude_indices.append(fail['id'])

    print(f"Analysis Results:")
    print(f"Circular Thinking / Loops: {circular_count}")
    print(f"Failed to Conclude (Max Token Limit / Truncation): {len(failed_conclude_indices)}")
    print(f"Wrong Answer (Finished but Incorrect): {len(wrong_answer_indices)}")
    
    print("\n--- Wrong Answer IDs (Indices in Results) ---")
    print(wrong_answer_indices)
    
    print("\n--- Failed to Conclude IDs ---")
    print(failed_conclude_indices)

analyze_failures()
