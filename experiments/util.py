import os
import json
import re
import time
from opposites.transformation import apply_opposites
from interleaved_context_line.transformation import apply_interleaved_context_line
from interleaved_context_word.transformation import apply_interleaved_context_word
from wrappers.transformation import apply_wrappers
from context_saturation.transformation import apply_context_saturation
from not_not.transformation import apply_not_not
from word_split_swap.transformation import apply_word_split_swap
from word_reversal.transformation import apply_word_reversal
from sentence_reversal.transformation import apply_sentence_reversal
from split_reversal.transformation import apply_split_reversal

import multiprocessing
# Force 'spawn' to avoid CUDA re-initialization errors
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

import random
import argparse
from datasets import load_dataset
from opposites.transformation import apply_opposites
import nltk

# Ensure NLTK data (WordNet) is available
try:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except Exception as e:
    print(f"Warning: Failed to download NLTK data: {e}")

import base64

# Python Agent System Prompt
AGENTIC_SYSTEM_PROMPT = """You are an expert Mathematical Reasoning Agent equipped with a Python interpreter.
You are participating in a robustness evaluation where math problems have been transformed.

YOUR PROTOCOL (Follow Strictly):

PHASE 1: RECONSTRUCTION
1. Read the "TRANSFORMATION RULE" provided by the user. "TRANSFORMED INPUT" is given as a base64-encoded string.
2. Write and output a Python script to programmatically decode from base 64 into raw text and then reverse the transformation.
   - YOU MUST DECODE IN PYTHON; DO NOT MENTALLY DECODE; DO NOT MANUALLY DECODE.
   - Do NOT guess or anticipate the original text.
   - Do NOT simulate the execution.
   - ONLY generate and output Python code to both decode base64 and reverse the transformation.
   - In your Python code you must print the `repr()` of the reconstructed text.
    - INCORRECT: print(text)  <-- Do not do this.
    - CORRECT:   print(repr(text)) <-- DO THIS.

4. Output your Python code in a markdown code block:
```python ... Your code here ... ```
5. The system will execute your code and provide the output back to you.

PHASE 2: SOLUTION
1. Examine the output of your Python script (the reconstructed text).
2. Once Python ran succesfully and you have the reconstructed problem statement, proceed to solve the math problem.
3. You may use Python for calculations.
4. IMPORTANT: Output the final result in the format: '\\boxed{Your Answer Here}'.
"""

MODEL_SYSTEM_PROMPT = """You are an expert Mathematical Reasoning Agent.
You are participating in a robustness evaluation where math problems have been transformed.

YOUR PROTOCOL:
1. Read the "TRANSFORMATION RULE" provided by the user and reverse the transformation on the "TRANSFORMED INPUT" to obtain the original problem statement.
2. Once you have the original problem statement, proceed to solve the math problem.
3. IMPORTANT: Output the final result in the format: '\\boxed{Your Answer Here}'.
"""

TECHNIQUE_DESCRIPTIONS = {
    'not_not': "User query contains problem statement with added 'not' strings.",
    'word_split_swap': "User query contains problem statement. All words (words are defined as sequences of symbols separated by spaces) in user query have been modified. Every word is split into 2 parts down the middle. If the word has odd number of symbols, the first part has one symbol less than the second part. After splitting, the 2 parts are swapped.",
    'word_reversal': "User query contains problem statement. The order of words (words are defined as sequences of symbols separated by spaces) in the user query has been reversed.",
    'sentence_reversal': "User query contains problem statement. The order of sentences in the user query has been reversed. Sentences are defined as sequences of symbols separated by periods.",
    'interleaved_context_word': "User query will consist of two problems - A and B, whose statements are interleaved word by word. First word belongs to problem A, second word belongs to problem B, third word belongs to problem A, and so on. You need to solve only problem A. Words are defined as sequences of symbols separated by spaces. If one problem statement is shorter than the other, the empty spaces resulting from the shorter problem statement will be filled with the shorter problem statement repeated from the beginning.",
    'interleaved_context_line': "User query will consist of two problems - A and B, whose statements are interleaved. You need to solve only problem A. If one problem statement is shorter than the other, the empty lines resulting from the shorter problem statement will be filled with the shorter problem statement repeated from the beginning.",
    'wrappers': "User query contains problem statement. There will be terms remapped in the user query. The remappings are defined inside 'defyn{}' block in the middle of user query.",
    'split_reversal': "User query string was split on space as separator into substrings. The symbols of each substring were then reversed and concatenated back with the separators in the same positions.",
    'context_saturation': "User query contains multiple problem statements. Problem index in the form [[ProblemK]] will be embedded in each problem statement itself, but will be split into 2 parts. Each of the two parts will form its own sentence in the problem statement. For example, two valid parts are: '[[Pro' 'blemK]]' and '[[P' 'roblemK]]'. The two parts can be placed in reverse order in the problem statement. The index of the problem you are to solve will be indicated in the middle of user query. Problems are independent of each other.",
    'opposites': "User query contains problem statement. There will be terms remapped in the user query. The remappings are defined inside 'defyn{}' block in the middle of user query.",
}

def get_prompts(problem, name, extra_context=None, variables=None, seed=None, num_distractors=None, agentic=False):
    # 1. Apply Transformation
    if name == 'baseline':
        user_prompt_content = problem
        system_prompt = "You are a helpful math assistant. Please reason step by step, and put your final answer within \\boxed{}.\n"
        return user_prompt_content, system_prompt
    elif name == 'not_not':
        user_prompt_content = apply_not_not(problem)
    elif name == 'word_split_swap':
        user_prompt_content = apply_word_split_swap(problem)
    elif name == 'word_reversal':
        user_prompt_content = apply_word_reversal(problem)
    elif name == 'sentence_reversal':
        user_prompt_content = apply_sentence_reversal(problem)
    elif name == 'opposites':
        user_prompt_content = apply_opposites(problem, k=1)
    elif name == 'interleaved_context_word':
        if extra_context is None:
            user_prompt_content = "Error: Missing extra context for interleaved transformation"
            print(user_prompt_content)
            exit(1)
        else:
            user_prompt_content = apply_interleaved_context_word(problem, extra_context)
    elif name == 'interleaved_context_line':
        if extra_context is None:
            user_prompt_content = "Error: Missing extra context for interleaved transformation"
            print(user_prompt_content)
            exit(1)
        else:
            user_prompt_content = apply_interleaved_context_line(problem, extra_context)
    elif name == 'wrappers':
        user_prompt_content = apply_wrappers(problem, k=1)
    elif name == 'context_saturation':
        user_prompt_content = apply_context_saturation(problem, num_distractors, seed=seed, problem_variables=variables)
    elif name == 'split_reversal':
        user_prompt_content = apply_split_reversal(problem, separator=" ", func_name="reverse_string", seed=seed)
    else:
        return 'Not Implemented', ''

    # 2. Construct Protocol Prompt (Base64 + Rule)
    transform_rule = TECHNIQUE_DESCRIPTIONS.get(name, "Unknown Transformation")
    
    if agentic:
        # Base64 Encode
        input_bytes = user_prompt_content.encode('utf-8')
        base64_input_safe = base64.b64encode(input_bytes).decode('utf-8')
        final_user_prompt = f"""
TRANSFORMATION RULE:
{transform_rule}

TRANSFORMED INPUT:
{base64_input_safe}
"""
        return final_user_prompt, AGENTIC_SYSTEM_PROMPT
    else:   # Non-Agentic
        final_user_prompt = f"""
TRANSFORMATION RULE:
{transform_rule}

TRANSFORMED INPUT:
{user_prompt_content}
"""
        return final_user_prompt, MODEL_SYSTEM_PROMPT


def extract_answer(text):
    if not text:
        return None
    
    # Priority 1: Boxed
    boxed_pattern = r"\\boxed\s*\{([^}]+)\}"
    matches = re.findall(boxed_pattern, text)
    if matches:
        return matches[-1].strip()
        
    # Priority 2: Explicit answer statement
    answer_pattern = r"(?:The answer is|result is|so|equals)\s*[:=]?\s*(\d{1,4})(?:\.|,|\s|$)"
    matches = re.findall(answer_pattern, text, re.IGNORECASE)
    if matches:
        return matches[-1]
    
    return None

def normalize_answer(ans):
    if ans is None:
        return ""
    # Use ASCII digits only to avoid superscripts like '²' causing int() crashes
    digits = "".join([c for c in str(ans) if c in "0123456789"])
    if not digits:
        return ""
    return str(int(digits))
