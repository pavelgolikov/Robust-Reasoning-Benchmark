import os
import json
import re
import time
from opposites.transformation import apply_opposites
from interleaved_context_line.transformation import apply_interleaved_context_line
from interleaved_context_word.transformation import apply_interleaved_context_word
from wrappers.transformation import apply_wrappers
from variables.transformation import apply_variables
from context_saturation.transformation import apply_context_saturation
from not_not_yot.transformation import apply_not_not_yot
from word_split_swap.transformation import apply_word_split_swap
from word_reversal.transformation import apply_word_reversal
from sentence_reversal.transformation import apply_sentence_reversal
from reversal.transformation import apply_reversal

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
    


def get_prompts(problem, name, extra_context=None, variables=None, seed=None, num_distractors=None, decode_find_only=False):
    # modify problem according to experiment name
    if name == 'baseline':
        user_prompt = problem
        system_prompt = "You are a helpful math assistant. Please reason step by step, and put your final answer within \\boxed{}.\n"
        return user_prompt, system_prompt
    elif name == 'not_not_yot':
        if decode_find_only:
            system_prompt = "You are a helpful math assistant.\n\
User query contains problem statement with added 'not' and 'yot' strings. Yot means the opposite of not.\n\
ONLY reconstruct the original problem statement without solving the problem.\n\
Please reason step by step, and put the final reconstructed problem statement inside \\boxed{}.\n"
        else:
            system_prompt = "You are a helpful math assistant. \n\
User query contains problem statement with added 'not' and 'yot' strings. Yot means the opposite of not.\n\
You need to reconstruct the original problem statement before solving it.\n\
Please reason step by step, and put your final answer within \\boxed{}.\n"
        user_prompt = apply_not_not_yot(problem)
        return user_prompt, system_prompt
    elif name == 'word_split_swap':
        if decode_find_only:
            system_prompt = "You are a helpful math assistant.\n\
User query contains problem statement. All words (words are defined as sequences of symbols separated by spaces) in user query have been modified.\n\
Every word is split into 2 parts down the middle. If the word has odd number of symbols, the first part has one symbol less than the second part.\n\
After splitting, the 2 parts are swapped.\n\
ONLY reconstruct the original problem statement without solving the problem.\n\
Please reason step by step, and put the final reconstructed problem statement inside \\boxed{}.\n"
        else:
            system_prompt = "You are a helpful math assistant.\n\
User query contains problem statement. All words (words are defined as sequences of symbols separated by spaces) in user query have been modified.\n\
Every word is split into 2 parts down the middle. If the word has odd number of symbols, the first part has one symbol less than the second part.\n\
After splitting, the 2 parts are swapped.\n\
You need to reconstruct the original problem statement before solving it.\n\
Please reason step by step, and put your final answer within \\boxed{}.\n"
        user_prompt = apply_word_split_swap(problem)
        return user_prompt, system_prompt
    elif name == 'word_reversal':
        if decode_find_only:
            system_prompt = "You are a helpful math assistant.\n\
User query contains problem statement. The order of words (words are defined as sequences of symbols separated by spaces) in the user query has been reversed.\n\
ONLY reconstruct the original problem statement without solving the problem.\n\
Please reason step by step, and put the final reconstructed problem statement inside \\boxed{}.\n"
        else:
            system_prompt = "You are a helpful math assistant.\n\
User query contains problem statement. The order of words (words are defined as sequences of symbols separated by spaces) in the user query has been reversed.\n\
You need to reconstruct the original problem statement before solving it.\n\
Please reason step by step, and put your final answer within \\boxed{}.\n"
        user_prompt = apply_word_reversal(problem)
        return user_prompt, system_prompt
    elif name == 'sentence_reversal':
        if decode_find_only:
            system_prompt = "You are a helpful math assistant.\n\
User query contains problem statement. The order of sentences in the user query has been reversed. Sentences are defined as sequences of symbols separated by periods.\n\
ONLY reconstruct the original problem statement without solving the problem.\n\
Please reason step by step, and put the final reconstructed problem statement inside \\boxed{}.\n"
        else:
            system_prompt = "You are a helpful math assistant.\n\
User query contains problem statement. The order of sentences in the user query has been reversed. Sentences are defined as sequences of symbols separated by periods.\n\
You need to reconstruct the original problem statement before solving it.\n\
Please reason step by step, and put your final answer within \\boxed{}.\n"
        user_prompt = apply_sentence_reversal(problem)
        return user_prompt, system_prompt
    elif name == 'opposites':
        if decode_find_only:
            system_prompt = "You are a helpful math assistant.\n\
User query contains problem statement. There will be terms remapped in the user query. The remappings are defined inside 'defyn{}' block in the middle of user query.\n\
ONLY reconstruct the original problem statement without solving the problem.\n\
Please reason step by step, and put the final reconstructed problem statement inside \\boxed{}.\n"
        else:
            system_prompt = "You are a helpful math assistant.\n\
User query contains problem statement. There will be terms remapped in the user query. The remappings are defined inside 'defyn{}' block in the middle of user query.\n\
You need to reconstruct the original problem statement before solving it.\n\
Please reason step by step, and put your final answer within \\boxed{}.\n"
        user_prompt = apply_opposites(problem, k=1)
        return user_prompt, system_prompt
    elif name == 'interleaved_context_word':
        if decode_find_only:
            system_prompt = "You are a helpful math assistant.\n\
User query will consist of two problems - A and B, whose statements are interleaved word by word.\n\
First word belongs to problem A, second word belongs to problem B, third word belongs to problem A, and so on.\n\
You need to reconstruct only problem A. Words are defined as sequences of symbols separated by spaces.\n\
If one problem statement is shorter than the other, the empty spaces resulting from the shorter problem statement\n\
will be filled with the shorter problem statement repeated from the beginning.\n\
ONLY reconstruct problem A statement without solving the problem.\n\
Please reason step by step, and put the final reconstructed problem statement inside \\boxed{}.\n"
        else:
            system_prompt = "You are a helpful math assistant.\n\
User query will consist of two problems - A and B, whose statements are interleaved word by word.\n\
First word belongs to problem A, second word belongs to problem B, third word belongs to problem A, and so on.\n\
You need to solve only problem A. Words are defined as sequences of symbols separated by spaces.\n\
If one problem statement is shorter than the other, the empty spaces resulting from the shorter problem statement\n\
will be filled with the shorter problem statement repeated from the beginning.\n\
You need to reconstruct the original problem statement before solving it.\n\
Please reason step by step, and put your final answer within \\boxed{}.\n"
        if extra_context is None:
            user_prompt = "Error: Missing extra context for interleaved transformation"
        else:
            user_prompt = apply_interleaved_context_word(problem, extra_context)
        return user_prompt, system_prompt
    elif name == 'interleaved_context_line':
        if decode_find_only:
            system_prompt = "You are a helpful math assistant.\n\
User query will consist of two problems - A and B, whose statements are interleaved.\n\
You need to reconstruct only problem A. If one problem statement is shorter than the other,\n\
the empty lines resulting from the shorter problem statement will be filled with lines from the\n\
shorter problem statement repeated from the beginning.\n\
ONLY reconstruct problem A statement without solving the problem.\n\
Please reason step by step, and put the final reconstructed problem statement inside \\boxed{}.\n"
        else:
            system_prompt = "You are a helpful math assistant.\n\
User query will consist of two problems - A and B, whose statements are interleaved.\n\
You need to solve only problem A. If one problem statement is shorter than the other,\n\
the empty lines resulting from the shorter problem statement will be filled with the\n\
shorter problem statement repeated from the beginning.\n\
You need to reconstruct the original problem statement before solving it.\n\
Please reason step by step, and put your final answer within \\boxed{}.\n"
        if extra_context is None:
            user_prompt = "Error: Missing extra context for interleaved transformation"
        else:
            user_prompt = apply_interleaved_context_line(problem, extra_context)
        return user_prompt, system_prompt
    elif name == 'wrappers':
        if decode_find_only:
            system_prompt = "You are a helpful math assistant.\n\
User query contains problem statement. There will be terms remapped in the user query. The remappings are defined inside 'defyn{}' block in the middle of user query.\n\
ONLY reconstruct the original problem statement without solving the problem.\n\
Please reason step by step, and put the final reconstructed problem statement inside \\boxed{}.\n"
        else:
            system_prompt = "You are a helpful math assistant.\n\
User query contains problem statement. There will be terms remapped in the user query. The remappings are defined inside 'defyn{}' block in the middle of user query.\n\
You need to reconstruct the original problem statement before solving it.\n\
Please reason step by step, and put your final answer within \\boxed{}.\n"
        user_prompt = apply_wrappers(problem, k=1)
        return user_prompt, system_prompt
    elif name == 'context_saturation':
        if decode_find_only:
            system_prompt = "You are a helpful math assistant.\n\
User query contains multiple problem statements. Problem index in the form [[ProblemK]] will be embedded in each problem\n\
statement itself, but will be split into 2 parts. Each of the two parts will form its own sentence in the problem statement\n\
For example, two valid parts are: '[[Pro'     'blemK]]' and '[[P'    'roblemK]]'. The two parts can be placed in reverse\n\
order in the problem statement. The index of the problem you are to identify will be indicated in the middle of user query.\n\
ONLY identify the correct problem statement without solving the problem.\n\
Please reason step by step, and put the final reconstructed problem statement inside \\boxed{}.\n"
        else:
            system_prompt = "You are a helpful math assistant.\n\
User query contains multiple problem statements. Problem index in the form [[ProblemK]] will be embedded in each problem\n\
statement itself, but will be split into 2 parts. Each of the two parts will form its own sentence in the problem statement\n\
For example, two valid parts are: '[[Pro'     'blemK]]' and '[[P'    'roblemK]]'. The two parts can be placed in reverse\n\
order in the problem statement. The index of the problem you are to solve will be indicated in the middle of user query. \n\
Problems are independent of each other.\n\
You need to identify the correct problem statement before solving it.\n\
Please reason step by step, and put your final answer within \\boxed{}.\n"
        user_prompt = apply_context_saturation(problem, num_distractors, seed=seed, problem_variables=variables)
        return user_prompt, system_prompt
    elif name == 'reversal':
        system_prompt = "You are a helpful math assistant.\n\
User query contains problem statement. User query string was split on space as separator into substrings.\n\
The symbols of each substring were then reversed and concatenated back with the separators in the same positions.\n\
You need to reconstruct the original problem statement before solving it.\n\
Please reason step by step, and put your final answer within \\boxed{}.\n"
        user_prompt = apply_reversal(problem, separator=" ", func_name="reverse_string", seed=seed):
        return user_prompt, system_prompt
    elif name == 'variables':
        system_prompt = "You are a helpful math assistant.\n\
Your goal is to identify important 'load-bearing' terms in a math problem that we will later target for redefinition.\n"
        user_prompt = apply_variables(problem)
        return user_prompt, system_prompt
    else:
        return 'Not Implemented', ''


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
