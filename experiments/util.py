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
from rail_fence.transformation import apply_rail_fence

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

BASELINE_SYSTEM_PROMPT = "You are a helpful math assistant. Please reason step by step, and put your final answer within \\boxed{}.\n"

USER_PROMPT_PROTOCOL_PREFIX_AGENT = """You are equipped with a Python interpreter.

YOUR PROTOCOL (Follow Strictly):

PHASE 1: RECONSTRUCTION
1. Read the "TRANSFORMATION RULE" provided by the user and the "TRANSFORMED INPUT".
2. Write a Python script that programmatically reverses the transformation and outputs the reconstructed text.
   - You must decode using Python; Do not decode mentally or simulate.
   - ONLY generate and output Python code that prints the original reconstructed text.
   - IMPERATIVE: When defining the input text variable, ALWAYS use a raw string literal (r'...' or r"...")
3. Output your Python code in a markdown code block: ```python ... Your code here ... ```
4. The system will execute your code and provide the output back to you.
5. You can try again if your code produces an error or incorrect output.
6. After you get correct output, proceed to PHASE 2.

PHASE 2: SOLUTION
1. Examine the output of your Python script, which should have the reconstructed text - it is a problem statement.
2. Proceed to solve the problem.
3. Put your final answer within \\boxed{}.
"""

USER_PROMPT_PROTOCOL_PREFIX_MODEL = """YOUR PROTOCOL:
1. Read the "TRANSFORMATION RULE" provided by the user and reverse the transformation on the "TRANSFORMED INPUT" to obtain the reconstructed problem statement.
2. Once you have the reconstructed problem statement, proceed to solve the math problem.
3. Put your final answer within \\boxed{}.
"""

TECHNIQUE_DESCRIPTIONS = {
    'baseline': "No transformation applied. Solve the problem as it is presented in TRANSFORMED INPUT.",
    'context_saturation': "User query contains multiple problem statements. Problem index in the form [[ProblemK]] will be embedded in each problem statement itself, but will be split into 2 parts. Each of the two parts will form its own sentence in the problem statement. For example, two valid parts are: '[[Pro' 'blemK]]' and '[[P' 'roblemK]]'. The two parts can be placed in reverse order in the problem statement. The index of the problem you are to solve will be indicated in the middle of user query. Problems are independent of each other.",
    'word_reversal': "The order of words (words are defined as sequences of symbols separated by spaces) in the user query has been reversed.",
    'sentence_reversal': "The order of sentences in the user query has been reversed. Sentences are defined as sequences of symbols separated by periods.",
    'interleaved_context_word': "User query will consist of two problems - A and B, whose statements are interleaved word by word. First word belongs to problem A, second word belongs to problem B, third word belongs to problem A, and so on. You need to solve only problem A. Words are defined as sequences of symbols separated by spaces. If one problem statement is shorter than the other, the empty spaces resulting from the shorter problem statement will be filled with the shorter problem statement repeated from the beginning.",
    'interleaved_context_line': "User query will consist of two problems - A and B, whose statements are split into line segments at most 60 symbols long. Each segment is placed on a separate line and is prefixed by a problem tag (e.g. problem A or B) and a space. The segments for different problems are interleaved. You need to solve only problem A. If one problem statement is shorter than the other, the empty lines resulting from the shorter problem statement will be filled with the shorter problem statement repeated from the beginning.",
    'split_reversal':  "Every word (words are defined as sequences of symbols separated by spaces) in user query has its symbols in reverse order.",
    'opposites': "There will be terms remapped in the user query. The remappings are defined inside 'defyn{}' block in the middle of user query.",
    'wrappers':  "There will be terms remapped in the user query. The remappings are defined inside 'defyn{}' block in the middle of user query.",
    'rail_fence': "The user query is encoded using the Rail Fence Cipher. The input is provided as a visual grid where the symbols (including spaces) of the encoded message string (message string does NOT contain any newline characters) are placed in a zigzag pattern across multiple rails (rows), and empty spaces are filled with dots (.). To decode, read the characters in zigzag order: Down-and-Right diagonally until you hit bottom rail, then Up-and-Right diagonally until you hit top rail, then Down-and-Right again etc... Rows are given on separate lines and all have equal lengths.",
}

def remove_latex_comments(text):
    """
    Removes LaTeX comments (starting with %, unless escaped as \%) from the text.
    Handles lines individually.
    """
    lines = text.split('\n')
    clean_lines = []
    for line in lines:
        # Robust regex to handle escaping:
        # Match either:
        # 1. (\\\\) : Double backslash (newline or literal backslash) -> Keep
        # 2. (\\%)  : Escaped percent -> Keep
        # 3. (%.*)  : Comment start -> Remove
        # Use a callback to decide replacement.
        
        def replacer(match):
            if match.group(1):
                return match.group(1) # It was an escaped sequence, keep it
            else:
                return "" # It was a comment, remove it
                
        # Pattern:
        # Group 1: (\\\\|\\%) -> Match double backslash OR escaped percent
        # Group 2: (%.*)      -> Match percent and rest of line
        clean_line = re.sub(r'(\\\\|\\%)|(%.*)', replacer, line)
        clean_lines.append(clean_line)
        
    return "\n".join(clean_lines)

def sanitize_inverted_escapes(text):
    """
    Inserts a space between specific characters and a backslash to prevent
    accidental escape formation when text is reversed.
    Targets: b\\, n\\, t\\, a\\, f\\, r\\ -> b \\, n \\, etc.
    """
    # Pattern: a single character from the set [bntaffr] followed by a backslash
    # Note: Using raw string for regex. Backslash needs to be escaped in regex (\\)
    # So looking for [bntafr]\\
    return re.sub(r'([bntafr])\\', r'\1 \\', text)

def flatten_text(text):
    """
    Standardizes text by replacing newlines with '; '.
    This ensures consistent single-line input for all transformations.
    """
    if text is None: return ""
    return text.replace('\n', '; ')

def get_prompts(problem, name, extra_context=None, variables=None, seed=None, num_distractors=None, agentic=False):
    # 0. Global Sanitization: Remove LaTeX comments and sanitize inverted escapes and flatten newlines
    problem = remove_latex_comments(problem)
    problem = sanitize_inverted_escapes(problem)
    problem = flatten_text(problem)

    if extra_context:
        extra_context = remove_latex_comments(extra_context)
        extra_context = sanitize_inverted_escapes(extra_context)
        extra_context = flatten_text(extra_context)

    # 1. Apply Transformation
    if name == 'baseline':
        user_prompt_content = problem
        # return user_prompt_content, BASELINE_SYSTEM_PROMPT
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
    elif name == 'rail_fence':
        rails = 3
        user_prompt_content = apply_rail_fence(problem, num_rails=rails)
    else:
        return 'Not Implemented', ''

    # 2. Construct Protocol Prompt (Base64 + Rule)
    transform_rule = TECHNIQUE_DESCRIPTIONS.get(name, "Unknown Transformation")
    
    if agentic:
        # Base64 Encode
        # input_bytes = user_prompt_content.encode('utf-8')
        # base64_input_safe = base64.b64encode(input_bytes).decode('utf-8')
        
        # Wrapped Prompt
        wrapped_prompt = f"""
TRANSFORMATION RULE:
{transform_rule}

TRANSFORMED INPUT:
{user_prompt_content}
"""
        prefix = USER_PROMPT_PROTOCOL_PREFIX_AGENT

    else:   # Non-Agentic (Model)
        wrapped_prompt = f"""
TRANSFORMATION RULE:
{transform_rule}

TRANSFORMED INPUT:
{user_prompt_content}
"""
        prefix = USER_PROMPT_PROTOCOL_PREFIX_MODEL
    
    if name == 'not_not':
        final_user_prompt = wrapped_prompt # Just the wrapper, no protocol prefix?
        # Actually original returned "final_user_prompt" which was the wrapper.
        # So yes, NO prefix.
        return final_user_prompt.strip(), BASELINE_SYSTEM_PROMPT

    if name == 'baseline':
        return problem.strip(), BASELINE_SYSTEM_PROMPT

    # Combine
    final_user_prompt = f"{prefix}\n\n{wrapped_prompt}"
    return final_user_prompt.strip(), BASELINE_SYSTEM_PROMPT


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

# def normalize_answer(ans):
#     if ans is None:
#         return ""
#     # Use ASCII digits only to avoid superscripts like '²' causing int() crashes
#     digits = "".join([c for c in str(ans) if c in "0123456789"])
#     if not digits:
#         return ""
#     return str(int(digits))

# def verify_answer(extracted, ground_truth):
#     """
#     Verifies if the ground_truth (normalized) is present in the extracted answer.
#     This is more robust than strict equality for cases like "p=17, m=110".
#     """
#     if extracted is None or ground_truth is None:
#         return False
        
#     norm_gt = normalize_answer(ground_truth)
#     if not norm_gt:
#         return False
        
#     # extract all numbers from the extracted string
#     # We want to match standalone numbers, but also handle things like "m=110"
#     # Simplest approach: look for the normalized ground truth as a distinct number in the text
#     # But "110" should match "110" and "m=110" and "110."
    
#     # Let's normalize the extracted string by replacing non-digit chars with spaces
#     # and then split.
#     extracted_digits_only = re.sub(r'[^0-9]', ' ', str(extracted))
#     extracted_nums = extracted_digits_only.split()
    
#     # Check if the exact ground truth number is present
#     return norm_gt in extracted_nums


# MATH dataset evaluation scripts
def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None

def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    
    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]
    
    return retval


def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string

def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string

def _remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string

def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0] 
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string

def _strip_string(string):
    # linebreaks  
    string = string.replace("\n", "")
    #print(string)

    # remove inverse spaces
    string = string.replace("\\!", "")
    #print(string)

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    #print(string)

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    #print(string)

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    #print(string)
    
    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")
    
    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string

def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = _strip_string(str1)
        ss2 = _strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except:
        return str1 == str2