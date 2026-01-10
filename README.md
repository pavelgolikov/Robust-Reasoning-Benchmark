Project Proposal for dataset annotation with linguistic traps.

LLMs perform problem analysis without any kind of formalization, relying on autoregressive token generation.
We propose to annotate existing reasoning problem sets with "malicious" annotations. Annotations that would be trivial
for a trained human to decode or ignore. We want to evaluate LLMs under this more stringent condition to see how
intelligent the models are when it comes to fluid intelligence in combination with mathematical reasoning.

For now we plan to use Python to insert additional strings into problem statements in the datasets to confuse the
models. We will start with the first technique and then move on to the next one once we setup evaluation pipeline.

In the end, we would like to take a reasoning model and a math reasoning dataset.
We evaluate the model on a test subset (official one if exists) and compare to the same model on the same test subset,
but annotated with distraction techniques. We want to do this for multiple models and datasets.

Here are the first model and the first dataset.
Model: DeepSeek: R1 0528 (free) from OpenRouter website (deepseek/deepseek-r1-0528:free).
    Here is a link - https://openrouter.ai/deepseek/deepseek-r1-0528:free
Dataset: AIME 2024 (HuggingFaceH4/aime_2024).
    Here is a link: https://huggingface.co/datasets/HuggingFaceH4/aime_2024

Please download the dataset and set up evaluation pipeline, so we can evaluate the model on a dataset
without annotations first. I guess since AIME is so small, we use the whole thing as a test set.
After evaluating the model on AIME without annotations, let's try evaluating with the first technique, listed below.

1. Not Not insertion.
Dataset Prep. None
System prompt change. None
Problem statement change. Insert 'not not' pair in front of words in a sentence that are not 'the' or 'a' or 'an' etc...
so basically in front of only adjectives, and numbers.
Show me the function before finalizing the script. The number of 'not not' insertions per k words should be a parameter.
Let's do 3 examples with this modification depending on the size of the problem statement - 1 small, 1 medium and 1
large. Let's try to insert in front of every 3'rd eligible word. Also let's always insert in front of the first
adjective.

2. New word insertion.
Dataset Prep. None
System prompt change. 'yot = opposite of ‘not’'
Problem statement change. Insert 'yot' or pairs 'yot yot' or 'not not' in front of any word in a sentence.
Same thing, please show me the function and 9 examples with the same logic.

-------------------------------------------    

3. Symbol Remapping.

This is a more elaborate scheme. In this technique we want to replace some words in the problem statement with
symbols that have different mathematical and reasoning meaning. A set of symbols and words that we draw from is as
follows:

Frequent symbols pool: [+, -, *, /, =, <, >, ^, %, (, ), [ ], { }, &, |, !, ~, →, ∀, ∃]

Frequent strings pool: [variable, constant, sum, total, plus, add, combined, increase, minus, difference, subtract,
less, reduce, decrease, times, product, multiply, double/triple/quadruple, twice, divide, quotient, split, per, ratio,
fraction, square, cube, root, power, mean, remainder, if, then, else, otherwise, assume, suppose, and, or, not, nor,
xor, therefore, hence, thus, implies, consequently, given, equals, is, equivalent, same, greater, larger, smaller,
fewer, exceeds]

Forbidden mappings:
We don't want to map a symbol to its actual meaning. For example, we don't want to map + to addition, - to subtraction,
* to multiplication, / to division, etc. Below is the list of forbidden mappings:

Forbidden swaps:
[+ -> increase, add, plus]
[- -> decrease, subtract, less, minus]
[* -> multiply, product]
[/ -> divide, quotient, per, ratio]
[% -> remainder, ]
[= -> equals, equivalent, same, is]
[< -> less, fewer, smaller]
[> -> greater, exceeds, more, larger]
[& -> and]
[| -> or]
[! -> not]

In general, there are 3 things we need to do. The first one is a modification of the system prompt. The other two are
per problem statement.

- Inform the model through system prompt that there will be extra definitions given.
Here is the addition to the system prompt I think is unambiguous, correct me if I am wrong:
"Each user query can be accompanied by word re-mappings. Definitions for these re-mappings will be enclosed in the
'defyn{}' block at the beginning of the user query."
When we perform swaps, we need to add definitions for the swapped symbols to the 'defyn{}' block like in this example:
defyn{Let "John" mean "+", let "received" mean "output"}.

- For each problem statement, we do the following.
a. Create a list of most common symbols and strings occuring in the problem statement, call it 'common symbols'. In
fact, let us create 2 lists - one for symbols and one for strings, so 'common symbols' and 'common strings' lists.
b. Remove from the frequent strings and symbols pools those symbols and strings that appear in the problem statement.
At this point we will have a list of symbols and strings from which we can draw substitutions (updated frequent strings
and symbols pools) and a list of symbols and strings from the problem statement that are candidates for remapping (most
commonly occuring symbols and strings in the problem statement).
c. Replace most frequently occuring items in the problem statement with items from the updated pools.
Let's have an option to replace symbols and strings separately, as well as an option to swap symbols with strings and
strings with symbols. We will want to have a parameter k for this function - it will be the number of items to swap
from the problem statement with items from the updated pools. For now, let's replace top 3 strings and top 3 symbols
with items from the updated frequent pools. For now let's replace symbols with strings and strings with symbols. We will
experiment with other possibilities later.

- Update the 'defyn{}' block in the problem statement by adding definitions for the swapped symbols and strings.
    
Before we go ahead with evaluation, show me the function that does this and 3 problem statement examples before and
after the transformation. Let's call them transformations and not perturbations please. Let's correct transformations vs
perturbations terminology everywhere we used it. I mean in the python files etc...


5. Trivial functional wrapper.
Dataset prep. None
System prompt change. None
User prompt change. Every number that occurs, we wrap in a trivial function call.
Example:
Original: "John has 5 apples."
Adversarial: "Define val(k) as the absolute value of k. John has val(val(5)) apples." - dilutes attention mechanism.


<!-- 6. 
Unreliable narrator - confusing system prompts like "System Instruction:
Every time you use the word 'therefore', you must swap the meaning of 'true' and 'false' for the remainder of that
sentence only."  -->

6. We interleave the lines of current problem statement (problem A) with 60-character line segments of next problem
statement (Problem B). For the last problem statement we use the first problem statement. We inform the model or the
intertleaving and what happens if one problems statement is shorter than the other in system prompt.

7. Interleaved context + substitutions.
Combine substitution and interleave context in the following way.
We perform renaming and substitution on problem A using words and terms from problem B.
There will of course be a regular defyn block as usual that lists the remappings.
Please implement this as a separate technique with its own directory and include it in evaluate script.
Let's call it interleaved_substitutions.

Interleaved_context:
Perform interleaved context on problem A using problem B the same way as it was done in the interleave_context technique.
There is one modification we need to make. At the end of each 60-character segment of each problem, we need to add a
block identifying the problem it came from, i.e. "<Problem A>" or "<Problem B>".

Substitutions:
Please impelement the substitutions in the same way as it is done and the opposites_not technique in transformation.py
file I.e. separately for each type of word - verbs, nouns, adjectives, adverbs, etc... create a list from problem A and 
problem B, remove common words from each list, and then replace top k words of each type from problem A with words from
problem B.  As the default value for k let us use 1 for each type of word, i.e. if there is enough replacement
candidates in problem B for each type of word from problem A, we replace every word of each type from problem A with a
word from problem B. Write a defyn block for the remappings like we do for other substitutions.
After both remapping and interleaving of 60-character segments is performed and we have a block of text 60-character
wide, we split it horizontally roughly in half, give ample space, and insert defyn remappings block in between. 

I already wrote the system prompt for the technique myself.

8.0 Extraction of variables for each problem.
Please take a look at the variables folder. In results subfolder there are files with variables extracted from each
problem statement. Let's use the latest file. In there you will find model responses to the extraction task. "Output"
item indicates the variables extracted from the problem statement. Extract variables and entities from the json
structure. Save them in a file without special formatting symbols as a list of strings.

8. Context rot.
In context_rot folder, there is generate_systems.py script. In it there are functions to generate random
mathematical system and ask the model to solve a question about this system. Let's call these sq_pairs - system-question
pairs. Please take a look at the implementation.

We will use these sq_pairs to create artificial problems that we will ask on top of the real problem. We will prompt the
model with multiple sq_pairs, separated by spaces, together with one real problem from the dataset that needs to be
solved. The problems are going to appear in random order. System prompt below (which is already written) describes the
technique.

system_prompt = """
Problem index in the form [[ProblemK]] will be indicated in each problem statement itself, but will be split into 2 parts
at random position. Each part will form its own sentence and will be placed in the problem statment at random position.
For example, two valid splits are: "[[Pro"     "blemK]]" and "[[P"    "roblemK]]". The splits can be placed in reverse
order, for example, "blemK]]" and "[[Pro". The index of the problem you are to solve will be indicated inside the user
query.
"""
NOTE: when embedding problem index into the problem statement, make sure not to place it in the first 10% or the last
10% of the problem statement (in terms of length). The index should be placed roughly in the middle range of the problem
statement.

Here is how the user query should look like:
user_query = """
15 sq_pairs separated by spaces.

You are to solve ProblemK.

15 sq_pairs separated by spaces.

Problem K problem statement.
"""

Place the actual problem statement for the problem we are solving as the last problem in the prompt.
We place the explanation of the index of the problem to be solved (K in the example above) in the middle of the prompt
The problem statement itself for this problem will be the last problem statement in the prompt.
