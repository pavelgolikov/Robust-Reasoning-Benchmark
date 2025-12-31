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
Dataset Prep. 
We want to replace most frequently occuring symbols in the problem statement with symbols that have different
mathematical and reasoning meaning. We divide relevant symbols into 3 groups: symbols, variables, and words.

Here are the lists of most frequently occuring symbols:
Arithmetic & Algebra
+ (Plus / Addition)
- (Minus / Subtraction)
* (Asterisk / Multiplication)
/ (Forward Slash / Division)
= (Equals)
< (Less Than)
> (Greater Than)
^ (Caret / Exponentiation)
% (Percent / Modulo)

Logic & Sets (High Confusion Potential)
& (Ampersand / Logical AND)
| (Pipe / Logical OR)
! (Exclamation / Logical NOT - often used in coding context)
~ (Tilde / Negation)
→ (Arrow / Implication - if dataset supports unicode)
∀ (For All)
∃ (There Exists)
Grouping (Structural Symbols)
( ) (Parentheses)
[ ] (Brackets)
{ } (Curly Braces)

"Malicious" Swap Tip:
If you swap structural symbols like ( or ) with a common word or symbol, you effectively destroy the model's ability to
parse the "tree" of the equation, which is devastating for Transformers. 

For the substitution algorithm to be effective ("malicious" but solvable), you want symbols and variables that trigger
strong priors in the model (e.g., seeing x makes the model think "algebra," and seeing + makes it think "addition"). 

Most Commonly Used Strings:
variable constant 
sum total plus added combined increase
minus difference subtract less reduce decrease
times product multiply double/triple/quadruple twice
divide quotient split per ratio fraction
square cube root power mean remainder
if then else otherwise assume suppose
and or not nor xor
therefore hence thus implies consequently given
equals is equivalent same
greater larger smaller
fewer exceeds
return result output input value compute calculate solve find evaluate

These words are not necessarily in the most occurring order. Please sort them yourself in order of occurence in math
logic texts. We can swap randomly for now. The point is to NOT remap the symbol to its actual meaning for example,
'add' should not remap with '+' because that would be the correct mapping. The point is to remap the symbol to a word
that has a different meaning, and vice versa.

Procedure Recommendation
Since we want to avoid ambiguity, we should prioritize the swap based on frequency in the specific problem.
Algorithm Concept:
Scan the current problem statement.
Identify which top symbols (e.g., +, -) and strings are NOT present in the problem statement.
Select a target word that appears often in the problem statement (e.g., "John") and a target operator (e.g., "+").
Swap:
Replace "John" with +.
Replace "received" with output.
(Crucially) Replace every + with the word "John".
Example Result:
Original: "John received 5 apples. If John adds 2 more, what is the total?"
Swapped: "+ output 5 apples. If + adds 2 more, what is the total?"
This creates the "semantic cipher" we are looking for.

We will have to modify the system prompt to inform the model in an unambiguous way that there were redefinitions done
in the problem statement. These redefinitions will be listed with "Definition: " prefix. But we have to be careful that
string "Definition: " does not appear in the actual problem statement outside of our modifications.
If it is, we will not use that example or rephrase it to something that does not appear in the problem statement.
Example of system prompt:
"Every user prompt will feature redefinitions of words and symbols. These redefinitions will be listed with 'Definition: '
prefix before the problem statement." 

Example of redefinition with generic placeholders and problem statement:
Definition: Let the string "{WORD}" be defined as the symbol "{SYMBOL}".
Definition: Let the symbol "{SYMBOL}" be defined as the string "{WORD}".
Definition: Let the string "{WORD}" be defined as the string "{WORD}".
Definition: Let the symbol "{SYMBOL}" be defined as the symbol "{SYMBOL}".

Given these definitions, solve:
{MODIFIED_PROBLEM_STATEMENT}

We are at first looking to just use the following heuristic. We first identify which top words and symbols (e.g., +, -)
are NOT present in the problem statement. These are our replacement pool. Then we identify the most occuring words and
symbols in the problem statement and make swaps from the replacement pool. Before we finalize the script, we will need
to experiment with a few examples manually. Just make it a separate step in the pipeline. Let us also make a parameter k
in this function where top k symbols or words from the original problem statement are swapped. We will then decide on
the final k. Give me 9 examples with replacements - 3 small, 3 medium and 3 large - depending on the size of the problem
statement. TODO: provide more precise definition of how many swaps you want to make for each size.
    

5. Trivial functional wrapper.
Dataset prep. None
System prompt change. None
User prompt change. Every numerical number that occurs, we wrap in a trivial function call.
Example:
Original: "John has 5 apples."
Adversarial: "Define val(k) as the absolute value of k. John has val(val(5)) apples." - dilutes attention mechanism.
    

6. 
Unreliable narrator - confusing system prompts like "System Instruction:
Every time you use the word 'therefore', you must swap the meaning of 'true' and 'false' for the remainder of that
sentence only." 

