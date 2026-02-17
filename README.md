Env prep - need to load modules:
module load gcc arrow/21.0.0
module load gcc opencv/4.12.0

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


# Repo description for agent.
This repo contains a machine learning project. The point of the project is to try and confuse LLMs or agents with linguistic tricks to reduce their performance on math reasoning tasks. Tricks that would have no effect on human (or general reasoner) performance. Please analyze the repository for structure and implementations of various techniques to confuse language models. Analysis is a utility folder. Tell me when you have taken a look at the repo and ready to proceed. The main 3 scripts are evaluate.py, which tests model's ability to undo the transformations themselves; evaluate_agent.py, which tests model with access to Python interpreter and is instructed to undo transformations in Python before solving the problem; and evaluate_conversaion.py, which tests context pollution by supplying several targeted distractor questions with made up math system and after the distractors, the real non-transformed question is asked.


Evaluation:
Datasets:
HuggingFaceH4/aime_2024 - 30/30
MathArena/aime_2025 - 30/30
MATH 500 - 500/500 - LLM verification for some
MathArena/hmmt_feb_2025 - 30/30 - LLM verification for some
<!-- meituan-longcat/AMO-Bench - 37/50 - we take 37 problems with 'number' and 'set' answer types to be able to verify answers more robustly. - LLM verification for some -->
For now total across 4 datasets is 500+30+30+30=590 problems.
GSM-Symbolic? We could generate 100 questions using their code

Olympiad Bench
College Bench
Omni-Math

Models:

HF:
Qwen/Qwen3-235B-A22B-Thinking-2507 - 235B
openai/gpt-oss-120b - 120B
deepseek-ai/DeepSeek-R1-Distill-Llama-70B
GAIR/LIMO-v2 - 32B
Qwen/Qwen3-30B-A3B-Thinking-2507
tiiuae/Falcon-H1R-7B

Closed:
GPT-5.1
Gemini 3 Pro
Need to send requests to them to run the evaluation on their models.

Observations:
1. Temperature and max response length both need to be adjustable for each individual task the model performs.
2. Since we want the model to perform complex tasks, we need to teach the model to adjust these parameters based on the task at hand.
Otherwise:
If temp too high, model starts exploring during mechanical tasks - instead of writing Python, it decides to solve manually AGAINST explicit instructions.
If temp is set too low, then model might be too conservative and not explore enough to find the solution.

Model length needs to be kept shorter for the agent compared to the model itself because the reasoning chain itself is
done in steps.

Idea:
Confuse the model by introducing multiple block boundaries in the text? Like multiple defyn blocks scattered across the text?


x, n, m, k - 4 lower_variables
A, B, C, D - 4 upper_variables
Alice, Bob, Carol, David - 4 names
