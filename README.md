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
This repo contains a machine learning project. The point of the project is to try and confuse LLMs or agents with
linguistic tricks to reduce their performance on math reasoning tasks. Tricks that would have no effect on human (or
general reasoner) performance. Please analyze the repository for structure and implementations of various techniques to
confuse language models. Analysis is a utility folder. Tell me when you have taken a look at the repo and ready to
proceed. The main 3 scripts are evaluate.py, which tests model's ability to undo the transformations themselves;
evaluate_api.py, which is used to evaluate closed-source models through api calls, and evaluate_context_api.py, which
tests context pollution of closed-source models by supplying several math-based distractor questions and after the
distractors, the real non-transformed question is asked. Please familiarize yourself with the repo. Ask any questions
you may have.

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
openai/gpt-oss-120b - 120B
deepseek-ai/DeepSeek-R1-Distill-Llama-70B
GAIR/LIMO-v2 - 32B
Qwen/Qwen3-30B-A3B-Thinking-2507
tiiuae/Falcon-H1R-7B

All compound eval done at 0.6 temp and 0.95 top_p - will need to do 0.7 temp later
New models for compound eval:
nvidia/OpenReasoning-Nemotron-32B
mistralai/Ministral-3-14B-Reasoning-2512


NOTE:
Next experiments:
long context

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


nohup python experiments/evaluate_context_api.py \
   --model gemini-3.1-pro-preview \
   --context_size 750000 \
   --n_samples 4 \
   --max_tokens 32000 \
   --context_file experiments/context_saturation/contexts/context_math_750K_google.json \
   --sleep 60 \
   --cache_ttl 36000 \
   --context_type math \
   --no_preview &> eval_api_gemini31.out &
   
   
python experiments/evaluate_context_api.py \
   --model claude-opus-4-6 \
   --context_size 750000 \
   --n_samples 4 \
   --max_tokens 32000 \
   --context_file experiments/context_saturation/contexts/context_math_750K_anthropic.json \
   --context_type math \
   --batch


python experiments/evaluate_context_api.py \
   --model gpt-5-4 \
   --context_size 750000 \
   --n_samples 4 \
   --max_tokens 32000 \
   --context_file experiments/context_saturation/contexts/context_math_750K_openai.json \
   --context_type math \
   --batch \


Hello World I am math p
.                     r
n                     o
a                     b
c                     l
                      e
uoY .em evlos esaelP .m


Think about a single metric that would capture the essence of the experiment and rank models against each other.



Need to expand functionality of visualize.py. We will add another plot type. This one is specifically for "compound"
experiment and we will call the plot type 'compound' as well. It will plot models on the x-axis and accuracy on the y-axis.
For each model, we plot several points, depending on how many compound experiment results are in the results directory.
Experiments differ by the number of distractor questions that were asked before the target question.
Models were tested with 1, 2, or 3 distractor questions. Some models were tested with only 1 or 2, not necessarily 3
distractors. For the baseline, please take the values from baseline directory, but "compound" subdirectory in results,
not "perturb".
here is the description of the plot - it will be a line chart with markers - one marker for each experiment.
X-Axis: "Target Problem Position in Context" (1st [Baseline], 2nd [1 Distractor], 3rd [2 Distractors], etc.)
Y-Axis: "Target Problem Accuracy (%)"
The Data: One distinct line (with dots/markers at each data point) for each model.
NO fallbacks - if something doesn't go as planned (file missing etc..) - throw errors.
Do you have any questions?

Two more changes:
1. Swap Claude and GPT-5.4 (only for this compound plot though).
2. Near the last marker for every model, place a number in red color of how much the total accuracy drop is 