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
distractors, the real non-transformed question is asked. IMPORTANT NOTE: When you work on this repo, do NOT implement
fallbacks, if something doens't work - throw an error. Please familiarize yourself with the repo. Ask any questions
you have.

TODO:
- Evaluate compound and perturb on AIME 2025 - Currently running compound baseline and 1 distractor
- Try to ask the model itself to annotate the substeps in the reasoning chain and redo some eval with a system prompt
  like that to try separate the subtasks.
  
  dataset names: HuggingFaceH4_aime_2024, MathArena_aime_2025, MATH_500, MathArena_hmmt_feb_2025