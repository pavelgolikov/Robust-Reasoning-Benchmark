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
  
  
We need to write another script. It will take one of the outputs from JSON results file from compound results folder (as
an example one of the outputs from this
/home/golikovp/Antigravity/Robust-Reasoning-Benchmark/experiments/compound/results/Qwen_Qwen3-30B-A3B-Thinking-2507/HuggingFaceH4_aime_2024/Qwen_Qwen3-30B-A3B-Thinking-2507_HuggingFaceH4_aime_2024_compound_s42_20260421_163210.json)
and perform an analysis of the model's attention heads over the distractor and target problems in the generated text.
I have already created a script called attention_map.py which can find the token boundary between distractor and target
problems, but it needs to be expanded to analyze the attention scores. To obtain the attention scores, use the same
method as we use in gather_attentions.py. For now expand attention_map.py. Given input
parameters: model_name, json_result, and others you can think of, it should produce attention distributions for tokens
in the distractor and target problems. Let's save the attention distributions in .pt format. Do you have any questions
before you begin?

Here is the algorithm to implement:
The Algorithm: Memory-Efficient Attention Dilution Tracking
Phase 1: Sequence Partitioning
Load your pre-generated text (Prompt + Distractor Problems + Target Problem + Model Output).
Use the get_token_boundary logic (from attention_map.py) to find the exact token index where the Target Problem begins.
Define your sets:
Distractor Indices ( I_D ): Token index 0 up to the Target boundary.
Target Indices ( I_T ): From the Target boundary to the end of the sequence (length N).
Phase 2: Model Setup and Interception
4. Load the language model into GPU memory using standard, memory-efficient attention (FlashAttention or SDPA). Do not use eager mode, and do not ask it to output attention matrices.
5. Attach an interception mechanism (a PyTorch forward hook) to the specific module that projects the Hidden States into Queries (Q) and Keys (K) for every layer.
Phase 3: The Forward Pass and On-The-Fly Computation
6. Pass the full token sequence (length N) into the model in a single forward pass.
7. As the model executes layer L, your interceptor pauses the forward pass immediately after Q ∈ R^{B × H × N × d_k} and K ∈ R^{B × H × N × d_k} are computed.
8. Inside the interceptor, isolate only the queries belonging to the target problem: Q_{target} = Q[:,:,I_T ,:].
9. To avoid Out-Of-Memory errors, split Q_{target} into small chunks (e.g., 500 tokens at a time).
10. For each chunk of queries:
* Multiply the query chunk by the full key matrix K_{T} .
* Apply the causal mask mathematically (setting scores for future tokens to -inf).
* Apply the Softmax to get the true probability distribution.
* Sum the probabilities strictly across the columns corresponding to the Distractor Indices (I_D).
* Save this aggregated result (a small vector) to CPU RAM.
* Delete the chunk's attention matrix from VRAM to free space.
11. Allow the model's native forward pass to continue to the next layer.
Phase 4: Output
12. Once the forward pass finishes, you will have an L × H × ∣ I_T ∣ matrix containing exactly what you need: for every token generated while solving the target problem, exactly what percentage of attention was looking back at the distractor problems.

I have already written the interceptor in `experiments/compound/attentions/attention_interceptor.py`. Please reuse it.