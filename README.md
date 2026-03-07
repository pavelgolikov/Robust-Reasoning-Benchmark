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
tests context pollution by supplying several math-based distractor questions and after the distractors, the real
non-transformed question is asked.
For now we test context saturation only on the closed-source models as they are the most interesting to evaluate.
Please familiarize yourself with the repo. Ask any questions you may have.

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



=== Processing Batch [0] (anthropic - not_not) ===
1. Downloading raw output from anthropic...
2026-03-05 22:05:29,682 - INFO - HTTP Request: GET https://api.anthropic.com/v1/messages/batches/msgbatch_019gVyroFxDfUiomiCsaxV9B "HTTP/1.1 200 OK"
2026-03-05 22:05:29,888 - INFO - HTTP Request: GET https://api.anthropic.com/v1/messages/batches/msgbatch_019gVyroFxDfUiomiCsaxV9B/results "HTTP/1.1 200 OK"
2. Parsing raw outputs...
3. Grading results...
   Accuracy (all):      97.50% (117/120)
   Accuracy (attempted): 97.50% (117/120)
   Refusals: 0
   Failures (non-refusal errors): 0
4. Saved final graded results to: ./experiments/not_not/results/claude-opus-4-6/HuggingFaceH4_aime_2024/claude-opus-4-6_HuggingFaceH4_aime_2024_not_not_20260305_215825.json
5. Removed tracking file: ./experiments/not_not/results/claude-opus-4-6/HuggingFaceH4_aime_2024/batch_tracking_20260305_215825.json
   Removed jobs file: /home/golikovp/Antigravity/Linguistic_traps/experiments/not_not/results/claude-opus-4-6/HuggingFaceH4_aime_2024/jobs_msgbatch_019gVyroFxDfUiomiCsaxV9B.json

=== Processing Batch [2] (anthropic - interleaved_context_line) ===
1. Downloading raw output from anthropic...
2026-03-05 22:05:30,153 - INFO - HTTP Request: GET https://api.anthropic.com/v1/messages/batches/msgbatch_01QYPAzKz1ng9VAT1vJUkLgw "HTTP/1.1 200 OK"
2026-03-05 22:05:30,331 - INFO - HTTP Request: GET https://api.anthropic.com/v1/messages/batches/msgbatch_01QYPAzKz1ng9VAT1vJUkLgw/results "HTTP/1.1 200 OK"
2. Parsing raw outputs...
3. Grading results...
   Accuracy (all):      98.33% (118/120)
   Accuracy (attempted): 98.33% (118/120)
   Refusals: 0
   Failures (non-refusal errors): 0
4. Saved final graded results to: ./experiments/interleaved_context_line/results/claude-opus-4-6/HuggingFaceH4_aime_2024/claude-opus-4-6_HuggingFaceH4_aime_2024_interleaved_context_line_20260305_215825.json
5. Removed tracking file: ./experiments/interleaved_context_line/results/claude-opus-4-6/HuggingFaceH4_aime_2024/batch_tracking_20260305_215825.json
   Removed jobs file: /home/golikovp/Antigravity/Linguistic_traps/experiments/interleaved_context_line/results/claude-opus-4-6/HuggingFaceH4_aime_2024/jobs_msgbatch_01QYPAzKz1ng9VAT1vJUkLgw.json

=== Processing Batch [4] (anthropic - wrappers) ===
1. Downloading raw output from anthropic...
2026-03-05 22:05:30,515 - INFO - HTTP Request: GET https://api.anthropic.com/v1/messages/batches/msgbatch_01JkBBg7MsAX9dZbLdgkvjY7 "HTTP/1.1 200 OK"
2026-03-05 22:05:30,711 - INFO - HTTP Request: GET https://api.anthropic.com/v1/messages/batches/msgbatch_01JkBBg7MsAX9dZbLdgkvjY7/results "HTTP/1.1 200 OK"
2. Parsing raw outputs...
3. Grading results...
   Accuracy (all):      98.33% (118/120)
   Accuracy (attempted): 98.33% (118/120)
   Refusals: 0
   Failures (non-refusal errors): 0
4. Saved final graded results to: ./experiments/wrappers/results/claude-opus-4-6/HuggingFaceH4_aime_2024/claude-opus-4-6_HuggingFaceH4_aime_2024_wrappers_20260305_215825.json
5. Removed tracking file: ./experiments/wrappers/results/claude-opus-4-6/HuggingFaceH4_aime_2024/batch_tracking_20260305_215825.json
   Removed jobs file: /home/golikovp/Antigravity/Linguistic_traps/experiments/wrappers/results/claude-opus-4-6/HuggingFaceH4_aime_2024/jobs_msgbatch_01JkBBg7MsAX9dZbLdgkvjY7.json

=== Processing Batch [6] (anthropic - sentence_reversal) ===
1. Downloading raw output from anthropic...
2026-03-05 22:05:30,940 - INFO - HTTP Request: GET https://api.anthropic.com/v1/messages/batches/msgbatch_01KsDagNZ2ghMncqiozvrnpz "HTTP/1.1 200 OK"
2026-03-05 22:05:31,096 - INFO - HTTP Request: GET https://api.anthropic.com/v1/messages/batches/msgbatch_01KsDagNZ2ghMncqiozvrnpz/results "HTTP/1.1 200 OK"
2. Parsing raw outputs...
3. Grading results...
   Accuracy (all):      99.17% (119/120)
   Accuracy (attempted): 99.17% (119/120)
   Refusals: 0
   Failures (non-refusal errors): 0
4. Saved final graded results to: ./experiments/sentence_reversal/results/claude-opus-4-6/HuggingFaceH4_aime_2024/claude-opus-4-6_HuggingFaceH4_aime_2024_sentence_reversal_20260305_215825.json
5. Removed tracking file: ./experiments/sentence_reversal/results/claude-opus-4-6/HuggingFaceH4_aime_2024/batch_tracking_20260305_215825.json
   Removed jobs file: /home/golikovp/Antigravity/Linguistic_traps/experiments/sentence_reversal/results/claude-opus-4-6/HuggingFaceH4_aime_2024/jobs_msgbatch_01KsDagNZ2ghMncqiozvrnpz.json

=== Processing Batch [8] (anthropic - interleaved_context_word) ===
1. Downloading raw output from anthropic...
2026-03-05 22:05:31,306 - INFO - HTTP Request: GET https://api.anthropic.com/v1/messages/batches/msgbatch_01P3Hz9Fu4n8PpuimNGCjsho "HTTP/1.1 200 OK"
2026-03-05 22:05:31,628 - INFO - HTTP Request: GET https://api.anthropic.com/v1/messages/batches/msgbatch_01P3Hz9Fu4n8PpuimNGCjsho/results "HTTP/1.1 200 OK"
2. Parsing raw outputs...
3. Grading results...
   Accuracy (all):      90.00% (108/120)
   Accuracy (attempted): 90.00% (108/120)
   Refusals: 0
   Failures (non-refusal errors): 0
4. Saved final graded results to: ./experiments/interleaved_context_word/results/claude-opus-4-6/HuggingFaceH4_aime_2024/claude-opus-4-6_HuggingFaceH4_aime_2024_interleaved_context_word_20260305_215825.json
5. Removed tracking file: ./experiments/interleaved_context_word/results/claude-opus-4-6/HuggingFaceH4_aime_2024/batch_tracking_20260305_215825.json
   Removed jobs file: /home/golikovp/Antigravity/Linguistic_traps/experiments/interleaved_context_word/results/claude-opus-4-6/HuggingFaceH4_aime_2024/jobs_msgbatch_01P3Hz9Fu4n8PpuimNGCjsho.json

=== Processing Batch [10] (anthropic - word_reversal) ===
1. Downloading raw output from anthropic...
2026-03-05 22:05:31,849 - INFO - HTTP Request: GET https://api.anthropic.com/v1/messages/batches/msgbatch_0177ZjeXybG9tHQRo7YFL431 "HTTP/1.1 200 OK"
2026-03-05 22:05:32,043 - INFO - HTTP Request: GET https://api.anthropic.com/v1/messages/batches/msgbatch_0177ZjeXybG9tHQRo7YFL431/results "HTTP/1.1 200 OK"
2. Parsing raw outputs...
3. Grading results...
   Accuracy (all):      50.83% (61/120)
   Accuracy (attempted): 87.14% (61/70)
   Refusals: 50
   Failures (non-refusal errors): 0
4. Saved final graded results to: ./experiments/word_reversal/results/claude-opus-4-6/HuggingFaceH4_aime_2024/claude-opus-4-6_HuggingFaceH4_aime_2024_word_reversal_20260305_215825.json
5. Removed tracking file: ./experiments/word_reversal/results/claude-opus-4-6/HuggingFaceH4_aime_2024/batch_tracking_20260305_215825.json
   Removed jobs file: /home/golikovp/Antigravity/Linguistic_traps/experiments/word_reversal/results/claude-opus-4-6/HuggingFaceH4_aime_2024/jobs_msgbatch_0177ZjeXybG9tHQRo7YFL431.json

=== Processing Batch [12] (anthropic - opposites) ===
1. Downloading raw output from anthropic...
2026-03-05 22:05:32,244 - INFO - HTTP Request: GET https://api.anthropic.com/v1/messages/batches/msgbatch_01Ya9KGbsSnaGNwV1okVYaWj "HTTP/1.1 200 OK"
2026-03-05 22:05:32,531 - INFO - HTTP Request: GET https://api.anthropic.com/v1/messages/batches/msgbatch_01Ya9KGbsSnaGNwV1okVYaWj/results "HTTP/1.1 200 OK"
2. Parsing raw outputs...
3. Grading results...
   Accuracy (all):      94.17% (113/120)
   Accuracy (attempted): 94.17% (113/120)
   Refusals: 0
   Failures (non-refusal errors): 0
4. Saved final graded results to: ./experiments/opposites/results/claude-opus-4-6/HuggingFaceH4_aime_2024/claude-opus-4-6_HuggingFaceH4_aime_2024_opposites_20260305_220111.json
5. Removed tracking file: ./experiments/opposites/results/claude-opus-4-6/HuggingFaceH4_aime_2024/batch_tracking_20260305_220111.json
   Removed jobs file: /home/golikovp/Antigravity/Linguistic_traps/experiments/opposites/results/claude-opus-4-6/HuggingFaceH4_aime_2024/jobs_msgbatch_01Ya9KGbsSnaGNwV1okVYaWj.json

=== Processing Batch [14] (anthropic - split_reversal) ===
1. Downloading raw output from anthropic...
2026-03-05 22:05:32,744 - INFO - HTTP Request: GET https://api.anthropic.com/v1/messages/batches/msgbatch_01TjV7J4chJ8YyviGVHeWK26 "HTTP/1.1 200 OK"
2026-03-05 22:05:32,934 - INFO - HTTP Request: GET https://api.anthropic.com/v1/messages/batches/msgbatch_01TjV7J4chJ8YyviGVHeWK26/results "HTTP/1.1 200 OK"
2. Parsing raw outputs...
3. Grading results...
   Accuracy (all):      3.33% (4/120)
   Accuracy (attempted): 100.00% (4/4)
   Refusals: 116
   Failures (non-refusal errors): 0
4. Saved final graded results to: ./experiments/split_reversal/results/claude-opus-4-6/HuggingFaceH4_aime_2024/claude-opus-4-6_HuggingFaceH4_aime_2024_split_reversal_20260305_215825.json
5. Removed tracking file: ./experiments/split_reversal/results/claude-opus-4-6/HuggingFaceH4_aime_2024/batch_tracking_20260305_215825.json
   Removed jobs file: /home/golikovp/Antigravity/Linguistic_traps/experiments/split_reversal/results/claude-opus-4-6/HuggingFaceH4_aime_2024/jobs_msgbatch_01TjV7J4chJ8YyviGVHeWK26.json

=== Processing Batch [16] (anthropic - interleaved_context_symbol) ===
1. Downloading raw output from anthropic...
2026-03-05 22:05:33,085 - INFO - HTTP Request: GET https://api.anthropic.com/v1/messages/batches/msgbatch_01G18oLHDcuCzmZyAk2qf7ss "HTTP/1.1 200 OK"
2026-03-05 22:05:33,317 - INFO - HTTP Request: GET https://api.anthropic.com/v1/messages/batches/msgbatch_01G18oLHDcuCzmZyAk2qf7ss/results "HTTP/1.1 200 OK"
2. Parsing raw outputs...
3. Grading results...
   Accuracy (all):      30.83% (37/120)
   Accuracy (attempted): 58.73% (37/63)
   Refusals: 57
   Failures (non-refusal errors): 0
4. Saved final graded results to: ./experiments/interleaved_context_symbol/results/claude-opus-4-6/HuggingFaceH4_aime_2024/claude-opus-4-6_HuggingFaceH4_aime_2024_interleaved_context_symbol_20260305_215825.json
5. Removed tracking file: ./experiments/interleaved_context_symbol/results/claude-opus-4-6/HuggingFaceH4_aime_2024/batch_tracking_20260305_215825.json
   Removed jobs file: /home/golikovp/Antigravity/Linguistic_traps/experiments/interleaved_context_symbol/results/claude-opus-4-6/HuggingFaceH4_aime_2024/jobs_msgbatch_01G18oLHDcuCzmZyAk2qf7ss.json

=== Processing Batch [18] (anthropic - rail_fence) ===
1. Downloading raw output from anthropic...
2026-03-05 22:05:33,527 - INFO - HTTP Request: GET https://api.anthropic.com/v1/messages/batches/msgbatch_015dQmfyLnw7oMg41pg5w2tS "HTTP/1.1 200 OK"
2026-03-05 22:05:33,707 - INFO - HTTP Request: GET https://api.anthropic.com/v1/messages/batches/msgbatch_015dQmfyLnw7oMg41pg5w2tS/results "HTTP/1.1 200 OK"
2. Parsing raw outputs...
3. Grading results...
   Accuracy (all):      1.67% (2/120)
   Accuracy (attempted): 4.17% (2/48)
   Refusals: 72
   Failures (non-refusal errors): 0
4. Saved final graded results to: ./experiments/rail_fence/results/claude-opus-4-6/HuggingFaceH4_aime_2024/claude-opus-4-6_HuggingFaceH4_aime_2024_rail_fence_20260305_215825.json
5. Removed tracking file: ./experiments/rail_fence/results/claude-opus-4-6/HuggingFaceH4_aime_2024/batch_tracking_20260305_215825.json
   Removed jobs file: /home/golikovp/Antigravity/Linguistic_traps/experiments/rail_fence/results/claude-opus-4-6/HuggingFaceH4_aime_2024/jobs_msgbatch_015dQmfyLnw7oMg41pg5w2tS.json

=== Processing Batch [20] (anthropic - baseline) ===
1. Downloading raw output from anthropic...
2026-03-05 22:05:33,862 - INFO - HTTP Request: GET https://api.anthropic.com/v1/messages/batches/msgbatch_01KxYBy7zLurvoY2kqebSenJ "HTTP/1.1 200 OK"
2026-03-05 22:05:34,019 - INFO - HTTP Request: GET https://api.anthropic.com/v1/messages/batches/msgbatch_01KxYBy7zLurvoY2kqebSenJ/results "HTTP/1.1 200 OK"
2. Parsing raw outputs...
3. Grading results...
   Accuracy (all):      99.17% (119/120)
   Accuracy (attempted): 99.17% (119/120)
   Refusals: 0
   Failures (non-refusal errors): 0
4. Saved final graded results to: ./experiments/baseline/results/claude-opus-4-6/HuggingFaceH4_aime_2024/claude-opus-4-6_HuggingFaceH4_aime_2024_baseline_20260305_220111.json
5. Removed tracking file: ./experiments/baseline/results/claude-opus-4-6/HuggingFaceH4_aime_2024/batch_tracking_20260305_220111.json
   Removed jobs file: /home/golikovp/Antigravity/Linguistic_traps/experiments/baseline/results/claude-opus-4-6/HuggingFaceH4_aime_2024/jobs_msgbatch_01KxYBy7zLurvoY2kqebSenJ.json
