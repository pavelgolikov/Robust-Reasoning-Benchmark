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

Here are the first model and the first dataset.
Model: DeepSeek: R1 0528 (free) from OpenRouter website (deepseek/deepseek-r1-0528:free).
    Here is a link - https://openrouter.ai/deepseek/deepseek-r1-0528:free
Dataset: AIME 2024 (HuggingFaceH4/aime_2024).
    Here is a link: https://huggingface.co/datasets/HuggingFaceH4/aime_2024


NOTE: We have added pre-processing step to the dataset to remove all empty lines from the problem statements.

NOTE: Technique to make an agent violate a previously given instruction - this essentially happens already when it can't
decode and solve.

0. Try to see if the degradation is more or less with text other than auto-generated math systems.
0.1. Need a script to analyze the output to see if the model is failing at decoding or solving after decoding.
0.2. Need a script to analyze the output to see how well the model performs decoding only.
1. Figure out if the model can't decode or can't solve after decoding?
2. Try see if the model is able to perform simple python scripts. Scalability here can come from a million-sized list of values.


PAPER OUTLINE:
Introduction
LLMs demonstrated remarkable capabilities.
However, their performance is neither robust nor reliable.
We propose a Robust Reasoning Framework (RRF) that evaluates the model's ability to reason robustly in a compound manner.
We release a set of python scripts that modify the math reasoning datasets problems with transformations
that we claim are trivial (if annoying) for a human or general reasoner to decode/defeat.
Because the transformations are very simple, we hypothesize that a model that is truly intelligent should be able to decode/defeat them
without ANY loss in performance on the actual reasoning task at hand.
We have verified that every transformation is easily reversible by a python script, normally with at most 20 lines of human readable code.

Our contributions are as follows:
1. We propose a set of linguistic trap techniques that can be used to evaluate the robustness of reasoning models.
2. We release a set of scripts that implement these techniques on existing math reasoning datasets.
3. We evaluate several existing reasoning models on the modified datasets and analyze their performance degradation.

Background and Related Work:
Define reasoning - look in Apple paper.

Related Works:
GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in Large Language Models. https://arxiv.org/abs/2410.05229
PlanBench: An Extensible Benchmark for Evaluating Large Language Models on Planning and Reasoning about Change. https://arxiv.org/abs/2206.10498
Alice in Wonderland: Simple Tasks Showing Complete Reasoning Breakdown in State-Of-the-Art Large Language Models. https://arxiv.org/abs/2406.02061
Embers of Autoregression: Understanding Large Language Models Through the Problem They are Trained to Solve. https://arxiv.org/abs/2309.13638
On the Paradox of Learning to Reason from Data. https://www.ijcai.org/proceedings/2023/375
Can Large Language Models Reason and Plan? https://arxiv.org/abs/2403.04121
Context Rot. https://research.trychroma.com/context-rot - related - TODO: need to test vs trivial context rot.
Artificial intelligence and illusions of understanding in scientific research. https://www.nature.com/articles/s41586-024-07146-0
A Causal Framework to Quantify the Robustness of Mathematical Reasoning with Language Models. https://arxiv.org/abs/2210.12023 - far more complex, plus old.
Evaluating llms' mathematical and coding competency through ontology-guided interventions.https://arxiv.org/abs/2401.09395 - much more complex, no guarantee that the transformations are correctly done at scale.
Functional Benchmarks for Robust Evaluation of Reasoning Performance, and the Reasoning Gap. https://arxiv.org/abs/2402.19450 - too comlicated, closed source method, our method is far simpler and faster to apply.

Prior works have proposed complex frameworks to evaluate the robustness of reasoning models. Modifications proposed in
prior works themselves rely on existing LLMs to rephrase or modify the problem statements, which may introduce unintended biases or errors.
We argue that existing modification frameworks are too complex in nature and modify the problems at too deep a level.
Our approach is fundamentally different from prior works because we (1) propose a python-only framework that can modify
existing math resoning datasets with minimal effort and (2) ensure that the modifications are trivial for a human to decode/defeat.

Linguistic Trap Techniques:
We propose the following techniques to modify existing math reasoning problems:
Describe techniques.


Evaluation:
Experiments. TODO: Decide models and datasets.
Datasets:
AIME 2024
AIME 2025
GSM8K
MATH
Olympiad Bench
College Bench
Omni-Math

Models:
Open:
GAIR/LIMO-v2
tiiuae/Falcon-H1R-7B
openai/gpt-oss-120b
XiaomiMiMo/MiMo-V2-Flash
deepseek-ai/DeepSeek-R1

Closed:
GPT-5.1
Gemini 3 Pro







Conclusion:




