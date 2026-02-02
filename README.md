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
1. Figure out if the model can't decode or can't solve after decoding?
2. Try see if the model is able to perform simple python scripts. Scalability here can come from a million-sized list of
   values. (?)

Repo description for agent.
This repo contains a machine learning project. The point of the project is to try and confuse LLMs or agents with linguistic tricks to reduce their performance on math reasoning tasks. Tricks that would have no effect on human (or general reasoner) performance. Please analyze the repository for structure and implementations of various techniques to confuse language models. Analysis is a utility folder. Tell me when you have taken a look at the repo and ready to proceed.


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


# Methodology:
# Dataset Preprocessing:
We preprocess the datasets to remove the possibilities of escape sequences interfering with our transformations.
For example, we observed in some problem statements, a variable, for example "b" can be surrounded by escaped brackets like "\(b\)".
If the transformation involves reversing the symbols in the problem statement, the snippet becomes ")\b(\".
Since \b is a backspace escape sequence, if text gets rendered or interpreted, the character before \b gets erased,
which is unfair to the model as there is loss of information.
We perform 3 preprocessing steps to avoid such issues:
1. Remove all empty lines from problem statements and replace them with "; " - a semicolon followed by a space.
   This ensures that the model sees a break in the text without having to deal with newlines.
2. Remove all LaTeX comments from problem statements. LaTeX comments start with a "%" symbol and continue until the end of the line.
   Since we have removed all newlines, we remove the comments to avoid any unintended consequences. That being said, we
   have not observed comments in any of the problem sets.
3. Inserts a space between specific characters (n, t, b, r, a, f) and a backslash to prevent accidental escape sequence
   formation if text is reversed.
We argue that these preprocessing steps do not change the problem statements in any meaningful way.
Moreover, when presented to a human, these preprocessing steps are trivial to ignore or decode.

# Linguistic Trap Implementations:
When implementing the linguistic trap techniques, we follow these general guidelines to ensure consistency and fairness
to the model during evaluation:
1. Each technique should not remove or add any information from the original problem statement or introduce any ambiguities
    that would confuse an actual general reasoner (human or AI).
2. Each technique needs to be easily reversible/defeatable by a human with minimal effort and without any specialized tools.
    We verify this by writing a simple python script for each technique that can reverse/defeat the transformation and
    restore the original problem statement.
3. Each technique should be simple enough that a human can decode/defeat it without losing performance on the 
    reasoning task at hand (e.g., solving a math problem itself).
4. Model is informed of the transformation and given its detailed description in the user prompt. Model is instructed
    to first defeat the transformation, recover the original problem statement, and only then solve the problem.
Description of each technique implementation is given in Section 3.

# Evaluation Setup:
We evaluate each model in two configurations:
1. Direct Prompting. We directly prompt the model to solve the modified problem statement by first defeating the transformation
    and then solving the problem.
2. Agentic Python Interpreter. We setup an agentic loop where the model can use a python interpreter to defeat the
    transformation and solve the problem after recovering the original problem statement. Our agentic loop follows the ReAct
    paradigm (Yao et al., 2023) where the model can interleave reasoning steps (Thoughts) with actions (code execution).
    We use the same prompt structure as in direct prompting, except the model is informed in the user query that it has
    access to a python interpreter to help defeat the transformation. We use a custom loop instead of frameworks like
    smolagents to exercise direct control over the prompt structures.



Methodology justification for manual loop and markdown:
Yes, this is a very strong, scientifically valid position. In the research community (specifically regarding Code LLMs and Mathematical Reasoning), the **Markdown/Free-Text Code approach** is often preferred over **Structured JSON/Tool-Use** for complex reasoning tasks.

Here are the three main arguments you can use in your paper, supported by foundational citations.

### 1. The "Pre-training Distribution" Argument
**Argument:** LLMs are pre-trained on massive datasets of source code (GitHub, StackOverflow) where code appears as raw text or inside Markdown blocks, not wrapped inside JSON strings. Forcing a model to write code inside a JSON string forces it out of its distribution and imposes a "Syntax Tax" (escaping newlines and quotes), which degrades reasoning performance.

*   **Source:** **"Llama 3 Technical Report" (Meta AI, 2024)** or **"DeepSeek-Coder-V2" (2024)**.
    *   *Relevance:* These technical reports emphasize that the models are fine-tuned on "Chat" formats where code is interleaved with natural language (Markdown), rather than strict API schema following.
*   **Source:** **"Codex: Evaluating Large Language Models Trained on Code" (Chen et al., 2021)**.
    *   *Relevance:* The paper that launched the modern code-agent era. It evaluates models based on their ability to generate functional code blocks (Markdown/Raw), establishing this as the gold standard for measuring coding capability (HumanEval benchmark).

### 2. The "Program-Aided Language (PAL)" Paradigm
**Argument:** Mathematical reasoning performance is maximized when the model is allowed to interleave natural language reasoning (Chain of Thought) with code generation. This "Literate Programming" style is natively supported by Markdown blocks but is difficult to achieve in rigid JSON Tool calls (which often force a separation between "thought" and "action" or force thoughts into a single string field).

*   **Source:** **"PAL: Program-aided Language Models" (Gao et al., 2023)**.
    *   *Citation:* Gao, L., et al. (2023). PAL: Program-aided Language Models. *ICML*.
    *   *Key Finding:* They demonstrate that delegating reasoning to a Python interpreter (via natural code generation) outperforms standard LLM reasoning. Their implementation uses free-form Python generation, not structured tool schemas.
*   **Source:** **"Program of Thoughts Prompting: Disentangling Computation from Reasoning" (Chen et al., 2022)**.
    *   *Citation:* Chen, W., et al. (2022). Program of Thoughts Prompting. *NeurIPS*.
    *   *Key Finding:* They argue that allowing the model to generate executable steps as "thoughts" (code) yields higher accuracy. This relies on the standard text-completion format.

### 3. The "Execution Feedback" Loop (ReAct)
**Argument:** The ReAct (Reasoning + Acting) pattern, which is the foundation of agentic behavior, was originally designed around free-text generation, not JSON function calling. The original implementation and subsequent benchmarks show that the flexibility of text-based actions is robust for multi-step problem solving.

*   **Source:** **"ReAct: Synergizing Reasoning and Acting in Language Models" (Yao et al., 2023)**.
    *   *Citation:* Yao, S., et al. (2023). ReAct: Synergizing Reasoning and Acting in Language Models. *ICLR*.
    *   *Key Finding:* The paper establishes the trace format `Thought: ... Action: ...` (text-based) as a robust method for solving problems. It shows that enforcing rigid formats can sometimes hurt the model's ability to "think" before acting.

### How to write this in your Paper (Draft Snippet)

You can include a paragraph like this in your **Methodology** or **Experimental Setup** section:

> "To implement the agentic loop, we utilize a free-form code generation approach (Markdown code blocks) rather than structured tool-use schemas (e.g., JSON function calling). We chose this format for three reasons:
>
> 1.  **Alignment with Pre-training:** SOTA coding models are primarily trained on source code and Markdown documentation (Chen et al., 2021), making free-form generation their most robust modality.
> 2.  **Reduction of Syntactic Overhead:** Structured tool calls require the model to handle complex string escaping (e.g., escaping newlines inside JSON values), which introduces a 'syntax tax' that competes with reasoning resources.
> 3.  **Interleaved Reasoning:** Following the Program-Aided Language (PAL) paradigm (Gao et al., 2023), we allow the model to interleave natural language reasoning with executable code. This 'Literate Programming' approach has been shown to improve performance on mathematical reasoning tasks compared to rigid tool-calling structures."

### Summary for your defense
*   **Benchmarks use Markdown:** The industry standard benchmarks (HumanEval, MBPP, SWE-bench) all evaluate models based on their ability to write Markdown/Raw code, not their ability to fill out a JSON form.
*   **JSON is for APIs, Markdown is for Agents:** The industry consensus is that JSON/XML is useful when integrating with *legacy software* (e.g., a Weather API), but Markdown is superior when the model *is the programmer* (e.g., Data Analysis, Math Solving).



Conclusion:


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


TODO:
1. Set up agent for distill-R1-70B that will just accept queries in a multi-turn format, no Python interpreter.

2. Rail Fence Cipher - we can later make it harder by using random letters instead of dots.

Context saturation. Currently trying 20 distractors on 4 GPUs.
What worked:
5 distractors on 2 GPU
110 distractors on 4 GPUs sequentially in 2 hours got cancelled due to time limit @ 262000 max model length with 1 sample


What didn't work:

Trying now:
batch 10, 30 distractors, 4 GPUs, 5 samples, 262000 max model length, 6 hours time limit - 


TODO: Add to paper that we have another insight that we demonstrate:
- Models are able to program to reverse these transformations with Python code, but unable to perform this decoding internally.
This shows that models do not possess true understanding of the text they process?