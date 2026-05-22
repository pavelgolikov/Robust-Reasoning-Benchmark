# Robust Reasoning Benchmark (RRB)

While Large Language Models (LLMs) achieve high performance on standard mathematical benchmarks, their problem-solving abilities depend heavily on context and textual formatting. The **Robust Reasoning Benchmark (RRB)** introduces a pipeline of 13 deterministic textual perturbations to evaluate true fluid intelligence and mathematical reasoning in LLMs, ensuring they do not merely rely on memorized syntactic patterns.

By applying techniques such as structural noise, tokenization disruption, and interleaved distractor problems to challenging math datasets (like AIME 2024 and AIME 2025), RRB provides a rigorous environment to test reasoning robustness of both frontier and open-weight models.

## Structure

The repository is structured into the following main directories:

- `experiments/`: Contains the core evaluation pipelines and transformation logic.
  - Core scripts (`evaluate.py`, `evaluate_api.py`, `api_utils.py`, `util.py`) handle generating prompts, querying local or API-based models, and robust grading using `math_verify`.
  - Individual folders for each of the 13 transformations (e.g., `not_not/`, `compound/`, `snake_vertical/`) contain the mathematical transformation logic and their generated results.
- `demo/`: Self-contained demonstration environment to visually evaluate the transformations and confirm reversibility.
- `old/`: Archive for legacy logic, unused model outputs, and prior experiments.

## Running Evaluations

### Local Models (vLLM)
Use `evaluate.py` to evaluate an open-weight model on the benchmark using vLLM:
```bash
python experiments/evaluate.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Llama-70B \
  --dataset MathArena/aime_2025 \
  --names all \
  --num_gpus 4
```

### API Models
Use `evaluate_api.py` to evaluate proprietary/API-based models (OpenAI, Anthropic, Google) locally or via async batch queues:
```bash
python experiments/evaluate_api.py \
  --model claude-opus-4-6 \
  --dataset HuggingFaceH4/aime_2024 \
  --names not_not,rail_fence,snake_vertical \
  --batch
```

### Polling API Batches
For asynchronous batch evaluations submitted via the `--batch` flag, use `poll_and_grade_batches.py` to monitor their status, download the results once completed, and automatically extract and grade them:
```bash
python experiments/poll_and_grade_batches.py --dir experiments/
```

## Citation

If you find this benchmark useful in your research, please cite our paper:

```bibtex
@misc{golikov2026robust,
      title={Robust Reasoning Benchmark}, 
      author={Pavel Golikov and Evgenii Opryshko and Gennady Pekhimenko and Mark C. Jeffrey},
      year={2026},
      eprint={2604.08571},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```