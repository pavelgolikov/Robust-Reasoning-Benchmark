# RRB rebuttal experiments — results summary

Findings from the two experiments run for the NeurIPS rebuttal: decode-only recovery and the
equal-length passive-context control. Written for whoever is drafting the rebuttal text in the
paper repo. All numbers below are reproducible from this repo; paths are relative to its root.

Scope note: this covers `plan.md` items 1 and 2 only. Items 3 (problem-level uncertainty) and 4
(cutoff clarification) are folded into the CSVs described under *Where the numbers live* rather
than written up here.

## Where the numbers live

| artifact | contents |
|---|---|
| `experiments/analysis/rebuttal_stats/summary_counts.csv` | per-run accuracy, Wilson and problem-level intervals, cutoff rate, accuracy excluding cutoffs |
| `experiments/analysis/rebuttal_stats/paired_comparisons.csv` | problem-level bootstrap differences between conditions (10k resamples over the 30 AIME problems) |
| `experiments/analysis/rebuttal_stats/decode_recovery_metrics.csv` | per-(model, dataset, transform) decode metrics |
| `experiments/decode_recovery/results/` | raw decode runs, 72 cells |
| `experiments/passive_context/results/` | raw passive runs, 6 cells |

Regenerate with `bash experiments/analysis/run_rebuttal_statistics.sh` and
`python3 experiments/analysis/decode_recovery_metrics.py`. Both are standard library only and
re-score stored outputs, so neither needs a GPU or a re-run.

Models are Qwen3-30B-A3B-Thinking-2507, OpenReasoning-Nemotron-32B and OpenReasoning-Nemotron-7B,
on AIME 2024 and AIME 2025, 30 problems x 16 samples = 480 samples per cell.

---

## 1. Passive-context control — usable, supports the paper's claim

**Question.** In the compound setting the model solves N distractor problems before the target,
and accuracy on the target falls. Is that fall caused by context length alone, or specifically by
the model's own reasoning occupying that context?

**Design.** Same target problem, but the prior problems and their solutions are replaced with a
contiguous, non-repeating passage of neutral prose (Gibbon, *Decline and Fall*), sized per sample
so the token count before target-solving begins matches what the paired compound run actually
produced. Matching is per sample, not per condition: for each compound sample the script finds
where in the model's own output it begins the target problem and matches that offset.

**Result.** Neutral text is far less harmful than the model's own chain-of-thought at equal length.

| model | dataset | baseline | d1 | d2 | d3 | passive | passive − d3 |
|---|---|---|---|---|---|---|---|
| Qwen3-30B | 2024 | 90.4 | 86.2 | 84.4 | 81.9 | 84.8 | +2.9 |
| Nemotron-32B | 2024 | 89.0 | 80.8 | 79.8 | 73.5 | 86.5 | +12.9 |
| Nemotron-7B | 2024 | 81.9 | 77.3 | 72.5 | 57.9 | 80.8 | +22.9 |
| Qwen3-30B | 2025 | 86.7 | 79.6 | 72.3 | 69.8 | 80.0 | +10.2 |
| Nemotron-32B | 2025 | 85.6 | 78.5 | 72.3 | 64.8 | 79.0 | +14.2 |
| Nemotron-7B | 2025 | 77.3 | 71.5 | 59.6 | 48.8 | 72.7 | +24.0 |

Problem-level bootstrap, passive minus compound d3 (positive = passive less harmful):

| model | dataset | difference | 95% interval | problems better/same/worse |
|---|---|---|---|---|
| Qwen3-30B | 2024 | +2.9 | [+0.0, +6.5] | 8/18/4 |
| Nemotron-32B | 2024 | +12.9 | [+6.9, +19.6] | 16/14/0 |
| Nemotron-7B | 2024 | +22.9 | [+14.8, +31.7] | 21/8/1 |
| Qwen3-30B | 2025 | +10.2 | [+5.0, +15.8] | 17/12/1 |
| Nemotron-32B | 2025 | +14.2 | [+6.7, +22.7] | 16/11/3 |
| Nemotron-7B | 2025 | +24.0 | [+15.8, +32.5] | 22/7/1 |

Five of six intervals exclude zero. **Qwen3-30B on AIME 2024 (+2.9, interval touching zero) should
not be claimed as significant.**

Passive context is not free, though. Against the single-problem baseline it still costs 1.0 to 6.7
points, significant in five of six cells (Nemotron-7B on AIME 2024, −1.0 [−4.4, +2.1], is the
exception). The honest statement is that length accounts for a small part of the compound
degradation and the model's own reasoning accounts for most of it — not that length is irrelevant.

**Run hygiene.** Context length matched to within 1–2 tokens in every cell (24,389 to 41,939
tokens depending on model and dataset). Passive text non-repeating throughout, repeat factor
exactly 1.0. Zero cutoffs in the passive arm, against 0.0–1.7% in compound d3. Grading is the same
function call as the compound runs use (`util.extract_and_grade(..., exp_name="compound")`), so
the two arms are scored identically.

### Caveats that belong in the text

**Target recency is not matched, and it favours the passive arm.** In compound the target
statement sits near the start of the prompt and the model reaches it tens of thousands of tokens
later, after its own reasoning. In passive the target is the last thing before generation. Both
arms have equal total context, but passive gives the model the easier geometry, so an unknown
share of the effect is position rather than content. This is the strongest available objection and
is better stated than waited for. Settling it would need a variant with the problem before the
passage, which was not run.

**Prefill versus self-generated context.** Compound's filler is produced autoregressively by the
model; passive's arrives as prompt tokens in prefill. These are different regimes, and the
distinction matters most for the 7B.

**No usable repetition comparison.** An earlier version of this control used a 411-word excerpt
repeated ~87–125x, and showed the opposite sign for Nemotron-7B (roughly 21–23 points *below*
compound). That was an artifact of the degenerate repetition loop. Those result files are no
longer in the repo, and the prompt has since changed as well, so the repeated arm differs from the
current one in two variables at once. **Do not make a claim about repetition without re-running
`--passive_mode repeat` under the current prompt.**

---

## 2. Decode-only recovery — partially usable, needs care

**Question.** When a model fails on a transformed problem, is it failing to read the problem or
failing to solve it? The model is given the transformed problem and the transformation rule, told
not to solve it, and asked to reconstruct the original statement between `<RECOVERED_PROBLEM>`
tags.

**Metric.** `recovered_rate` in the CSV is a binary per-sample gate: normalized exact match, or
character error rate ≤ 0.02, after normalizing away LaTeX presentation and whitespace. It is
**not** a semantic-similarity score. An earlier embedding-based approach (windowed cosine
similarity over the solve trace, `old/prompt_reconstruction/analyze_prompt_recovery.py`) was
abandoned in favour of collecting reconstructions directly. Graded alternatives are in the same
CSV: `ngram{1,2,4}_f1`, `cer_p25/p50/p75`, and `recovered_rate_spacing_sensitive` (the same gate
before whitespace normalization).

### The metric confounds decoding with compliance — read `stub_rate` alongside it

A large share of samples emit the tags and then put nothing usable between them: outputs like
`and` or `[the text]`, typically after thousands of reasoning tokens. `stub_rate` counts
reconstructions shorter than a quarter of the oracle. These are failures to attempt the task, not
failed reconstructions, but the gate scores them as decode failures.

| model | stub rate across its 24 cells |
|---|---|
| Qwen3-30B | 0.0% – 74.0% |
| Nemotron-32B | 3.3% – 46.0% |
| Nemotron-7B | 37.9% – 64.8% |

Qwen's high end is confined to the transforms it cannot decode at all (`rail_fence` 72.9/74.0,
`snake_vertical` 60.2/67.5); on the nine transforms where it produces real reconstructions it
stubs on 0.0–2.3% of samples. Nemotron-7B stubs on 38–65% of samples in *every* cell, including
ones it solves well. Conditioning on tag emission does not fix this — tag rates are 86–98% on the
semantic transforms, so the models do emit tags and then produce nothing.

### What is reportable

**Qwen3-30B is clean and carries the argument.** On the nine transforms where it attempts,
recovery exceeds solve accuracy on four (AIME 2024: `opposites`, `wrappers`, `sentence_reversal`,
`interleaved_context_line`) and five (AIME 2025: those four plus `split_reversal`). Where recovery
is at or above solve, the solve failures demonstrably are not reading failures — that is the
existence proof the experiment was run to produce.

Recovery falls *below* solve on the remaining transforms, including `word_reversal` (28.7 vs 47.5
on AIME 2024) and `interleaved_context_word` (9.2 vs 24.6). **The claim "recovery tracks or
exceeds solve accuracy" is false as a general statement and should not be made.** On those
transforms the decode number is a lower bound on comprehension, not a measure of it.

**The Nemotron numbers are internally inconsistent and should not stand alone.** Nemotron-7B
reconstructs `opposites` at 10.8% while solving it at 59.2%; a model cannot solve a problem it
cannot read, so the 10.8% is measuring something other than comprehension. Nemotron-32B is less
extreme but still 20–33 points under solve on the semantic transforms with a steady 20–37% stub
rate. If these are reported, print `stub_rate` beside them as an explicit compliance floor;
otherwise omit them and report decode recovery for Qwen alone.

### Prompt matching, for the methods section

The decode prompt and the solve prompt share the same wrapper (`TRANSFORMATION RULE:` /
`TRANSFORMED INPUT:`) and differ only in the instruction — reconstruct versus reconstruct and
solve. The transformation rule text was reconciled between the two scripts before these runs, and
`not_not` was dropped from the decode matrix on the grounds that it has nothing to recover, giving
12 transforms rather than 13. Sampling matches the perturbation runs (temperature 0.7, top_p 1.0,
16 samples, 32K context).

One measured negative result worth a sentence if space allows: the richer transformation-rule text
raised attempt rates on the hard visual transforms (Nemotron-32B's tag rate on `rail_fence` rose
from 45.4% to 59.8%, on `rectangle_perimeter` from 14.6% to 22.7%) while recovery on those
transforms stayed at 0.0%. Better instructions produced more attempts and no more successes.
