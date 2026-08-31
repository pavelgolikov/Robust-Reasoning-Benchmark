Proposed plan:

1. Decode-only recovery experiment
Reviewer targets: Bnc8 and 5sft both ask whether failures are decoding failures vs math reasoning failures. Meta-review also flags this as a central concern.
This should be a new model run, not post-hoc CoT detection. Post-hoc recovery from CoT is too unreliable because semantic similarity is misleading after our transformations.
Experiment:
Give transformed prompt.
Instruct model: do not solve, only reconstruct the original problem.
Output between tags like <RECOVERED_PROBLEM>...</RECOVERED_PROBLEM>.
Compare recovered text to oracle-reversed original.
Score with normalized exact match, character error rate, ordered n-gram overlap, and transformation-specific residual checks, e.g. copied transformed input, still reversed, partially interleaved.
Runs needed:
Open-weight models only. No proprietary reruns, I ran out of API budget.
Prioritize Qwen3-30B-A3B, Nemotron-7B, Nemotron-32B. Add DSR1-70B/GPT-OSS-120B only if compute allows.
Ideally AIME 2024 + 2025, all 13 transforms. If time is tight, run a representative subset: semantic, syntactic, visual,
interleaved, plus hardest cases.

2. Equal-length passive-context control
Reviewer target: 5sft/meta-review question whether attention dilution is really attention pollution vs just longer context.
This is the most useful new attention-dilution experiment.
Experiment:
Same target problem as in the sequential multi-problem setup.
Prepend neutral/passive text with the same token length as the prior solved-problem traces.
No previous math problems, no previous answers, no generated CoT.
Ask model to solve the target.
Compare:
single-problem baseline
target after prior solved math/CoT
target after equal-length neutral text
Desired result:
If neutral text hurts much less than prior reasoning traces, then length alone does not explain the degradation.
This supports our claim that generated reasoning traces are more harmful than passive context.
Runs needed:
New model runs.
Same open-weight models as above if feasible.
We should not do “only solve final problem” or priority-instruction controls for now; they test task-following more than
our actual context-reset claim.

3. Problem-level uncertainty analysis
Reviewer target: 5sft and meta-review complain about small datasets and no statistical significance.
No model runs needed. Use existing results.
Analysis:
For each AIME problem, compute baseline accuracy.
For the same problem, compute perturbation accuracy.
Compute the drop per problem.
Summarize average drop and uncertainty across problems.
Important: treat the AIME problem, not each trajectory, as the unit of analysis.
In rebuttal we can call this “problem-level uncertainty analysis” and avoid jargon.

4. Cutoff clarification
Reviewer target: attention-dilution degradation might be due to max-token cutoffs.
No new runs needed.
Figure 1 already reports cutoff rates.
We should explicitly state that cutoff rates are low in the affected conditions and do not explain the accuracy decline.
If easy, also recompute accuracy excluding cutoff samples.