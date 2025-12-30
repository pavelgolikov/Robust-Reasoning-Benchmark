# Linguistic Traps Evaluation Report

## Summary
- **Baseline Accuracy**: 3.33%
- **Adversarial Accuracy (Not Not, k=3)**: 0.00%
- **Performance Gap**: 3.33%

## Detailed Flipping Analysis
| ID | Baseline | Adversarial | Flip Type | Original Problem | Perturbed Problem |
|---|---|---|---|---|---|
| 72 | CORRECT | INCORRECT | SUCCESS (Trap Worked) | Find the largest possible real part of \[(75+117i)z+\frac{96+144i}{z}\]where $z$ is a complex number... | Find the not not largest possible not not real part of \[(75+117i)z+\frac{96+not not 144i}{z}\]where... |
