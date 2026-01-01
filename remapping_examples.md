# Symbol Remapping Transformation Examples

This file contains 3 examples of the symbol remapping transformation using **Strict Cross-Type Mapping** and **Dynamic Frequency Analysis** on IDs 11, 26, 28:
*   **Symbols** -> **Words**.
*   **Words** -> **Symbols**.
*   **Dynamic**: Most frequent words in the problem text are targeted.
*   **Case Sensitive Definitions**: Proper nouns like "Bob" are defined with their original casing.

## Example ID: 11

### Original Problem
```latex
Consider the paths of length $16$ that follow the lines from the lower left corner to the upper right corner on an $8\times 8$ grid. Find the number of such paths that change direction exactly four times, as in the examples shown below.
```

### Transformed Problem
```latex
defyn{let ")" mean "paths", let "<" mean "corner", let "(" mean "Consider"}.

( the ) of length $16$ that follow the lines from the lower left < to the upper right < on an $8\times 8$ grid. Find the number of such ) that change direction exactly four times, as in the examples shown below.
```

*(Words "paths", "corner", "Consider" mapped to Symbols)*

---

## Example ID: 26

### Original Problem
```latex
Alice chooses a set $A$ of positive integers. Then Bob lists all finite nonempty sets $B$ of positive integers with the property that the maximum element of $B$ belongs to $A$. Bob's list has 2024 sets. Find the sum of the elements of A.
```

### Transformed Problem
```latex
defyn{let "~" mean "positive", let "]" mean "integers", let "}" mean "Bob"}.

Alice chooses a set $A$ of ~ ]. Then } lists all finite nonempty sets $B$ of ~ ] with the property that the maximum element of $B$ belongs to $A$. }'s list has 2024 sets. Find the sum of the elements of A.
```

*(Words "positive", "integers", "Bob" mapped to Symbols)*

---

## Example ID: 28

### Original Problem
```latex
Torus $T$ is the surface produced by revolving a circle with radius $3$ around an axis in the plane of the circle that is a distance $6$ from the center of the circle (so like a donut). Let $S$ be a sphere with a radius $11$. When $T$ rests on the outside of $S$, it is externally tangent to $S$ along a circle with radius $r_i$, and when $T$ rests on the outside of $S$, it is externally tangent to $S$ along a circle with radius $r_o$. The difference $r_i-r_o$ can be written as $\tfrac{m}{n}$, where $m$ and $n$ are relatively prime positive integers. Find $m+n$.
[asy] unitsize(0.3 inch); draw(ellipse((0,0), 3, 1.75)); draw((-1.2,0.1)..(-0.8,-0.03)..(-0.4,-0.11)..(0,-0.15)..(0.4,-0.11)..(0.8,-0.03)..(1.2,0.1)); draw((-1,0.04)..(-0.5,0.12)..(0,0.16)..(0.5,0.12)..(1,0.04)); draw((0,2.4)--(0,-0.15)); draw((0,-0.15)--(0,-1.75), dashed); draw((0,-1.75)--(0,-2.25)); draw(ellipse((2,0), 1, 0.9)); draw((2.03,-0.02)--(2.9,-0.4)); [/asy]
```

### Transformed Problem
```latex
defyn{let "multiply" mean "(", let "evaluate" mean ")", let "quadruple" mean "-", let "∀" mean "circle", let "=" mean "radius", let "&" mean "rests"}.

Torus $T$ is the surface produced by revolving a ∀ with = $3$ around an axis in the plane of the ∀ that is a distance $6$ from the center of the ∀ multiplyso like a donutevaluate. Let $S$ be a sphere with a = $11$. When $T$ & on the outside of $S$, it is externally tangent to $S$ along a ∀ with = $r_i$, and when $T$ & on the outside of $S$, it is externally tangent to $S$ along a ∀ with = $r_o$. The difference $r_iquadrupler_o$ can be written as $\tfrac{m}{n}$, where $m$ and $n$ are relatively prime positive integers. Find $m+n$.
[asy] unitsizemultiply0.3 inchevaluate; drawmultiplyellipsemultiplymultiply0,0evaluate, 3, 1.75evaluateevaluate; drawmultiplymultiplyquadruple1.2,0.1evaluate..multiplyquadruple0.8,quadruple0.03evaluate..multiplyquadruple0.4,quadruple0.11evaluate..multiply0,quadruple0.15evaluate..multiply0.4,quadruple0.11evaluate..multiply0.8,quadruple0.03evaluate..multiply1.2,0.1evaluateevaluate; drawmultiplymultiplyquadruple1,0.04evaluate..multiplyquadruple0.5,0.12evaluate..multiply0,0.16evaluate..multiply0.5,0.12evaluate..multiply1,0.04evaluateevaluate; drawmultiplymultiply0,2.4evaluatequadruplequadruplemultiply0,quadruple0.15evaluateevaluate; drawmultiplymultiply0,quadruple0.15evaluatequadruplequadruplemultiply0,quadruple1.75evaluate, dashedevaluate; drawmultiplymultiply0,quadruple1.75evaluatequadruplequadruplemultiply0,quadruple2.25evaluateevaluate; drawmultiplyellipsemultiplymultiply2,0evaluate, 1, 0.9evaluateevaluate; drawmultiplymultiply2.03,quadruple0.02evaluatequadruplequadruplemultiply2.9,quadruple0.4evaluateevaluate; [/asy]
```

*(Symbols `(`, `)`, `-` mapped to Words; Words `circle`, `radius`, `rests` mapped to Symbols)*
