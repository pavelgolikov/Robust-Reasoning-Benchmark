import spacy
nlp = spacy.load("en_core_web_sm")

text = r"""\node[above]  at (A) {$A$};
    \node[below]       at (B) {$B$};
    \node[below]  at (C) {$C$};
    \node[left]       at (D) {$D$};
    \node[above left]       at (E) {$E$};
    \node[below]        at (F) {$F$};
    \node[below left]        at (G) {$G$};
    \node[right]        at (M) {$M$};
    \node[left]       at (N) {$N$};"""

doc = nlp(text)
print(f"{'Token':<15} {'POS':<6} {'Tag':<6} {'Is Alpha'}")
print("-" * 40)
for token in doc:
    if token.text in ['right', 'left', 'above', 'below', 'node']:
        print(f"{token.text:<15} {token.pos_:<6} {token.tag_:<6} {token.is_alpha}")
