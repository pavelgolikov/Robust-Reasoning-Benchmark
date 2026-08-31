import re

RAW = "gibbon_vol1.txt"   # https://www.gutenberg.org/cache/epub/25717/pg25717.txt
OUT = "gibbon_decline_and_fall_long.txt"
TARGET_WORDS = 260000
SKIP_LEADING = 46          # Gutenberg + editor front matter; narrative begins "In the second century..."

text = open(RAW, encoding="utf-8").read()
start = text.index("\n", text.index("*** START OF THE PROJECT GUTENBERG")) + 1
end = text.index("*** END OF THE PROJECT GUTENBERG")
paras = re.split(r"\n\s*\n", text[start:end])

BAD_SUBSTRINGS = ("project gutenberg", "david reed", "david widger", "ebook", "html",
                  "pg #", "public domain", "copyright", "transcriber", "etext")

def clean_para(p):
    s = " ".join(p.split())
    if "(return)" in s or s.startswith("["):
        return None
    if re.match(r"^\d+\s*\(return\)", s):
        return None
    if re.match(r"^(Chapter|Part|Volume|Note|Footnote|Preface|Contents)\b", s, re.I):
        return None
    low = s.lower()
    if any(b in low for b in BAD_SUBSTRINGS):
        return None
    if s.isupper():
        return None
    non_ascii = sum(1 for c in s if ord(c) > 127)
    if non_ascii > 0.02 * max(1, len(s)):        # foreign-language quotations, verse
        return None
    letters = sum(c.isalpha() for c in s)
    if letters < 0.72 * len(s):                  # tables, citation soup
        return None
    # drop standalone footnote-reference numbers, keep prose punctuation
    toks = []
    for tok in s.split():
        core = tok.strip(".,;:!?()[]“”\"'—-")
        if core.isdigit():
            if tok and tok[-1] in ".,;:!?" and toks:
                toks[-1] += tok[-1]
            continue
        toks.append(tok)
    s = " ".join(toks)
    s = re.sub(r"\s+([.,;:!?])", r"\1", s)
    s = re.sub(r"\s+", " ", s).strip()
    if len(s.split()) < 30:
        return None
    return s

clean = [c for c in (clean_para(p) for p in paras) if c][SKIP_LEADING:]

out, n = [], 0
for s in clean:
    out.append(s)
    n += len(s.split())
    if n >= TARGET_WORDS:
        break

result = "\n\n".join(out) + "\n"
open(OUT, "w", encoding="utf-8").write(result)
print("paragraphs kept:", len(out), "of", len(clean), "available; words:", n, "chars:", len(result))
