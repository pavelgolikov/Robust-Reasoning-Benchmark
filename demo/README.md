# Transformation Reversibility Demo

This standalone demo applies all linguistic transformations from the Robust Reasoning Benchmark
to the AIME 2024 and AIME 2025 datasets, and verifies that each transformation is reversible.

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Run

```bash
python demo.py
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--num_print_samples` | 30 | Number of matching samples to print per transformation |
| `--seed` | 42 | Random seed |
| `--output_dir` | `demo_output` | Directory to write reports into |

## Output

Reports are written to `demo_output/` with one file per dataset:
- `HuggingFaceH4_aime_2024_reversibility_report.txt`
- `MathArena_aime_2025_reversibility_report.txt`

Each report shows:
- **MATCH EXAMPLE** entries with the original, transformed, and reversed text
- **MISMATCH** entries (if any) with detailed comparison

## Transformations descriptions given to the model

TECHNIQUE_DESCRIPTIONS = {
    'baseline': "No transformation applied. Solve the problem as it is presented in TRANSFORMED INPUT.",
    'word_reversal': "The order of words (words are defined as sequences of symbols separated by spaces) in the user query has been reversed.",
    'sentence_reversal': "The order of sentences in the user query has been reversed. Sentences are defined as sequences of symbols separated by periods.",
    'interleaved_context_word': "User query will consist of two problems - A and B, whose statements are interleaved word by word. First word belongs to problem A, second word belongs to problem B, third word belongs to problem A, and so on. You need to solve only problem A. Words are defined as sequences of symbols separated by spaces. If one problem statement is shorter than the other, the shorter problem statement will be repeated from the beginning to fill the remaining space.",
    'interleaved_context_symbol': "User query will consist of two problems - A and B, whose statements are interleaved symbol by symbol (including punctuation and spaces). First symbol belongs to problem A, second symbol belongs to problem B, third symbol belongs to problem A, and so on. You need to solve only problem A. If one problem statement is shorter than the other, the shorter problem statement will be repeated from the beginning to fill the remaining space.",
    'interleaved_context_line': "User query will consist of two problems - A and B, whose statements are split into line segments at most 60 symbols long. Each segment is placed on a separate line and is prefixed by a problem tag (e.g. problem A or B) and a space. The segments for different problems are interleaved. You need to solve only problem A. If one problem statement is shorter than the other, the shorter problem statement will be repeated from the beginning to fill the remaining space.",
    'split_reversal':  "Every word (words are defined as sequences of symbols separated by spaces) in user query has its symbols in reverse order.",
    'opposites': "There will be terms remapped in the user query. The remappings are defined inside 'defyn{}' block in the middle of user query.",
    'wrappers':  "There will be terms remapped in the user query. The remappings are defined inside 'defyn{}' block in the middle of user query.",
    'rail_fence': "The user query is encoded using the Rail Fence Cipher. The input is provided as a visual grid where the symbols (including spaces) of the encoded message string (message string does NOT contain any newline characters) are placed in a zigzag pattern across multiple rails (rows), and empty spaces are filled with dots (.). To decode, read the characters in zigzag order: Down-and-Right diagonally until you hit bottom rail, then Up-and-Right diagonally until you hit top rail, then Down-and-Right again etc... Rows are given on separate lines and all have equal lengths.",
    'rectangle_perimeter': "The user query is mapped onto the perimeter of a rectangle. The message is written as a single continuous string following the edges of the shape in a clockwise manner, beginning at the top-left. The TRANSFORMED INPUT is provided as a visual text block representing this rectangle with GRID START and GRID END markers. The center of the shape is filled with dots.",
    'snake_vertical': "The user query is written into a grid using a vertical 'snake' (zigzag) pattern. Starting from the top-left, the text is written down the first column, then up the second column, then down the third, and so on. The TRANSFORMED INPUT is provided as a visual grid with GRID START and GRID END markers.",
    'snake_horizontal': "The user query is written into a grid using a horizontal 'snake' (zigzag) pattern. Starting from the top-left, the text is written across the first row, then left across the second row, then right across the third, and so on. The TRANSFORMED INPUT is provided as a visual grid with GRID START and GRID END markers.",
    'compound': "The user query contains multiple unrelated math problems.",
}
