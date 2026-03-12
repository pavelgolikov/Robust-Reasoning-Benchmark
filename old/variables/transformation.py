def apply_variables(problem_text):
    extraction_prompt_template = """Task:
        Extract the following three categories of load-bearing terms from the input problem and return them in JSON format.

        1. "variables":
        - List all single-letter mathematical variables (e.g., n, x, k, S).
        - Ignore units (cm, deg) or standard constants (pi) unless they are variables.

        2. "entities":
        - List proper nouns (e.g., Alice, Bob) or specific object names (e.g., Stack, Pile, Triangle) that act as the subjects of the problem.

        3. "operators":
        - List the Standard Mathematical Symbol (+, -, *, /) corresponding to operations found in the text.
        - If the text says "removes", "decreases" or "difference", you output "-".
        - If the text says "total", "sum", or "adds", you output "+".
        - If the text says "product", "area", or "times", you output "*".
        - If the text says "per", "out of", "divisible" or "ratio", you output "/".
        - Similarly for other common mathematical operations described in words.
        - Only include operators that are actually used or implied in the logic.
        
        4. "adjectives":
        - List important mathematical adjectives that modify quantities, such as "consecutive", "even", "odd", "prime", "positive", "negative", "integer", "real", etc.

        Only output the following JSON structure exactly as shown:
        {{
        "variables": ["list", "of", "strings"],
        "entities": ["list", "of", "strings"],
        "operators": ["list", "of", "symbols"]
        "adjectives": ["list", "of", "strings"]
        }}

        Input Problem:
        {problem_text}
    """
    return extraction_prompt_template.format(problem_text=problem_text)
