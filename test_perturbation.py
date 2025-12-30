from perturbation import apply_not_not_perturbation

def test_perturbation():
    # Define examples (Small, Medium, Large)
    
    # Small Example
    small_text = "Find the sum of three distinct positive integers."
    # Adjectives: distinct (pos), positive (pos). Numbers: three (pos).
    # Eligible: three, distinct, positive
    # First ADJ: distinct.
    
    # Medium Example (approx 2 sentences)
    medium_text = "Let a, b, and c be real numbers such that a + b + c = 0. Determine the maximum value of the product abc."
    
    # Large Example (AIME style problem)
    large_text = "Let S be the set of all positive integers n such that n < 1000 and the number of divisors of n is even. Find the number of elements in S. Assume that k is a positive integer greater than 1."
    
    examples = [
        ("Small", small_text),
        ("Medium", medium_text),
        ("Large", large_text)
    ]
    
    k = 3
    print(f"--- Testing Not Not Perturbation (k={k}) ---\n")
    
    for name, text in examples:
        print(f"### {name} Example ###")
        print(f"Original:\n{text}")
        perturbed = apply_not_not_perturbation(text, k=k)
        print(f"Perturbed:\n{perturbed}")
        print("-" * 40 + "\n")

if __name__ == "__main__":
    test_perturbation()
