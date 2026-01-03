import sys
import os
import re
from datasets import load_dataset

# Add project root to sys.path to find 'utils' if needed, though here we just need local transformation
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

from transformation import apply_opposite_semantic_remapping

def generate_preview():
    # Load dataset
    dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")
    # Take first 30 (limit for evaluation)
    dataset = dataset.select(range(min(30, len(dataset))))

    output_path = os.path.join(os.path.dirname(__file__), "preview_remappings_k1_s42.md")
    
    lines = ["# Preview of Remappings: Opposites (100%)", "", "Configuration: k=1, seed=42", ""]

    for i, example in enumerate(dataset):
        problem = example['problem']
        
        # Apply transformation with k=1 (100%) and seed=42 (default for eval)
        # Note: We must re-seed for each example OR rely on the transformation's internal seeding if it's deterministic per call.
        # Checking transformation.py: it takes a seed. If we pass the SAME seed every time, it might produce identical random choices if the text length varies? 
        # Actually, evaluate.py passes the SAME seed to the function each time?
        # Let's check evaluation.py: 
        # problem_input = transformation_function(problem, k=k, seed=seed)
        # Yes, it passes the same seed. 
        # However, inside transformation.py:
        # if seed is not None: random.seed(seed)
        # This resets the RNG every time. This is Good for reproducibility per problem, 
        # but might be weird if we want variety across problems? 
        # Wait, if we reset seed every time, `random.sample` will pick the SAME sequence of indices relative to the list size?
        # Actually, `random.sample` depends on list content? No.
        # If I have 5 verbs, and seed is 42. It picks indices [0, 3].
        # If the next problem has 5 verbs, it will pick indices [0, 3] again.
        # For 100% (k=1), it picks ALL of them, so randomness only affects... well, nothing if it's 100%.
        # Except maybe if there's a choice of antonyms?
        # transformation.py line 40: unique_antonyms.sort(...) and then takes [0]. Deterministic.
        # So for 100%, it should be deterministic regardless of seed, mostly.
        # But let's stick to the calling convention of evaluate.py to be exact.
        
        transformed = apply_opposite_semantic_remapping(problem, k=1, seed=42)
        
        # Extract defyn block
        match = re.match(r"(defyn\{.*?\})\.\n\n", transformed, re.DOTALL)
        if match:
            def_block = match.group(1)
            # Parse contents
            inner = re.search(r"defyn\{(.*?)\}", def_block, re.DOTALL).group(1)
            mappings = inner.split(',')
            
            lines.append(f"## Problem {i+1}")
            lines.append("| Replacement | Original |")
            lines.append("| :--- | :--- |")
            for m in mappings:
                m = m.strip()
                # let "X" mean "Y"
                g = re.match(r'let\s+"(.*?)"\s+mean\s+"(.*?)"', m)
                if g:
                    lines.append(f"| {g.group(1)} | {g.group(2)} |")
            lines.append("")
        else:
            lines.append(f"## Problem {i+1}")
            lines.append("*No remappings generated.*")
            lines.append("")

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"Preview generated at {output_path}")

if __name__ == "__main__":
    generate_preview()
