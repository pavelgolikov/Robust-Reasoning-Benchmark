from datasets import load_dataset
from opposite_verbs_adj.transformation import apply_opposite_semantic_remapping

def main():
    print("Loading dataset...")
    dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")
    
    examples = []
    for i in range(2):
        prob = dataset[i]['problem']
        remapped = apply_opposite_semantic_remapping(prob, k=1, seed=42)
        examples.append((prob, remapped))

    with open("preview_opposites.md", "w") as f:
        f.write("# Preview of 'Opposites' Transformation\n\n")
        for i, (orig, remap) in enumerate(examples):
            f.write(f"## Example {i+1}\n\n")
            f.write("### Original\n")
            f.write(f"{orig}\n\n")
            f.write("### Remapped (Opposites)\n")
            f.write(f"{remap}\n\n")
            f.write("---\n\n")

    print("Preview generated in preview_opposites.md")

if __name__ == "__main__":
    main()
