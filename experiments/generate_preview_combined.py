from datasets import load_dataset
try:
    from opposites_not.transformation import apply_opposites_not_yot_transformation
except ImportError:
    import sys
    sys.path.append('.')
    from opposites_not.transformation import apply_opposites_not_yot_transformation

def main():
    print("Loading dataset...")
    dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")
    
    examples = []
    print("Generating examples...")
    for i in range(2):
        prob = dataset[i]['problem']
        # k_opp=1 (100% opposites), k_not=3
        remapped = apply_opposites_not_yot_transformation(prob, k_opp=1, k_not=3, seed=42)
        examples.append((prob, remapped))

    with open("preview_opposites_not.md", "w") as f:
        f.write("# Preview of 'Opposites + Not Not + Yot' Transformation\n\n")
        
        for i, (orig, remap) in enumerate(examples):
            f.write(f"## Example {i+1}\n\n")
            f.write("### Original\n")
            f.write(f"{orig}\n\n")
            f.write("### Transformed\n")
            f.write(f"{remap}\n\n")
            f.write("---\n\n")

    print("Preview generated in preview_opposites_not.md")

if __name__ == "__main__":
    main()
