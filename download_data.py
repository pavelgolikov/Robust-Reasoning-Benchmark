from datasets import load_dataset

def main():
    print("Downloading AIME 2024 dataset...")
    try:
        dataset = load_dataset("HuggingFaceH4/aime_2024")
        print("Dataset downloaded successfully.")
        print(f"Dataset structure: {dataset}")
        
        # Check splits
        for split in dataset.keys():
            print(f"Split: {split}, Size: {len(dataset[split])}")
            print(f"Sample from {split}:")
            print(dataset[split][0])
            
    except Exception as e:
        print(f"Error downloading dataset: {e}")

if __name__ == "__main__":
    main()
