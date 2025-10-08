"""
Helper script to create a sample dataset for testing
"""
from utils import create_sample_dataset
from pathlib import Path

if __name__ == "__main__":
    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Create sample dataset
    output_path = data_dir / "sample_dataset.json"
    create_sample_dataset(str(output_path), num_samples=10)

    print(f"\nSample dataset created!")
    print(f"Location: {output_path}")
    print("\nNext steps:")
    print("1. Add your images to the data/images/ directory")
    print("2. Update the sample_dataset.json with actual image paths")
    print("3. Run evaluation:")
    print("   python evaluate.py --model internvl3.5 --data_path data/sample_dataset.json --image_root data/images/")
