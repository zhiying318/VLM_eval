"""
Quick test script to verify the Controlled Images dataset loading
"""
import json
from pathlib import Path
from PIL import Image

def test_dataset_loading():
    """Test if we can load the dataset correctly"""
    print("Testing Controlled Images Dataset loading...")
    
    # Load dataset JSON directly
    dataset_path = Path("dataset/controlled_images_dataset.json")
    image_root = Path("dataset")
    
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    print(f"Dataset size: {len(data)}")
    
    # Test loading first few samples
    print("\nTesting first 3 samples:")
    for i in range(min(3, len(data))):
        item = data[i]
        # Fix path mismatch: JSON has "controlled_images" but actual folder is "controlled_image"
        image_path = image_root / item['image_path']
        
        print(f"\nSample {i}:")
        print(f"  Image path: {image_path}")
        print(f"  Image exists: {image_path.exists()}")
        
        # Try to load image
        try:
            if image_path.exists():
                image = Image.open(image_path).convert('RGB')
                print(f"  Image size: {image.size}")
            else:
                print(f"  Image NOT found!")
        except Exception as e:
            print(f"  Error loading image: {e}")
        
        print(f"  Correct caption: {item['caption_options'][0]}")
        print(f"  All options:")
        for j, option in enumerate(item['caption_options']):
            marker = "✓" if j == 0 else " "  # First option is always correct
            print(f"    [{marker}] {j}: {option}")

if __name__ == "__main__":
    test_dataset_loading()