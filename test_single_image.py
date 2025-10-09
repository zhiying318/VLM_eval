"""
Simple single image test for mLLM spatial reasoning
Test one image at a time to understand how the model works
"""
import json
from pathlib import Path
from PIL import Image
import torch

# Import our model loader (will need dependencies installed)
try:
    from models import ModelLoader
    from utils import load_image
    MODELS_AVAILABLE = True
except ImportError as e:
    print(f"Model imports not available: {e}")
    print("You can still test dataset loading without running the model")
    MODELS_AVAILABLE = False


def load_single_sample(dataset_path: str, image_root: str, sample_index: int = 0):
    """Load a single sample from the dataset"""
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    if sample_index >= len(data):
        raise ValueError(f"Sample index {sample_index} out of range. Dataset has {len(data)} samples.")
    
    item = data[sample_index]
    image_path = Path(image_root) / item['image_path']
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    return {
        'image': image,
        'image_path': str(image_path),
        'caption_options': item['caption_options'],
        'correct_caption': item['caption_options'][0],  # First option is always correct
        'correct_index': 0,
        'sample_index': sample_index
    }


def test_single_image_with_model(sample, model_name="qwen2.5", device="cuda"):   # change model name manually "internvl3.5"
    """Test with actual model - simplified version"""
    if not MODELS_AVAILABLE:
        print("❌ Model dependencies not available.")
        return None
    
    print(f"Loading {model_name}...")
    model_loader = ModelLoader(model_name=model_name, device=device)
    model, tokenizer = model_loader.load_model()
    
    # Prepare the question
    options = sample['caption_options']
    question = f"""Look at this image and choose the most accurate description:

A) {options[3]}
B) {options[1]} 
C) {options[2]}
D) {options[0]}

Answer with only the letter (A, B, C, or D)."""

    try:
        # Fix: Prepare image for mLLM - remove the extra unsqueeze
        pixel_values = load_image(sample['image'], max_num=12).to(torch.bfloat16).to(device)
        
        # Generate response
        generation_config = dict(max_new_tokens=10, do_sample=False)
        response = model.chat(
            tokenizer,
            pixel_values,  # Remove .unsqueeze(0) - load_image already handles batching
            question,
            generation_config
        )
        
        print(f"Model response: {response}")
        
        # Memory cleanup
        del pixel_values
        torch.cuda.empty_cache()
        
        return response
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def main():
    # Configuration
    dataset_path = "dataset/controlled_images_dataset.json"
    image_root = "dataset"
    
    # Ask user which sample to test
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    print(f"Dataset has {len(data)} samples")
    
    try:
        sample_index = int(input(f"Which sample to test? (0-{len(data)-1}): ") or "0")
    except ValueError:
        sample_index = 0
    
    # Load the sample
    sample = load_single_sample(dataset_path, image_root, sample_index)
    print(f"Testing: {sample['image_path']}")
    print(f"Correct: {sample['correct_caption']}")
    
    # Test with model
    if MODELS_AVAILABLE:
        # Check available GPUs and choose the least used one
        if torch.cuda.is_available():
            # Check GPU 1 first (often less used)
            torch.cuda.set_device(1)  # Use GPU 1
            device = "cuda:1"
        else:
            device = "cpu"
        
        print(f"Using device: {device}")
        result = test_single_image_with_model(sample, device=device)
    else:
        print("Model testing not available. Install dependencies first.")


if __name__ == "__main__":
    main()