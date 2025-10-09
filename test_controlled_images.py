"""
Controlled Images Dataset wrapper for testing VLM spatial reasoning
Based on the original SPARK evaluation framework
"""
import argparse
import json
import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from models import ModelLoader
from utils import load_image


class ControlledImagesDataset(Dataset):
    """
    Dataset for Controlled Images spatial reasoning evaluation
    
    Format:
    {
        "image_path": "data/controlled_images/beer-bottle_on_armchair.jpeg",
        "caption_options": [
            "A beer bottle on a armchair",     # Correct (index 0)
            "A beer bottle under a armchair",  # Incorrect
            "A beer bottle to the left of a armchair",  # Incorrect  
            "A beer bottle to the right of a armchair"  # Incorrect
        ]
    }
    """
    
    def __init__(self, dataset_path: str, image_root: str = "dataset"):
        self.dataset_path = Path(dataset_path)
        self.image_root = Path(image_root)
        
        # Load dataset
        with open(self.dataset_path, 'r') as f:
            self.data = json.load(f)
        
        print(f"Loaded {len(self.data)} samples from {self.dataset_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Construct full image path
        image_path = self.image_root / item['image_path']
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a dummy image if loading fails
            image = Image.new('RGB', (224, 224), color='white')
        
        return {
            'image': image,
            'image_path': str(image_path),
            'caption_options': item['caption_options'],
            'correct_caption': item['caption_options'][0],  # First option is always correct
            'correct_index': 0
        }


def evaluate_controlled_images_internvl(model, tokenizer, dataloader, device="cuda"):
    """
    Evaluate InternVL model on Controlled Images dataset
    
    For each image, we test all 4 caption options and see which one 
    the model thinks is most likely/accurate.
    """
    results = []
    
    print("Starting evaluation with InternVL...")
    
    for batch in tqdm(dataloader, desc="Evaluating spatial reasoning"):
        for item in batch:
            image = item['image']
            caption_options = item['caption_options']
            correct_index = item['correct_index']
            correct_caption = item['correct_caption']
            
            # Prepare image for InternVL
            try:
                pixel_values = load_image(image, max_num=12).to(torch.bfloat16).to(device)
                
                # Method 1: Ask model to choose the correct description
                question = f"""Look at this image and choose the most accurate description from the following options:

A) {caption_options[0]}
B) {caption_options[1]} 
C) {caption_options[2]}
D) {caption_options[3]}

Please respond with only the letter (A, B, C, or D) of the most accurate description."""

                generation_config = dict(max_new_tokens=10, do_sample=False)
                
                response = model.chat(
                    tokenizer,
                    pixel_values.unsqueeze(0),
                    question,
                    generation_config
                )
                
                # Parse response to get selected option
                response_clean = response.strip().upper()
                if 'A' in response_clean:
                    predicted_index = 0
                elif 'B' in response_clean:
                    predicted_index = 1
                elif 'C' in response_clean:
                    predicted_index = 2
                elif 'D' in response_clean:
                    predicted_index = 3
                else:
                    predicted_index = -1  # Invalid response
                
                # Check if prediction is correct
                is_correct = predicted_index == correct_index
                
                results.append({
                    'image_path': item['image_path'],
                    'caption_options': caption_options,
                    'correct_index': correct_index,
                    'correct_caption': correct_caption,
                    'predicted_index': predicted_index,
                    'predicted_caption': caption_options[predicted_index] if 0 <= predicted_index < 4 else "INVALID",
                    'raw_response': response,
                    'is_correct': is_correct,
                    'method': 'multiple_choice'
                })
                
            except Exception as e:
                print(f"Error processing {item['image_path']}: {e}")
                results.append({
                    'image_path': item['image_path'],
                    'caption_options': caption_options,
                    'correct_index': correct_index,
                    'correct_caption': correct_caption,
                    'predicted_index': -1,
                    'predicted_caption': "ERROR",
                    'raw_response': f"ERROR: {str(e)}",
                    'is_correct': False,
                    'method': 'multiple_choice'
                })
            
            # Memory cleanup
            if 'pixel_values' in locals():
                del pixel_values
            torch.cuda.empty_cache()
    
    return results


def calculate_spatial_reasoning_metrics(results: List[Dict]) -> Dict:
    """Calculate metrics for spatial reasoning evaluation"""
    if not results:
        return {}
    
    total = len(results)
    correct = sum(1 for r in results if r['is_correct'])
    errors = sum(1 for r in results if r['predicted_index'] == -1)
    
    # Analyze by spatial relationship type
    relationship_stats = {}
    for result in results:
        # Extract spatial relationship from correct caption
        correct_caption = result['correct_caption'].lower()
        if ' on ' in correct_caption:
            rel_type = 'on'
        elif ' under ' in correct_caption:
            rel_type = 'under'
        elif ' to the left of ' in correct_caption:
            rel_type = 'left'
        elif ' to the right of ' in correct_caption:
            rel_type = 'right'
        else:
            rel_type = 'other'
        
        if rel_type not in relationship_stats:
            relationship_stats[rel_type] = {'total': 0, 'correct': 0}
        
        relationship_stats[rel_type]['total'] += 1
        if result['is_correct']:
            relationship_stats[rel_type]['correct'] += 1
    
    # Calculate accuracy per relationship type
    for rel_type in relationship_stats:
        stats = relationship_stats[rel_type]
        stats['accuracy'] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
    
    return {
        'total_samples': total,
        'correct_predictions': correct,
        'errors': errors,
        'overall_accuracy': correct / (total - errors) if (total - errors) > 0 else 0,
        'error_rate': errors / total,
        'relationship_breakdown': relationship_stats
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate VLM on Controlled Images Dataset")
    
    # Dataset arguments
    parser.add_argument('--dataset_path', type=str, 
                       default='dataset/controlled_images_dataset.json',
                       help='Path to controlled images dataset JSON file')
    parser.add_argument('--image_root', type=str, default='dataset',
                       help='Root directory containing the dataset images')
    
    # Model arguments  
    parser.add_argument('--model', type=str, default='internvl3.5',
                       choices=['internvl3.5', 'qwen2.5'],
                       help='Model to evaluate')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Custom model path (uses default if not specified)')
    
    # Evaluation arguments
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for evaluation')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Directory to save results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for evaluation')
    
    args = parser.parse_args()
    
    # Setup
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    dataset = ControlledImagesDataset(
        dataset_path=args.dataset_path,
        image_root=args.image_root
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda x: x  # Return list of dicts
    )
    
    # Load model
    print(f"Loading model: {args.model}...")
    model_loader = ModelLoader(model_name=args.model, device=device)
    model, tokenizer = model_loader.load_model(args.model_path)
    
    # Evaluate
    if args.model == "internvl3.5":
        results = evaluate_controlled_images_internvl(model, tokenizer, dataloader, device)
    else:
        raise NotImplementedError(f"Evaluation for {args.model} not implemented yet")
    
    # Calculate metrics
    metrics = calculate_spatial_reasoning_metrics(results)
    
    # Print results
    print("\n" + "="*60)
    print("CONTROLLED IMAGES SPATIAL REASONING EVALUATION")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Total samples: {metrics['total_samples']}")
    print(f"Correct predictions: {metrics['correct_predictions']}")
    print(f"Errors: {metrics['errors']}")
    print(f"Overall accuracy: {metrics['overall_accuracy']:.2%}")
    print(f"Error rate: {metrics['error_rate']:.2%}")
    
    print("\nBreakdown by spatial relationship:")
    for rel_type, stats in metrics['relationship_breakdown'].items():
        print(f"  {rel_type.upper()}: {stats['correct']}/{stats['total']} = {stats['accuracy']:.2%}")
    
    print("="*60)
    
    # Save detailed results
    output_file = output_dir / f"controlled_images_{args.model}_results.json"
    final_results = {
        'metadata': {
            'model': args.model,
            'model_path': args.model_path,
            'dataset_path': args.dataset_path,
            'total_samples': len(results)
        },
        'metrics': metrics,
        'detailed_results': results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    print(f"\nDetailed results saved to: {output_file}")
    
    # Save summary CSV
    summary_csv = output_dir / f"controlled_images_{args.model}_summary.csv"
    summary_data = []
    for result in results:
        summary_data.append({
            'image_path': result['image_path'],
            'correct_caption': result['correct_caption'],
            'predicted_caption': result['predicted_caption'],
            'is_correct': result['is_correct'],
            'raw_response': result['raw_response']
        })
    
    df = pd.DataFrame(summary_data)
    df.to_csv(summary_csv, index=False)
    print(f"Summary CSV saved to: {summary_csv}")


if __name__ == "__main__":
    main()