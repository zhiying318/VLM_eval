"""
Main evaluation script for InternVL3.5 and Qwen2.5 models
"""
import gc
import json
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

from models import ModelLoader
from utils import SimpleVQADataset, load_image


def evaluate_internvl35(model, tokenizer, dataloader, device="cuda"):
    """Evaluate InternVL3.5 model"""
    results = []

    for batch in tqdm(dataloader, desc="Evaluating InternVL3.5"):
        # Process batch
        for item in batch:
            question = item['question']

            # Prepare image - InternVL3.5 uses dynamic preprocessing
            pixel_values = load_image(item['image'], max_num=12).to(torch.bfloat16).to(device)

            # Generate response
            generation_config = dict(max_new_tokens=512, do_sample=False)

            try:
                response = model.chat(
                    tokenizer,
                    pixel_values.unsqueeze(0),
                    question,
                    generation_config
                )

                results.append({
                    'image_path': item['image_path'],
                    'question': question,
                    'ground_truth': item['answer'],
                    'prediction': response,
                    'metadata': item['metadata']
                })
            except Exception as e:
                print(f"Error processing {item['image_path']}: {e}")
                results.append({
                    'image_path': item['image_path'],
                    'question': question,
                    'ground_truth': item['answer'],
                    'prediction': f"ERROR: {str(e)}",
                    'metadata': item['metadata']
                })

            # Memory cleanup
            del pixel_values
            torch.cuda.empty_cache()
            gc.collect()

    return results


def evaluate_qwen25(model, tokenizer, dataloader, device="cuda"):
    """Evaluate Qwen2.5-VL model"""
    results = []

    for batch in tqdm(dataloader, desc="Evaluating Qwen2.5-VL"):
        for item in batch:
            question = item['question']

            try:
                # Prepare input for Qwen2.5-VL
                query = tokenizer.from_list_format([
                    {'image': item['image_path']},
                    {'text': question},
                ])

                response, _ = model.chat(tokenizer, query=query, history=None)

                results.append({
                    'image_path': item['image_path'],
                    'question': question,
                    'ground_truth': item['answer'],
                    'prediction': response,
                    'metadata': item['metadata']
                })
            except Exception as e:
                print(f"Error processing {item['image_path']}: {e}")
                results.append({
                    'image_path': item['image_path'],
                    'question': question,
                    'ground_truth': item['answer'],
                    'prediction': f"ERROR: {str(e)}",
                    'metadata': item['metadata']
                })

            # Memory cleanup
            torch.cuda.empty_cache()
            gc.collect()

    return results


def calculate_metrics(results):
    """Calculate simple accuracy metrics"""
    if not results:
        return {}

    total = len(results)
    correct = 0
    errors = 0

    for result in results:
        if result['prediction'].startswith('ERROR'):
            errors += 1
        elif result['ground_truth'] and result['ground_truth'].lower() in result['prediction'].lower():
            correct += 1

    return {
        'total': total,
        'correct': correct,
        'errors': errors,
        'accuracy': correct / (total - errors) if (total - errors) > 0 else 0,
        'error_rate': errors / total
    }


def main(args):
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create results directory
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print(f"Loading dataset from {args.data_path}...")
    dataset = SimpleVQADataset(
        data_path=args.data_path,
        image_root=args.image_root
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda x: x  # Return list of dicts
    )
    print(f"Loaded {len(dataset)} samples")

    # Load model
    print(f"Loading model: {args.model}...")
    model_loader = ModelLoader(model_name=args.model, device=device)
    model, tokenizer = model_loader.load_model(args.model_path)

    # Evaluate
    print("Starting evaluation...")
    if args.model == "internvl3.5":
        results = evaluate_internvl35(model, tokenizer, dataloader, device)
    elif args.model == "qwen2.5":
        results = evaluate_qwen25(model, tokenizer, dataloader, device)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    # Calculate metrics
    metrics = calculate_metrics(results)
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Total samples: {metrics['total']}")
    print(f"Correct predictions: {metrics['correct']}")
    print(f"Errors: {metrics['errors']}")
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"Error rate: {metrics['error_rate']:.2%}")
    print("="*50)

    # Save results
    output_file = results_dir / f"{args.model}_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'metrics': metrics,
            'results': results
        }, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate VLM models on VQA datasets")

    # Data arguments
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to dataset JSON/CSV file')
    parser.add_argument('--image_root', type=str, default=None,
                        help='Root directory for images (if paths are relative)')

    # Model arguments
    parser.add_argument('--model', type=str, required=True,
                        choices=['internvl3.5', 'qwen2.5'],
                        help='Model to evaluate')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Custom model path (uses default if not specified)')

    # Evaluation arguments
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for evaluation')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Directory to save results')

    args = parser.parse_args()
    main(args)
