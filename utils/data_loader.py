"""
Simple data loader for VQA tasks
"""
import json
from pathlib import Path
from PIL import Image
from typing import List, Dict, Optional
from torch.utils.data import Dataset


class SimpleVQADataset(Dataset):
    """
    Simple VQA dataset loader
    Supports both JSON and CSV formats

    Expected format:
    - image_path: path to image
    - question: question text
    - answer: ground truth answer (optional for inference)
    """

    def __init__(self, data_path: str, image_root: Optional[str] = None):
        """
        Args:
            data_path: path to JSON/CSV file with questions
            image_root: root directory for images (if paths are relative)
        """
        self.data_path = Path(data_path)
        self.image_root = Path(image_root) if image_root else None
        self.data = self._load_data()

    def _load_data(self) -> List[Dict]:
        """Load data from file"""
        if self.data_path.suffix == '.json':
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        elif self.data_path.suffix == '.csv':
            import pandas as pd
            df = pd.read_csv(self.data_path)
            data = df.to_dict('records')
        else:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Load image
        image_path = item['image_path']
        if self.image_root:
            image_path = self.image_root / image_path
        else:
            image_path = Path(image_path)

        image = Image.open(image_path).convert('RGB')

        return {
            'image': image,
            'question': item['question'],
            'answer': item.get('answer', None),  # May be None for inference
            'image_path': str(image_path),
            'metadata': {k: v for k, v in item.items() if k not in ['image_path', 'question', 'answer']}
        }


def create_sample_dataset(output_path: str, num_samples: int = 10):
    """
    Create a sample dataset file for testing

    Args:
        output_path: where to save the sample dataset
        num_samples: number of sample questions to generate
    """
    sample_data = []

    for i in range(num_samples):
        sample_data.append({
            "image_path": f"images/sample_{i}.jpg",
            "question": f"What is in this image? (sample {i})",
            "answer": f"Sample answer {i}"
        })

    output_path = Path(output_path)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)

    print(f"Sample dataset created at {output_path}")
