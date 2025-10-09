"""
Model loading utilities for InternVL and Qwen2.5
Adapted from SPARK project
"""
import torch
from transformers import AutoModel, AutoTokenizer
from typing import Literal, Optional


class ModelLoader:
    """Load and manage vision-language models"""

    def __init__(self, model_name: Literal["internvl3.5", "qwen2.5"], device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None

    def load_model(self, model_path: Optional[str] = None):
        """Load the specified model"""
        if self.model_name == "internvl3.5":
            self._load_internvl35(model_path)
        elif self.model_name == "qwen2.5":
            self._load_qwen25(model_path)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        return self.model, self.tokenizer

    def _load_internvl35(self, model_path: Optional[str] = None):
        """Load InternVL3.5 model"""
        # Default to InternVL3_5-8B if no path specified (note the underscore)
        path = model_path or "OpenGVLab/InternVL3_5-8B"

        print(f"Loading InternVL3.5 from {path}...")
        self.model = AutoModel.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).eval().to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            path,
            trust_remote_code=True,
            use_fast=False
        )
        print("InternVL3.5 loaded successfully!")

    def _load_qwen25(self, model_path: Optional[str] = None):
        """Load Qwen2.5-VL model"""
        # Default to Qwen2.5-VL-7B if no path specified
        path = model_path or "Qwen/Qwen2.5-VL-7B-Instruct"

        print(f"Loading Qwen2.5-VL from {path}...")
        self.model = AutoModel.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).eval().to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            path,
            trust_remote_code=True
        )
        print("Qwen2.5-VL loaded successfully!")

    def generate(self, **kwargs):
        """Wrapper for model generation"""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self.model.generate(**kwargs)
