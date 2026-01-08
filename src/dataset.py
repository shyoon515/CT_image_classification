"""Dataset class for CT Image Classification"""

import json
from pathlib import Path
from typing import Tuple, Optional, Callable

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class CTDataset(Dataset):
    """CT Image Dataset for Normal vs Pneumonia Classification
    
    Args:
        image_dir: Directory containing images
        label_dir: Directory containing JSON label files
        transform: Optional transform to apply to images
        target_label: Target label to classify ('Normal' or 'pneumonia')
    """
    
    def __init__(
        self,
        image_dir: Path,
        label_dir: Path,
        transform: Optional[Callable] = None,
        target_label: str = 'pneumonia'
    ):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.transform = transform
        self.target_label = target_label
        
        # 모든 이미지 경로 수집
        self.image_paths = sorted(list(self.image_dir.rglob('*.png')))
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {self.image_dir}")
        
        # 라벨 로드
        self.labels = self._load_labels()
        
    def _load_labels(self) -> list:
        """Load labels from JSON files"""
        labels = []
        for img_path in self.image_paths:
            label_path = self.label_dir / f"{img_path.stem}.json"
            if not label_path.exists():
                raise FileNotFoundError(f"Label file not found: {label_path}")
            
            with open(label_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            
            # Binary classification: 1 if target_label, 0 otherwise
            label = 1 if meta['label'] == self.target_label else 0
            labels.append(label)
        
        return labels
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get image and label at index
        
        Returns:
            image: Tensor of shape (C, H, W)
            label: 0 (Normal) or 1 (target class)
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_distribution(self) -> dict:
        """Get distribution of classes in dataset"""
        unique, counts = np.unique(self.labels, return_counts=True)
        return {
            'Normal': int(counts[0]) if 0 in unique else 0,
            self.target_label: int(counts[1]) if 1 in unique else 0
        }
