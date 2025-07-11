"""
Data processing utilities for MecAgents
"""
from datasets import load_dataset, Dataset
from typing import List, Dict, Any, Optional

from .config import DataConfig


class DataProcessor:
    """Handles dataset loading and preprocessing for CAD code generation"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.datasets = None
        self.train_dataset = None
        self.test_dataset = None
    
    def load_dataset(self) -> tuple:
        """Load the CAD dataset"""
        print(f"Loading dataset: {self.config.dataset_name}")
        
        self.datasets = load_dataset(
            self.config.dataset_name,
            num_proc=self.config.num_proc,
            split=self.config.splits,
            cache_dir=self.config.cache_dir,
        )
        
        if len(self.datasets) >= 2:
            self.train_dataset, self.test_dataset = self.datasets
        else:
            self.train_dataset = self.datasets[0]
            self.test_dataset = None
        
        # Apply sampling if specified
        if self.config.sample_size and self.train_dataset:
            print(f"Sampling {self.config.sample_size} examples from training set")
            self.train_dataset = self.train_dataset.take(self.config.sample_size)
        
        return self.train_dataset, self.test_dataset
    
    def convert_to_conversation_format(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a single sample to conversation format for training"""
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": sample["prompt"]},
                    {"type": "image", "image": sample["image"]},
                ],
            },
            {
                "role": "assistant", 
                "content": [{"type": "text", "text": sample["cadquery"]}]
            },
        ]
        return {"messages": conversation}
    
    def convert_dataset_to_conversations(self, dataset: Optional[Dataset] = None) -> List[Dict[str, Any]]:
        """Convert entire dataset to conversation format"""
        if dataset is None:
            dataset = self.train_dataset
        
        if dataset is None:
            raise ValueError("No dataset available. Load dataset first.")
        
        print("Converting dataset to conversation format...")
        converted_dataset = [
            self.convert_to_conversation_format(sample) 
            for sample in dataset
        ]
        
        print(f"Converted {len(converted_dataset)} samples")
        return converted_dataset
    
    def create_inference_messages(self, image: Any, instruction: Optional[str] = None) -> List[Dict[str, Any]]:
        """Create messages format for inference"""
        if instruction is None:
            instruction = self.config.instruction
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"}, 
                    {"type": "text", "text": instruction}
                ],
            }
        ]
        return messages
    
    def get_sample_data(self, index: int = 0, from_test: bool = False) -> Dict[str, Any]:
        """Get a sample from the dataset"""
        dataset = self.test_dataset if from_test and self.test_dataset else self.train_dataset
        
        if dataset is None:
            raise ValueError("Dataset not loaded")
        
        if index >= len(dataset):
            raise ValueError(f"Index {index} out of range for dataset of size {len(dataset)}")
        
        return dataset[index]
    
    def print_sample_info(self, index: int = 0):
        """Print information about a sample"""
        sample = self.get_sample_data(index)
        
        print(f"Sample {index}:")
        print(f"Image: {type(sample['image'])}")
        print(f"Prompt: {sample['prompt'][:100]}...")
        print(f"CADQuery code: {sample['cadquery'][:200]}...")
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """Get statistics about the loaded datasets"""
        stats = {}
        
        if self.train_dataset:
            stats["train_size"] = len(self.train_dataset)
            stats["train_features"] = list(self.train_dataset.features.keys())
        
        if self.test_dataset:
            stats["test_size"] = len(self.test_dataset)
            stats["test_features"] = list(self.test_dataset.features.keys())
        
        return stats
