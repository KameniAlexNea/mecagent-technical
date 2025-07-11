"""
Model management utilities for MecAgents
"""
import torch
from typing import Tuple, Any
from unsloth import FastVisionModel, get_chat_template

from .config import ModelConfig


class ModelManager:
    """Manages model loading, configuration, and LoRA setup"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.processor = None
    
    def load_model(self) -> Tuple[Any, Any]:
        """Load the base vision model and processor"""
        print(f"Loading model: {self.config.model_name}")
        
        model, processor = FastVisionModel.from_pretrained(
            self.config.model_name,
            load_in_4bit=self.config.load_in_4bit,
            use_gradient_checkpointing=self.config.use_gradient_checkpointing,
        )
        
        self.model = model
        self.processor = processor
        return model, processor
    
    def setup_lora(self) -> Any:
        """Configure LoRA adapters for parameter efficient fine-tuning"""
        if self.model is None:
            raise ValueError("Model must be loaded before setting up LoRA")
        
        print("Setting up LoRA adapters...")
        
        self.model = FastVisionModel.get_peft_model(
            self.model,
            finetune_vision_layers=self.config.finetune_vision_layers,
            finetune_language_layers=self.config.finetune_language_layers,
            finetune_attention_modules=self.config.finetune_attention_modules,
            finetune_mlp_modules=self.config.finetune_mlp_modules,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias=self.config.bias,
            random_state=self.config.random_state,
            use_rslora=self.config.use_rslora,
            loftq_config=None,
            target_modules=self.config.target_modules,
            modules_to_save=self.config.modules_to_save,
        )
        
        return self.model
    
    def setup_chat_template(self, template_name: str = "gemma-3") -> Any:
        """Setup chat template for the processor"""
        if self.processor is None:
            raise ValueError("Processor must be loaded before setting up chat template")
        
        print(f"Setting up chat template: {template_name}")
        self.processor = get_chat_template(self.processor, template_name)
        return self.processor
    
    def prepare_for_training(self):
        """Prepare model for training mode"""
        if self.model is None:
            raise ValueError("Model must be loaded before preparing for training")
        
        FastVisionModel.for_training(self.model)
        print("Model prepared for training")
    
    def prepare_for_inference(self):
        """Prepare model for inference mode"""
        if self.model is None:
            raise ValueError("Model must be loaded before preparing for inference")
        
        FastVisionModel.for_inference(self.model)
        print("Model prepared for inference")
    
    def save_model(self, output_path: str, save_processor: bool = True):
        """Save the fine-tuned model"""
        if self.model is None:
            raise ValueError("Model must be loaded before saving")
        
        print(f"Saving model to: {output_path}")
        self.model.save_pretrained(output_path)
        
        if save_processor and self.processor is not None:
            self.processor.save_pretrained(output_path)
            print(f"Processor saved to: {output_path}")
    
    def save_merged_model(self, output_path: str):
        """Save model merged to float16"""
        if self.model is None or self.processor is None:
            raise ValueError("Model and processor must be loaded before saving merged model")
        
        print(f"Saving merged model to: {output_path}")
        self.model.save_pretrained_merged(output_path, self.processor)
    
    def get_memory_stats(self) -> dict:
        """Get GPU memory statistics"""
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}
        
        gpu_stats = torch.cuda.get_device_properties(0)
        current_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        
        return {
            "gpu_name": gpu_stats.name,
            "max_memory_gb": max_memory,
            "current_memory_gb": current_memory,
            "memory_percentage": round(current_memory / max_memory * 100, 3)
        }
