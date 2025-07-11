"""
Training utilities for MecAgents
"""
import torch
from typing import Any, Dict, List
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

from .config import TrainingConfig


class TrainingManager:
    """Manages the training process for vision-language models"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.trainer = None
        self.training_stats = None
    
    def create_trainer(
        self, 
        model: Any, 
        processor: Any, 
        train_dataset: List[Dict[str, Any]]
    ) -> SFTTrainer:
        """Create and configure the SFT trainer"""
        print("Creating SFT trainer...")
        
        # Create training configuration
        training_args = SFTConfig(
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            gradient_checkpointing=self.config.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            max_grad_norm=self.config.max_grad_norm,
            warmup_ratio=self.config.warmup_ratio,
            max_steps=self.config.max_steps,
            num_train_epochs=self.config.num_train_epochs,
            learning_rate=self.config.learning_rate,
            logging_steps=self.config.logging_steps,
            save_strategy=self.config.save_strategy,
            save_steps=self.config.save_steps,
            optim=self.config.optim,
            weight_decay=self.config.weight_decay,
            lr_scheduler_type=self.config.lr_scheduler_type,
            seed=self.config.seed,
            output_dir=self.config.output_dir,
            report_to=self.config.report_to,
            
            # Vision specific settings
            remove_unused_columns=self.config.remove_unused_columns,
            dataset_text_field=self.config.dataset_text_field,
            dataset_kwargs={"skip_prepare_dataset": self.config.skip_prepare_dataset},
            max_seq_length=self.config.max_seq_length,
        )
        
        # Create trainer
        self.trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            processing_class=processor.tokenizer,
            data_collator=UnslothVisionDataCollator(model, processor),
            args=training_args,
        )
        
        print(f"Trainer created with {len(train_dataset)} training samples")
        return self.trainer
    
    def train(self) -> Dict[str, Any]:
        """Execute the training process"""
        if self.trainer is None:
            raise ValueError("Trainer must be created before training")
        
        print("Starting training...")
        
        # Get initial memory stats
        start_memory = self._get_memory_usage()
        
        # Train the model
        self.training_stats = self.trainer.train()
        
        # Get final memory stats
        end_memory = self._get_memory_usage()
        
        # Calculate training statistics
        training_summary = self._create_training_summary(start_memory, end_memory)
        
        print("Training completed!")
        self._print_training_summary(training_summary)
        
        return training_summary
    
    def _get_memory_usage(self) -> float:
        """Get current GPU memory usage in GB"""
        if not torch.cuda.is_available():
            return 0.0
        return round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    
    def _create_training_summary(self, start_memory: float, end_memory: float) -> Dict[str, Any]:
        """Create a summary of training statistics"""
        if not self.training_stats:
            return {}
        
        gpu_stats = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3) if gpu_stats else 0
        
        memory_used_for_training = round(end_memory - start_memory, 3)
        
        summary = {
            "training_time_seconds": self.training_stats.metrics.get('train_runtime', 0),
            "training_time_minutes": round(self.training_stats.metrics.get('train_runtime', 0) / 60, 2),
            "start_memory_gb": start_memory,
            "end_memory_gb": end_memory,
            "memory_used_for_training_gb": memory_used_for_training,
            "max_gpu_memory_gb": max_memory,
            "memory_percentage": round(end_memory / max_memory * 100, 3) if max_memory > 0 else 0,
            "training_memory_percentage": round(memory_used_for_training / max_memory * 100, 3) if max_memory > 0 else 0,
            "gpu_name": gpu_stats.name if gpu_stats else "Unknown",
        }
        
        # Add training metrics if available
        if hasattr(self.training_stats, 'metrics'):
            summary.update({
                "train_loss": self.training_stats.metrics.get('train_loss'),
                "train_samples_per_second": self.training_stats.metrics.get('train_samples_per_second'),
                "train_steps_per_second": self.training_stats.metrics.get('train_steps_per_second'),
            })
        
        return summary
    
    def _print_training_summary(self, summary: Dict[str, Any]):
        """Print a formatted training summary"""
        print("\n" + "="*50)
        print("TRAINING SUMMARY")
        print("="*50)
        
        print(f"GPU: {summary.get('gpu_name', 'Unknown')}")
        print(f"Training time: {summary.get('training_time_minutes', 0)} minutes")
        print(f"Peak memory usage: {summary.get('end_memory_gb', 0)} GB")
        print(f"Memory used for training: {summary.get('memory_used_for_training_gb', 0)} GB")
        print(f"Memory percentage: {summary.get('memory_percentage', 0)}%")
        
        if summary.get('train_loss'):
            print(f"Final train loss: {summary['train_loss']:.4f}")
        
        print("="*50)
    
    def save_trainer_state(self, output_path: str):
        """Save trainer state and logs"""
        if self.trainer is None:
            raise ValueError("Trainer must be created before saving state")
        
        # Save training logs if available
        if hasattr(self.trainer, 'state') and self.trainer.state.log_history:
            import json
            log_path = f"{output_path}/training_logs.json"
            with open(log_path, 'w') as f:
                json.dump(self.trainer.state.log_history, f, indent=2)
            print(f"Training logs saved to: {log_path}")
    
    def get_training_config_dict(self) -> Dict[str, Any]:
        """Get training configuration as dictionary"""
        return {
            "per_device_train_batch_size": self.config.per_device_train_batch_size,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "learning_rate": self.config.learning_rate,
            "max_steps": self.config.max_steps,
            "warmup_ratio": self.config.warmup_ratio,
            "weight_decay": self.config.weight_decay,
            "lr_scheduler_type": self.config.lr_scheduler_type,
            "optim": self.config.optim,
        }
