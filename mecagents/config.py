"""
Configuration classes for MecAgents training framework
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ModelConfig:
    """Configuration for model setup"""

    model_name: str = "unsloth/gemma-3-4b-pt"
    load_in_4bit: bool = True
    use_gradient_checkpointing: str = "unsloth"

    # LoRA configuration: turn to true for improved performance
    finetune_vision_layers: bool = False
    finetune_language_layers: bool = False
    finetune_attention_modules: bool = False
    finetune_mlp_modules: bool = True

    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    bias: str = "none"
    random_state: int = 3407
    use_rslora: bool = False
    target_modules: str = "all-linear"
    modules_to_save: List[str] = None

    def __post_init__(self):
        if self.modules_to_save is None:
            self.modules_to_save = ["lm_head", "embed_tokens"]


@dataclass
class TrainingConfig:
    """Configuration for training setup"""

    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    gradient_checkpointing: bool = True
    max_grad_norm: float = 0.3
    warmup_ratio: float = 0.03
    max_steps: int = 1000
    num_train_epochs: Optional[int] = 5
    learning_rate: float = 2e-4
    logging_steps: int = 5
    save_strategy: str = "steps"
    save_steps: int = 50
    optim: str = "adamw_torch_fused"
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    seed: int = 3407
    output_dir: str = "outputs"
    report_to: str = "wandb"
    max_seq_length: int = 1024

    # Vision specific settings
    remove_unused_columns: bool = False
    dataset_text_field: str = ""
    skip_prepare_dataset: bool = True


@dataclass
class DataConfig:
    """Configuration for data processing"""

    dataset_name: str = "CADCODER/GenCAD-Code"
    cache_dir: str = "./Volumes/BIG-DATA/HUGGINGFACE_CACHE"
    num_proc: int = 16
    splits: List[str] = None
    sample_size: Optional[int] = 10_000
    instruction: str = (
        "Generate the CADQuery code needed to create the CAD for the provided image. Just the code, no other words."
    )

    def __post_init__(self):
        if self.splits is None:
            self.splits = ["train", "test"]


@dataclass
class InferenceConfig:
    """Configuration for inference"""

    max_new_tokens: int = 128
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 64
    use_cache: bool = True
    chat_template: str = "gemma-3"
