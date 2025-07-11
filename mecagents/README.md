# MecAgents - Modular CAD Code Generation Framework

MecAgents is a modular framework for training and using vision-language models for CAD code generation. It provides a clean, structured approach to working with models that can generate CADQuery code from mechanical part images.

## Architecture

The framework is organized into several key modules:

### Core Modules

- **`config.py`** - Configuration classes for all components
- **`model.py`** - Model management and LoRA setup
- **`data.py`** - Dataset loading and preprocessing
- **`training.py`** - Training pipeline management
- **`inference.py`** - Inference and evaluation utilities
- **`utils.py`** - Helper functions and utilities

## Quick Start

### Basic Usage

```python
from mecagents import (
    ModelManager, DataProcessor, TrainingManager, InferenceManager,
    ModelConfig, TrainingConfig, DataConfig, InferenceConfig
)

# 1. Setup configurations
model_config = ModelConfig(model_name="unsloth/gemma-3-4b-pt")
data_config = DataConfig(dataset_name="CADCODER/GenCAD-Code")
training_config = TrainingConfig(max_steps=100)
inference_config = InferenceConfig()

# 2. Load and process data
data_processor = DataProcessor(data_config)
train_dataset, test_dataset = data_processor.load_dataset()
converted_dataset = data_processor.convert_dataset_to_conversations()

# 3. Setup model
model_manager = ModelManager(model_config)
model, processor = model_manager.load_model()
model = model_manager.setup_lora()
processor = model_manager.setup_chat_template()

# 4. Train
training_manager = TrainingManager(training_config)
model_manager.prepare_for_training()
trainer = training_manager.create_trainer(model, processor, converted_dataset)
results = training_manager.train()

# 5. Inference
model_manager.prepare_for_inference()
inference_manager = InferenceManager(inference_config)
sample = data_processor.get_sample_data(0)
messages = data_processor.create_inference_messages(sample["image"])
generated_code = inference_manager.generate_cad_code(model, processor, sample["image"], messages)
```

### Configuration Classes

#### ModelConfig
```python
@dataclass
class ModelConfig:
    model_name: str = "unsloth/gemma-3-4b-pt"
    load_in_4bit: bool = True
    use_gradient_checkpointing: str = "unsloth"
    
    # LoRA configuration
    finetune_vision_layers: bool = True
    finetune_language_layers: bool = True
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    # ... more options
```

#### DataConfig
```python
@dataclass
class DataConfig:
    dataset_name: str = "CADCODER/GenCAD-Code"
    cache_dir: str = "./Volumes/BIG-DATA/HUGGINGFACE_CACHE"
    num_proc: int = 16
    sample_size: Optional[int] = 10_000
    instruction: str = "Generate the CADQuery code..."
```

#### TrainingConfig
```python
@dataclass
class TrainingConfig:
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_steps: int = 100
    learning_rate: float = 2e-4
    output_dir: str = "outputs"
    # ... more options
```

#### InferenceConfig
```python
@dataclass
class InferenceConfig:
    max_new_tokens: int = 128
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 64
    use_cache: bool = True
```

## Module Details

### ModelManager

Handles all model-related operations:

```python
model_manager = ModelManager(model_config)

# Load model and processor
model, processor = model_manager.load_model()

# Setup LoRA for efficient fine-tuning
model = model_manager.setup_lora()

# Configure chat template
processor = model_manager.setup_chat_template("gemma-3")

# Switch between training and inference modes
model_manager.prepare_for_training()
model_manager.prepare_for_inference()

# Save models
model_manager.save_model("path/to/save")
model_manager.save_merged_model("path/to/merged")

# Monitor memory usage
memory_stats = model_manager.get_memory_stats()
```

### DataProcessor

Manages dataset operations:

```python
data_processor = DataProcessor(data_config)

# Load dataset
train_dataset, test_dataset = data_processor.load_dataset()

# Convert to conversation format for training
converted_dataset = data_processor.convert_dataset_to_conversations()

# Create inference messages
messages = data_processor.create_inference_messages(image, instruction)

# Get samples and statistics
sample = data_processor.get_sample_data(index=0)
stats = data_processor.get_dataset_stats()
data_processor.print_sample_info(index=2)
```

### TrainingManager

Handles the training process:

```python
training_manager = TrainingManager(training_config)

# Create trainer
trainer = training_manager.create_trainer(model, processor, dataset)

# Execute training
training_summary = training_manager.train()

# Save training state and logs
training_manager.save_trainer_state("output/path")
```

### InferenceManager

Manages inference and evaluation:

```python
inference_manager = InferenceManager(inference_config)

# Generate CAD code
generated_code = inference_manager.generate_cad_code(
    model, processor, image, messages, stream_output=True
)

# Evaluate single sample
result = inference_manager.evaluate_single_sample(model, processor, sample)

# Batch evaluation
results = inference_manager.batch_evaluate(model, processor, dataset, num_samples=5)

# Interactive generation with custom instructions
code = inference_manager.interactive_generation(model, processor, image, custom_instruction)

# Test different generation parameters
param_results = inference_manager.generate_with_different_parameters(
    model, processor, image, messages, parameter_sets
)
```

### Utilities

The `utils` module provides helpful functions:

```python
from mecagents.utils import (
    check_gpu_availability, save_config_to_json, 
    calculate_model_parameters, create_experiment_directory,
    cleanup_gpu_memory, validate_dataset_format
)

# Check GPU availability
gpu_info = check_gpu_availability()

# Save/load configurations
save_config_to_json(config_dict, "path", "config.json")
config = load_config_from_json("path/config.json")

# Create experiment directories with timestamps
exp_dir = create_experiment_directory("experiments", "my_experiment")

# Model analysis
param_info = calculate_model_parameters(model)

# Dataset validation
validation = validate_dataset_format(dataset)

# Memory management
cleanup_gpu_memory()
```

## Examples

The framework includes comprehensive examples:

1. **Quick Inference** (`examples.py`) - Basic inference with minimal setup
2. **Custom Training** - Full training pipeline with experiment tracking
3. **Evaluation & Comparison** - Model evaluation and output comparison
4. **Parameter Experiments** - Testing different generation parameters
5. **Custom Instructions** - Experimenting with different prompts

Run examples:
```bash
python examples.py
```

## Advanced Usage

### Custom Experiment Setup

```python
from mecagents.utils import create_experiment_directory, save_config_to_json

# Create experiment directory
exp_dir = create_experiment_directory("experiments", "custom_experiment")

# Custom configurations
model_config = ModelConfig(lora_r=32, lora_alpha=32)
training_config = TrainingConfig(max_steps=200, learning_rate=1e-4)

# Save configurations for reproducibility
save_config_to_json(model_config.__dict__, exp_dir, "model_config.json")
save_config_to_json(training_config.__dict__, exp_dir, "training_config.json")

# ... proceed with training
```

### Memory Management

```python
# Monitor memory throughout training
initial_memory = model_manager.get_memory_stats()

# ... training code ...

final_memory = model_manager.get_memory_stats()
memory_used = final_memory["current_memory_gb"] - initial_memory["current_memory_gb"]

# Clean up when done
cleanup_gpu_memory()
```

### Evaluation Pipeline

```python
# Comprehensive evaluation
results = []
for i in range(10):
    sample = data_processor.get_sample_data(i, from_test=True)
    result = inference_manager.evaluate_single_sample(model, processor, sample)
    
    # Compare with ground truth
    comparison = compare_model_outputs(result["predicted_code"], result["ground_truth"])
    result.update(comparison)
    results.append(result)

# Save evaluation results
save_config_to_json(results, exp_dir, "evaluation_results.json")
```

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- Unsloth
- Datasets
- TRL (Transformer Reinforcement Learning)

## Installation

The framework is designed to work with the existing project structure. Simply import the modules:

```python
from mecagents import ModelManager, DataProcessor, TrainingManager, InferenceManager
```

## Project Structure

```
mecagents/
├── __init__.py          # Main module exports
├── config.py           # Configuration classes
├── model.py            # Model management
├── data.py             # Data processing
├── training.py         # Training utilities
├── inference.py        # Inference utilities
└── utils.py            # Helper functions

# Usage examples
train_modular.py        # Main modular training script
examples.py            # Comprehensive examples
train.py              # Original training script (for reference)
```

This modular design makes it easy to:
- Customize individual components
- Experiment with different configurations
- Track experiments systematically
- Reuse code across different projects
- Maintain and extend functionality
