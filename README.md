# MecAgent Technical

A modular framework for training CAD code generation models using vision-language architectures. This project provides both the original training approach and a new modular framework (MecAgents) for better maintainability and reusability.

## 🚀 Quick Start

### Using the Modular Framework (Recommended)

```python
from mecagents import (
    ModelManager, DataProcessor, TrainingManager, InferenceManager,
    ModelConfig, TrainingConfig, DataConfig, InferenceConfig
)

# Configure your training
model_config = ModelConfig(model_name="unsloth/gemma-3-4b-pt")
data_config = DataConfig(dataset_name="CADCODER/GenCAD-Code", sample_size=10_000)
training_config = TrainingConfig(max_steps=100)

# Run training
data_processor = DataProcessor(data_config)
train_dataset, _ = data_processor.load_dataset()

model_manager = ModelManager(model_config)
model, processor = model_manager.load_model()

training_manager = TrainingManager(training_config)
# ... training pipeline
```

### Run Examples

```bash
# Quick demonstration
python demo.py

# Full modular training
python train_modular.py

# Migration comparison
python migrate.py

# Comprehensive examples
python examples.py
```

## 📁 Project Structure

```
mecagent-technical/
├── 📄 train_modular.py      # New modular training script 
├── 📄 good_luck.ipynb       # Jupyter notebook examples
├── 📄 01-ExploreCADDataset.ipynb
├── 📄 02-EvaluateExistingSolution.ipynb
└── 📁 mecagents/            # 🎯 Modular framework
    ├── __init__.py          # Module exports
    ├── config.py            # Configuration classes
    ├── model.py             # Model management
    ├── data.py              # Data processing
    ├── training.py          # Training utilities
    ├── inference.py         # Inference utilities
    ├── utils.py             # Helper functions
```

## 🏗️ Framework Architecture

### Core Modules

The **MecAgents** framework is organized into focused modules:

#### 🔧 Configuration (`config.py`)

- `ModelConfig`: Model and LoRA settings
- `DataConfig`: Dataset and preprocessing settings
- `TrainingConfig`: Training hyperparameters
- `InferenceConfig`: Generation parameters

#### 🤖 Model Management (`model.py`)

- Load and configure vision-language models
- Setup LoRA adapters for efficient fine-tuning
- Handle model state transitions (training/inference)
- Memory monitoring and model saving

#### 📊 Data Processing (`data.py`)

- Load and validate CAD datasets
- Convert to conversation format for training
- Sample management and statistics
- Inference message creation

#### 🏋️ Training (`training.py`)

- Managed training pipeline with SFTTrainer
- Training metrics and memory tracking
- Automatic logging and state saving
- Configuration-driven training setup

#### 🎯 Inference (`inference.py`)

- Single and batch inference
- Custom instruction support
- Parameter experimentation
- Model evaluation utilities

#### 🛠️ Utilities (`utils.py`)

- GPU availability checking
- Configuration serialization
- Experiment directory management
- Memory management helpers

## 📖 Usage Examples

### 1. Quick Training

```python
from mecagents import ModelManager, DataProcessor, TrainingManager
from mecagents import ModelConfig, DataConfig, TrainingConfig

# Simple setup
model_config = ModelConfig()
data_config = DataConfig(sample_size=1000)
training_config = TrainingConfig(max_steps=50)

# Execute training
data_processor = DataProcessor(data_config)
train_dataset, _ = data_processor.load_dataset()

model_manager = ModelManager(model_config)
model, processor = model_manager.load_model()
model = model_manager.setup_lora()

training_manager = TrainingManager(training_config)
model_manager.prepare_for_training()
trainer = training_manager.create_trainer(model, processor, converted_dataset)
results = training_manager.train()
```

### 2. Inference and Evaluation

```python
from mecagents import InferenceManager, InferenceConfig

# Setup inference
inference_config = InferenceConfig(temperature=0.7, max_new_tokens=256)
inference_manager = InferenceManager(inference_config)

# Generate CAD code
sample = data_processor.get_sample_data(0)
messages = data_processor.create_inference_messages(sample["image"])
generated_code = inference_manager.generate_cad_code(
    model, processor, sample["image"], messages
)

# Batch evaluation
results = inference_manager.batch_evaluate(
    model, processor, test_dataset, num_samples=10
)
```

## 🎯 Key Features

- **🔧 Configuration-Driven**: All settings managed through typed configuration classes
- **🏗️ Modular Design**: Independent components that can be used separately
- **🔄 Easy Migration**: Smooth transition from original monolithic approach
- **📊 Built-in Evaluation**: Comprehensive model assessment tools
- **🎛️ Experiment Tracking**: Built-in support for comparing different configurations
- **💾 Model Management**: Automated saving, loading, and memory monitoring
- **📝 Type Safety**: Full type hints for better development experience
- **📚 Documentation**: Comprehensive documentation and examples

## 🛠️ Installation

The framework works with the existing project dependencies:

```bash
# Install dependencies (if not already installed)
uv sync

# The mecagents framework is included in this repository
# Simply import and use the modules
```

## 🤝 Contributing

The modular design makes it easy to extend the framework:

1. **Use argument parser to run the configuration code**
2. Add support for GPRO in training (usage of metrics)
