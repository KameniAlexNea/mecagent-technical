#!/usr/bin/env python3
"""
MecAgents Training Script - Modular CAD Code Generation Training

This script demonstrates how to use the modular MecAgents framework
for training vision-language models on CAD code generation tasks.
"""

import os
from mecagents import (
    ModelManager, DataProcessor, TrainingManager, InferenceManager,
    ModelConfig, TrainingConfig, DataConfig, InferenceConfig
)


def main():
    """Main training pipeline"""
    print("=" * 60)
    print("MecAgents - CAD Code Generation Training")
    print("=" * 60)
    
    # 1. Setup Configurations
    print("\n1. Setting up configurations...")
    
    model_config = ModelConfig(
        model_name="unsloth/gemma-3-4b-pt",
        load_in_4bit=True,
        lora_r=16,
        lora_alpha=16,
    )
    
    data_config = DataConfig(
        dataset_name="CADCODER/GenCAD-Code",
        sample_size=10_000,  # Use 10k samples for faster training
        cache_dir="./Volumes/BIG-DATA/HUGGINGFACE_CACHE"
    )
    
    training_config = TrainingConfig(
        max_steps=100,  # Quick training for demonstration
        learning_rate=2e-4,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        output_dir="outputs/mecagents_model"
    )
    
    inference_config = InferenceConfig(
        max_new_tokens=128,
        temperature=1.0,
        top_p=0.95,
        top_k=64
    )
    
    # 2. Load and Process Data
    print("\n2. Loading and processing data...")
    data_processor = DataProcessor(data_config)
    train_dataset, test_dataset = data_processor.load_dataset()
    
    # Print dataset statistics
    stats = data_processor.get_dataset_stats()
    print(f"Dataset stats: {stats}")
    
    # Show a sample
    data_processor.print_sample_info(index=2)
    
    # Convert to conversation format
    converted_dataset = data_processor.convert_dataset_to_conversations()
    
    # 3. Setup Model
    print("\n3. Setting up model...")
    model_manager = ModelManager(model_config)
    model, processor = model_manager.load_model()
    model = model_manager.setup_lora()
    processor = model_manager.setup_chat_template("gemma-3")
    
    # Show memory stats
    memory_stats = model_manager.get_memory_stats()
    print(f"Initial memory stats: {memory_stats}")
    
    # 4. Test Base Model Performance
    print("\n4. Testing base model performance...")
    inference_manager = InferenceManager(inference_config)
    model_manager.prepare_for_inference()
    
    # Test on a sample
    sample = data_processor.get_sample_data(index=2)
    messages = data_processor.create_inference_messages(sample["image"])
    
    print("Base model output:")
    base_output = inference_manager.generate_cad_code(
        model, processor, sample["image"], messages, stream_output=True
    )
    
    # 5. Train the Model
    print("\n5. Starting training...")
    training_manager = TrainingManager(training_config)
    model_manager.prepare_for_training()
    
    trainer = training_manager.create_trainer(model, processor, converted_dataset)
    training_summary = training_manager.train()
    
    # 6. Test Trained Model
    print("\n6. Testing trained model...")
    model_manager.prepare_for_inference()
    
    # Test on the same sample
    print("Trained model output:")
    trained_output = inference_manager.generate_cad_code(
        model, processor, sample["image"], messages, stream_output=True
    )
    
    # 7. Evaluate on Multiple Samples
    print("\n7. Evaluating on multiple samples...")
    if test_dataset:
        evaluation_results = inference_manager.batch_evaluate(
            model, processor, test_dataset, num_samples=3
        )
        
        for i, result in enumerate(evaluation_results):
            print(f"\nSample {i+1}:")
            print(f"Predicted: {result['predicted_code'][:100]}...")
            print(f"Ground truth: {result['ground_truth'][:100]}...")
    
    # 8. Save the Model
    print("\n8. Saving the model...")
    os.makedirs(training_config.output_dir, exist_ok=True)
    
    # Save LoRA adapters
    model_manager.save_model(f"{training_config.output_dir}/lora_adapters")
    
    # Save merged model
    model_manager.save_merged_model(f"{training_config.output_dir}/merged_model")
    
    # Save training logs
    training_manager.save_trainer_state(training_config.output_dir)
    
    print("\n9. Final memory stats:")
    final_memory_stats = model_manager.get_memory_stats()
    print(f"Final memory stats: {final_memory_stats}")
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print(f"Models saved to: {training_config.output_dir}")
    print("=" * 60)


def demo_inference_only():
    """Demonstration of using the framework for inference only"""
    print("\n" + "=" * 60)
    print("MecAgents - Inference Demo")
    print("=" * 60)
    
    # Setup configurations
    model_config = ModelConfig(model_name="unsloth/gemma-3-4b-pt")
    data_config = DataConfig(sample_size=100)  # Small sample for demo
    inference_config = InferenceConfig()
    
    # Load data
    data_processor = DataProcessor(data_config)
    train_dataset, _ = data_processor.load_dataset()
    
    # Load model
    model_manager = ModelManager(model_config)
    model, processor = model_manager.load_model()
    model_manager.setup_chat_template()
    model_manager.prepare_for_inference()
    
    # Setup inference
    inference_manager = InferenceManager(inference_config)
    
    # Interactive generation
    sample = data_processor.get_sample_data(index=0)
    generated_code = inference_manager.interactive_generation(
        model, processor, sample["image"],
        custom_instruction="Create detailed CADQuery code for this mechanical part."
    )
    
    print("Inference demo completed!")


def test_different_parameters():
    """Test generation with different parameters"""
    print("\n" + "=" * 60)
    print("MecAgents - Parameter Testing")
    print("=" * 60)
    
    # Quick setup
    model_config = ModelConfig(model_name="unsloth/gemma-3-4b-pt")
    data_config = DataConfig(sample_size=10)
    inference_config = InferenceConfig()
    
    data_processor = DataProcessor(data_config)
    train_dataset, _ = data_processor.load_dataset()
    
    model_manager = ModelManager(model_config)
    model, processor = model_manager.load_model()
    model_manager.setup_chat_template()
    model_manager.prepare_for_inference()
    
    inference_manager = InferenceManager(inference_config)
    
    # Test different parameter sets
    sample = data_processor.get_sample_data(index=0)
    messages = data_processor.create_inference_messages(sample["image"])
    
    parameter_sets = [
        {"temperature": 0.1, "top_p": 0.9},  # More deterministic
        {"temperature": 1.0, "top_p": 0.95}, # Default
        {"temperature": 1.5, "top_p": 0.9},  # More creative
    ]
    
    results = inference_manager.generate_with_different_parameters(
        model, processor, sample["image"], messages, parameter_sets
    )
    
    for i, result in enumerate(results):
        print(f"\nResult {i+1} with {result['parameters']}:")
        print(f"Generated: {result['generated_code'][:150]}...")


if __name__ == "__main__":
    # Run the main training pipeline
    main()
    
    # Uncomment to run additional demos
    # demo_inference_only()
    # test_different_parameters()
