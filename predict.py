import os

os.environ["WANDB_PROJECT"] = "mecagents-cad-code-generation"
os.environ["WANDB_WATCH"] = "none"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import unsloth  # noqa: F401
import json
from mecagents import (
    ModelManager,
    DataProcessor,
    InferenceManager,
    ModelConfig,
    DataConfig,
    InferenceConfig,
)


def demo_inference_only():
    """Demonstration of using the framework for inference only"""
    print("\n" + "=" * 60)
    print("MecAgents - Inference Demo")
    print("=" * 60)

    # Setup configurations
    save_dir = "llm_output/mecagents_model_imp.json"
    model_config = ModelConfig(model_name="outputs/mecagents_model/merged_model")
    data_config = DataConfig(sample_size=100)  # Small sample for demo
    inference_config = InferenceConfig(
        max_new_tokens=256
    )

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

    results = []

    # Interactive generation
    for index in range(25):
        sample = data_processor.get_sample_data(index=index, from_test=False)
        generated_code = inference_manager.interactive_generation(
            model,
            processor,
            sample["image"],
            custom_instruction=sample["prompt"],
            stream_output=False,
        )
        results.append({
            "predicted_code": generated_code,
            "ground_truth": sample.get("cadquery", ""),
        })
    
    # Save results
    with open(save_dir, "w") as f:
        json.dump(results, f, indent=4)

    print("Inference demo completed!")

if __name__ == "__main__":
    demo_inference_only()