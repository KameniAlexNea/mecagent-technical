"""
Inference utilities for MecAgents
"""
import torch
from typing import Any, Dict, List, Optional
from transformers import TextStreamer

from .config import InferenceConfig


class InferenceManager:
    """Manages inference for trained vision-language models"""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
    
    def generate_cad_code(
        self, 
        model: Any, 
        processor: Any, 
        image: Any, 
        messages: List[Dict[str, Any]],
        stream_output: bool = True,
        **generation_kwargs
    ) -> str:
        """Generate CAD code from an image using the trained model"""
        
        # Apply chat template
        input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
        
        # Prepare inputs
        inputs = processor(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        
        # Setup generation parameters
        gen_params = {
            "max_new_tokens": generation_kwargs.get("max_new_tokens", self.config.max_new_tokens),
            "temperature": generation_kwargs.get("temperature", self.config.temperature),
            "top_p": generation_kwargs.get("top_p", self.config.top_p),
            "top_k": generation_kwargs.get("top_k", self.config.top_k),
            "use_cache": generation_kwargs.get("use_cache", self.config.use_cache),
        }
        
        # Setup streamer if requested
        streamer = None
        if stream_output:
            streamer = TextStreamer(processor.tokenizer, skip_prompt=True)
            gen_params["streamer"] = streamer
        
        # Generate
        with torch.no_grad():
            result = model.generate(**inputs, **gen_params)
        
        # Decode the result if not streaming
        if not stream_output:
            generated_text = processor.tokenizer.decode(result[0], skip_special_tokens=True)
            # Remove the input prompt from the generated text
            input_length = len(processor.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True))
            generated_code = generated_text[input_length:].strip()
            return generated_code
        
        return "Generated with streaming output"
    
    def evaluate_single_sample(
        self, 
        model: Any, 
        processor: Any, 
        sample: Dict[str, Any],
        instruction: Optional[str] = None
    ) -> Dict[str, Any]:
        """Evaluate model on a single sample"""
        
        if instruction is None:
            instruction = "Generate the CADQuery code needed to create the CAD for the provided image. Just the code, no other words."
        
        # Create messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"}, 
                    {"type": "text", "text": instruction}
                ],
            }
        ]
        
        # Generate prediction
        predicted_code = self.generate_cad_code(
            model, processor, sample["image"], messages, stream_output=False
        )
        
        # Get ground truth
        ground_truth = sample.get("cadquery", "")
        
        return {
            "predicted_code": predicted_code,
            "ground_truth": ground_truth,
            "prompt": sample.get("prompt", ""),
        }
    
    def batch_evaluate(
        self, 
        model: Any, 
        processor: Any, 
        dataset: Any,
        num_samples: int = 5,
        instruction: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Evaluate model on multiple samples"""
        
        results = []
        
        for i in range(min(num_samples, len(dataset))):
            print(f"\nEvaluating sample {i+1}/{num_samples}")
            
            sample = dataset[i]
            result = self.evaluate_single_sample(model, processor, sample, instruction)
            result["sample_index"] = i
            results.append(result)
            
            print(f"Sample {i+1} completed")
        
        return results
    
    def compare_with_baseline(
        self,
        model: Any,
        processor: Any,
        sample: Dict[str, Any],
        baseline_prediction: str,
        instruction: Optional[str] = None
    ) -> Dict[str, Any]:
        """Compare model output with baseline prediction"""
        
        # Get model prediction
        model_result = self.evaluate_single_sample(model, processor, sample, instruction)
        
        return {
            "model_prediction": model_result["predicted_code"],
            "baseline_prediction": baseline_prediction,
            "ground_truth": model_result["ground_truth"],
            "prompt": model_result["prompt"],
        }
    
    def interactive_generation(
        self, 
        model: Any, 
        processor: Any, 
        image: Any,
        custom_instruction: Optional[str] = None
    ) -> str:
        """Interactive CAD code generation with custom instruction"""
        
        instruction = custom_instruction or "Generate the CADQuery code needed to create the CAD for the provided image. Just the code, no other words."
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"}, 
                    {"type": "text", "text": instruction}
                ],
            }
        ]
        
        print(f"Instruction: {instruction}")
        print("Generated CAD Code:")
        print("-" * 50)
        
        generated_code = self.generate_cad_code(
            model, processor, image, messages, stream_output=True
        )
        
        return generated_code
    
    def generate_with_different_parameters(
        self,
        model: Any,
        processor: Any,
        image: Any,
        messages: List[Dict[str, Any]],
        parameter_sets: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate CAD code with different generation parameters"""
        
        results = []
        
        for i, params in enumerate(parameter_sets):
            print(f"\nGenerating with parameter set {i+1}: {params}")
            
            generated_code = self.generate_cad_code(
                model, processor, image, messages, 
                stream_output=False, **params
            )
            
            results.append({
                "parameters": params,
                "generated_code": generated_code
            })
        
        return results
