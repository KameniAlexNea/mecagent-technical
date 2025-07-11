"""
Utility functions for MecAgents
"""
import json
import os
import torch
from typing import Dict, Any, List, Optional
from pathlib import Path


def save_config_to_json(config_dict: Dict[str, Any], output_path: str, filename: str = "config.json"):
    """Save configuration dictionary to JSON file"""
    os.makedirs(output_path, exist_ok=True)
    config_path = os.path.join(output_path, filename)
    
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"Configuration saved to: {config_path}")
    return config_path


def load_config_from_json(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"Configuration loaded from: {config_path}")
    return config


def check_gpu_availability() -> Dict[str, Any]:
    """Check GPU availability and specs"""
    if not torch.cuda.is_available():
        return {
            "available": False,
            "message": "CUDA not available"
        }
    
    gpu_count = torch.cuda.device_count()
    current_device = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(current_device)
    gpu_properties = torch.cuda.get_device_properties(current_device)
    
    total_memory = gpu_properties.total_memory / 1024 / 1024 / 1024  # GB
    
    return {
        "available": True,
        "gpu_count": gpu_count,
        "current_device": current_device,
        "gpu_name": gpu_name,
        "total_memory_gb": round(total_memory, 2),
        "compute_capability": f"{gpu_properties.major}.{gpu_properties.minor}"
    }


def format_memory_usage(bytes_size: int) -> str:
    """Format memory usage in human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} TB"


def create_experiment_directory(base_path: str, experiment_name: str) -> str:
    """Create a unique experiment directory with timestamp"""
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(base_path, f"{experiment_name}_{timestamp}")
    
    os.makedirs(experiment_dir, exist_ok=True)
    print(f"Experiment directory created: {experiment_dir}")
    
    return experiment_dir


def save_training_results(
    results: Dict[str, Any], 
    output_path: str, 
    filename: str = "training_results.json"
):
    """Save training results to JSON file"""
    return save_config_to_json(results, output_path, filename)


def calculate_model_parameters(model) -> Dict[str, int]:
    """Calculate the number of parameters in a model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": total_params - trainable_params,
        "trainable_percentage": round(trainable_params / total_params * 100, 2) if total_params > 0 else 0
    }


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Setup logging configuration"""
    import logging
    
    # Set log level
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup handlers
    handlers = [logging.StreamHandler()]
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # Configure logging
    logging.basicConfig(
        level=level,
        handlers=handlers,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    return logging.getLogger(__name__)


def validate_dataset_format(dataset) -> Dict[str, Any]:
    """Validate dataset format for CAD code generation"""
    required_fields = ["image", "prompt", "cadquery"]
    
    if len(dataset) == 0:
        return {"valid": False, "error": "Dataset is empty"}
    
    # Check first sample
    sample = dataset[0]
    missing_fields = [field for field in required_fields if field not in sample]
    
    if missing_fields:
        return {
            "valid": False,
            "error": f"Missing required fields: {missing_fields}"
        }
    
    # Additional validation
    validation_results = {
        "valid": True,
        "total_samples": len(dataset),
        "sample_fields": list(sample.keys()),
        "image_type": type(sample["image"]).__name__,
        "avg_prompt_length": sum(len(str(s.get("prompt", ""))) for s in dataset) / len(dataset),
        "avg_code_length": sum(len(str(s.get("cadquery", ""))) for s in dataset) / len(dataset),
    }
    
    return validation_results


def compare_model_outputs(
    prediction: str, 
    ground_truth: str, 
    metrics: List[str] = None
) -> Dict[str, Any]:
    """Compare model prediction with ground truth"""
    if metrics is None:
        metrics = ["length_ratio", "word_overlap", "line_count"]
    
    results = {}
    
    if "length_ratio" in metrics:
        pred_len = len(prediction)
        gt_len = len(ground_truth)
        results["length_ratio"] = pred_len / gt_len if gt_len > 0 else 0
    
    if "word_overlap" in metrics:
        pred_words = set(prediction.lower().split())
        gt_words = set(ground_truth.lower().split())
        if gt_words:
            results["word_overlap_ratio"] = len(pred_words & gt_words) / len(gt_words)
        else:
            results["word_overlap_ratio"] = 0
    
    if "line_count" in metrics:
        results["prediction_lines"] = len(prediction.split('\n'))
        results["ground_truth_lines"] = len(ground_truth.split('\n'))
    
    return results


def cleanup_gpu_memory():
    """Clean up GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("GPU memory cache cleared")
    else:
        print("CUDA not available, no GPU memory to clean")


def get_project_structure(root_path: str, max_depth: int = 3) -> Dict[str, Any]:
    """Get project directory structure"""
    structure = {}
    
    def _scan_directory(path: Path, current_depth: int = 0):
        if current_depth >= max_depth:
            return "..."
        
        items = {}
        try:
            for item in sorted(path.iterdir()):
                if item.name.startswith('.'):
                    continue
                
                if item.is_file():
                    items[item.name] = f"file ({item.stat().st_size} bytes)"
                elif item.is_dir():
                    items[item.name] = _scan_directory(item, current_depth + 1)
        except PermissionError:
            return "Permission denied"
        
        return items
    
    structure = _scan_directory(Path(root_path))
    return structure
