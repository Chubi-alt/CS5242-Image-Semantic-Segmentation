import os
import datetime

def generate_run_name(model_name="unetpp", backbone="resnet34", extra_tag=""):
    """
    Generates a unique experiment name based on model architecture, backbone, 
    optional tags, and the current timestamp.

    Args:
        model_name (str): Name of the primary architecture (e.g., 'unetpp').
        backbone (str): Name of the encoder backbone (e.g., 'resnet50').
        extra_tag (str): Any additional experiment details (e.g., 'clear_weather', 'lr1e-4').

    Returns:
        str: A formatted string like 'unetpp_resnet50_clear_weather_20260310_1315'.
    """
    # Fetch current time and format as YYYYMMDD_HHMM
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    
    # Construct the base identifier
    components = [model_name, backbone]
    
    # Include extra tag if provided to distinguish specific ablation studies
    if extra_tag:
        components.append(extra_tag)
        
    components.append(timestamp)
    
    # Join all components with underscores for safe filename usage
    run_name = "_".join(components)
    
    return run_name

def setup_experiment_directories(run_name, base_dirs=["checkpoints", "outputs", "logs"]):
    """
    Creates isolated directories for a specific experiment run.
    This prevents outputs from different runs from overwriting each other.

    Args:
        run_name (str): The unique name generated for the current run.
        base_dirs (list): List of parent directories to organize the outputs.

    Returns:
        dict: A dictionary mapping directory types to their specific paths.
    """
    paths = {}
    for d in base_dirs:
        # e.g., creates 'outputs/unetpp_resnet34_20260310_1315'
        specific_path = os.path.join(d, run_name)
        os.makedirs(specific_path, exist_ok=True)
        paths[d] = specific_path
        
    return paths