"""
Autumn Nodes - Custom ComfyUI Node Library
"""
import sys
import os
import importlib

# Add modules directory to path
modules_path = os.path.join(os.path.dirname(__file__), "modules")
if modules_path not in sys.path:
    sys.path.append(modules_path)

# --- List of modules to load ---
module_names = [
    "llm_prompt_builder",
    "gemini_prompt_builder",
    "gemini_img2prompt",
]

# --- ComfyUI Dictionaries ---
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# --- Loop through and load each module ---
for module_name in module_names:
    try:
        # Import the module from the list
        module = importlib.import_module(module_name)

        # Get the mappings from the module and add them to our main
        NODE_CLASS_MAPPINGS.update(getattr(module, "NODE_CLASS_MAPPINGS", {}))
        NODE_DISPLAY_NAME_MAPPINGS.update(getattr(module, "NODE_DISPLAY_NAME_MAPPINGS", {}))
    except ImportError as e:
        print(f"[Autumn Nodes] Warning: Could not import module '{module_name}'. {e}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]