# Autumn Nodes - Gemini Image To Prompt.

import os
import json
import importlib

# --- Dependency Check ---
# A dictionary of required modules and their corresponding pip package names.
# The key is the name to import, and the value is the name to install.
REQUIRED_LIBS = {
    "google.generativeai": "google-generativeai",
    "PIL": "Pillow",
    "torch": "torch",
    "numpy": "numpy"
}

missing_libs = []
for module_name, package_name in REQUIRED_LIBS.items():
    try:
        importlib.import_module(module_name)
    except ImportError:
        missing_libs.append(package_name)

if missing_libs:
    # Get the directory of the current node to provide a full path.
    node_dir = os.path.dirname(__file__)
    requirements_path = os.path.join(node_dir, "requirements.txt")
    
    # Build the streamlined error message.
    error_message = (
        "#####################################################################\n"
        "## Missing Required Python Libraries for Gemini-Image-To-Prompt\n"
        "##\n"
        f"## The following libraries are missing: {', '.join(missing_libs)}\n"
        "##\n"
        "## This node includes a 'requirements.txt' file to install them.\n"
        "## Please run the following command from your ComfyUI virtual\n"
        "## environment's command line:\n"
        "##\n"
        f"## pip install -r \"{requirements_path}\"\n"
        "##\n"
        "## After installation, please restart ComfyUI.\n"
        "#####################################################################"
    )
    print(error_message)
    raise ImportError(f"Gemini-Image-To-Prompt missing dependencies: {', '.join(missing_libs)}")

# --- Imports (run after successful dependency check) ---
import folder_paths
import google.generativeai as genai
from PIL import Image
import torch
import numpy

print("Gemini-Image-To-Prompt: All dependencies loaded successfully.")

# --- File Path Configuration ---
# Get the directory where this script is located
NODE_DIRECTORY = os.path.dirname(__file__)

# The API key folder
API_KEY_DIR = os.path.join(folder_paths.base_path, "api_keys")

# The model configuration JSON folder
MODELS_JSON_PATH = os.path.join(NODE_DIRECTORY, "gemini_models.json")


class GeminiImageToPrompt:
    """
    - Connects to Gemini API with an API key from a file.
    - Takes an image and a text prompt as input for multimodal analysis.
    - Loads model list from gemini_models.json located in the node's directory.
    - Control over sampling parameters (temp, top-p, top-k).
    - Deterministic mode for reproducible results (sets temp to 0).
    - Stateful bypass toggle to lock the last generated prompt.
    """

    def __init__(self):
        self.last_response = ""

    DEFAULT_MODELS = [
        "gemini-1.5-pro-latest",
        "gemini-1.5-flash-latest",
        "gemini-pro-vision",
        "gemini-1.0-pro", # Note: This model does not handle image input and will error
    ]
    
    @staticmethod
    def load_models_from_json():
        """
        Loads the model list from gemini_models.json in the node's directory.
        If the file doesn't exist or is invalid, it returns a default list.
        """
        if os.path.exists(MODELS_JSON_PATH):
            try:
                with open(MODELS_JSON_PATH, 'r') as f:
                    models = json.load(f)
                if isinstance(models, list) and all(isinstance(m, str) for m in models) and models:
                    print(f"Gemini-Image-To-Prompt: Loaded {len(models)} models from {MODELS_JSON_PATH}")
                    return models
                else:
                    print(f"Gemini-Image-To-Prompt: WARNING - {MODELS_JSON_PATH} is not a valid list of strings. Using default models.")
                    return GeminiImageToPrompt.DEFAULT_MODELS
            except Exception as e:
                print(f"Gemini-Image-To-Prompt: ERROR - Could not read or parse {MODELS_JSON_PATH}: {e}. Using default models.")
                return GeminiImageToPrompt.DEFAULT_MODELS
        else:
            # This 'else' block handles the case where the file is not found, fulfilling the request.
            print(f"Gemini-Image-To-Prompt: INFO - {MODELS_JSON_PATH} not found. Using default model list. You can create this file to customize the list.")
            return GeminiImageToPrompt.DEFAULT_MODELS

    AVAILABLE_MODELS = load_models_from_json()

    @staticmethod
    def get_api_key_files():
        """Scans the 'api_keys' directory for .txt files."""
        if not os.path.exists(API_KEY_DIR):
            print(f"Gemini-Image-To-Prompt: Creating API key directory at: {API_KEY_DIR}")
            os.makedirs(API_KEY_DIR)
        
        try:
            files = [f for f in os.listdir(API_KEY_DIR) if f.endswith('.txt')]
            if not files:
                return ["api_key_file_not_found.txt"]
            return sorted(files)
        except Exception as e:
            print(f"Gemini-Image-To-Prompt: Error scanning API key directory: {e}")
            return ["error_scanning_api_keys_dir.txt"]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_image": ("IMAGE",),
                "user_prompt": ("STRING", {"multiline": True, "default": "Describe this image in detail. Create a vivid and evocative prompt for an image generation model."}),
                "bypass": ("BOOLEAN", {"default": False}),
                "api_key_file": (s.get_api_key_files(),),
                "model": (s.AVAILABLE_MODELS,),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.05}),
                "top_k": ("INT", {"default": 40, "min": 1, "max": 128, "step": 1}),
                "max_tokens": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 64}),
                "force_deterministic": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "system_prompt": ("STRING", {
                    "multiline": True, 
                    "default": "You are an expert at analyzing images and creating artistic, descriptive text prompts for image generation models like Stable Diffusion. Describe the image's subject, environment, lighting, composition, colors, and mood with rich detail."
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response_text",)
    FUNCTION = "generate"
    CATEGORY = "Autumn Nodes"
    OUTPUT_NODE = True

    def generate(self, input_image, user_prompt, bypass, api_key_file, model, temperature, top_p, top_k, max_tokens, force_deterministic, system_prompt=None):
        
        if bypass:
            print("Gemini-Image-To-Prompt: Bypass enabled. Returning last generated prompt.")
            return (self.last_response,)

        if "not_found" in api_key_file or "error_scanning" in api_key_file:
            error_message = f"ERROR: No API key file selected or found. Please create a .txt file with your API key inside the '{API_KEY_DIR}' folder and refresh ComfyUI."
            self.last_response = error_message
            return (error_message,)

        try:
            key_path = os.path.join(API_KEY_DIR, api_key_file)
            with open(key_path, 'r') as f:
                api_key = f.read().strip()
            if not api_key:
                raise ValueError("API key file is empty.")
        except Exception as e:
            error_message = f"ERROR: Could not read API key from '{key_path}'. Please ensure the file exists and is not empty. Details: {e}"
            self.last_response = error_message
            return (error_message,)

        # --- Image Conversion ---
        # Convert the input tensor to a PIL Image
        try:
            i = 255. * input_image[0].cpu().numpy()
            img = Image.fromarray(numpy.clip(i, 0, 255).astype(numpy.uint8))
        except Exception as e:
            error_message = f"ERROR: Could not convert input image tensor to PIL Image. Details: {e}"
            self.last_response = error_message
            return (error_message,)

        try:
            genai.configure(api_key=api_key)

            final_temperature = 0.0 if force_deterministic else temperature
            
            if force_deterministic:
                print("Gemini-Image-To-Prompt: Deterministic output enabled. Forcing temperature to 0.")

            # Warn if a non-multimodal model is selected
            if 'vision' not in model and '1.5' not in model:
                print(f"Gemini-Image-To-Prompt: WARNING - The selected model '{model}' may not support image inputs. This may cause an error. Use a 'vision' or '1.5' model for image analysis.")

            generation_config = genai.GenerationConfig(
                temperature=final_temperature,
                top_p=top_p,
                top_k=top_k,
                max_output_tokens=max_tokens,
            )
            
            gemini_model = genai.GenerativeModel(
                model_name=model,
                generation_config=generation_config,
                system_instruction=system_prompt if system_prompt and system_prompt.strip() else None,
            )
            
            print(f"Gemini-Image-To-Prompt: Sending request with image... (Model: {model}, Temp: {final_temperature}, Top-P: {top_p}, Top-K: {top_k})")
            
            # Create the multimodal prompt
            prompt_parts = [user_prompt, img]
            response = gemini_model.generate_content(prompt_parts)
            
            if not response.parts:
                try:
                    block_reason = response.prompt_feedback.block_reason.name
                    error_message = f"ERROR: The prompt was blocked by Google's safety filters. Reason: {block_reason}. Please adjust your prompt."
                except Exception:
                    error_message = "ERROR: Received an empty response from Gemini. This is often due to safety filters. Please adjust your prompt."
                
                print(f"Gemini-Image-To-Prompt: {error_message}")
                self.last_response = error_message
                return (error_message,)
            
            response_text = response.text
            print("Gemini-Image-To-Prompt: Received response successfully.")
            
            self.last_response = response_text

        except Exception as e:
            error_message = f"ERROR: Could not get response from Gemini API. Check console for details. Error: {e}"
            print(f"Gemini-Image-To-Prompt: {error_message}")
            self.last_response = error_message
            return (error_message,)

        return (response_text,)


# --- Node Registration ---
NODE_CLASS_MAPPINGS = {
    "GeminiImageToPrompt": GeminiImageToPrompt
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiImageToPrompt": "Gemini-Image-To-Prompt"
}