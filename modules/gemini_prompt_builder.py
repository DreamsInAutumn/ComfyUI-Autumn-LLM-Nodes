# Autumn Nodes - Gemini Prompt Builder.

import os
import json
import folder_paths

# --- Dependency Check ---
try:
    import google.generativeai as genai
    print("Google Generative AI library loaded successfully.")
except ImportError:
    print("#####################################################################")
    print("## Google Generative AI Library Not Found ")
    print("##")
    print("## Please install the required library by running the following command")
    print("## from your ComfyUI virtual environment's command line:")
    print("##")
    print("## pip install google-generativeai")
    print("##")
    print("#####################################################################")
    raise ImportError("Google Generative AI library not found. Please install it.")

# --- File Path Configuration ---
# Get the directory where this script is located
NODE_DIRECTORY = os.path.dirname(__file__)

# The API key folder
API_KEY_DIR = os.path.join(folder_paths.base_path, "api_keys")

# The model configuration JSON folder
MODELS_JSON_PATH = os.path.join(NODE_DIRECTORY, "gemini_models.json")


class GeminiPromptBuilder:
    """
    - Connects to Gemini API with an API key from a file.
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
        "gemini-1.0-pro",
        "gemini-pro-vision", # Note: This node doesn't handle image input
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
                    print(f"Gemini-Prompt-Builder: Loaded {len(models)} models from {MODELS_JSON_PATH}")
                    return models
                else:
                    print(f"Gemini-Prompt-Builder: WARNING - {MODELS_JSON_PATH} is not a valid list of strings. Using default models.")
                    return GeminiPromptBuilder.DEFAULT_MODELS
            except Exception as e:
                print(f"Gemini-Prompt-Builder: ERROR - Could not read or parse {MODELS_JSON_PATH}: {e}. Using default models.")
                return GeminiPromptBuilder.DEFAULT_MODELS
        else:
            # This 'else' block handles the case where the file is not found, fulfilling the request.
            print(f"Gemini-Prompt-Builder: INFO - {MODELS_JSON_PATH} not found. Using default model list. You can create this file to customize the list.")
            return GeminiPromptBuilder.DEFAULT_MODELS

    AVAILABLE_MODELS = load_models_from_json()

    @staticmethod
    def get_api_key_files():
        """Scans the 'api_keys' directory for .txt files."""
        if not os.path.exists(API_KEY_DIR):
            print(f"Gemini-Prompt-Builder: Creating API key directory at: {API_KEY_DIR}")
            os.makedirs(API_KEY_DIR)
        
        try:
            files = [f for f in os.listdir(API_KEY_DIR) if f.endswith('.txt')]
            if not files:
                return ["api_key_file_not_found.txt"]
            return sorted(files)
        except Exception as e:
            print(f"Gemini-Prompt-Builder: Error scanning API key directory: {e}")
            return ["error_scanning_api_keys_dir.txt"]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "main_prompt": ("STRING", {"multiline": True, "default": "A majestic dragon soaring through a cyberpunk city."}),
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
                    "default": "You are an expert prompt engineer. Expand the user's concept into a detailed, descriptive, and artistic paragraph suitable for an image generation model like Stable Diffusion. Focus on visual details, lighting, composition, and mood."
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response_text",)
    FUNCTION = "generate"
    CATEGORY = "Autumn Nodes"
    OUTPUT_NODE = True

    def generate(self, main_prompt, bypass, api_key_file, model, temperature, top_p, top_k, max_tokens, force_deterministic, system_prompt=None):
        
        if bypass:
            print("Gemini-Prompt-Builder: Bypass enabled. Returning last generated prompt.")
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

        try:
            genai.configure(api_key=api_key)

            final_temperature = 0.0 if force_deterministic else temperature
            
            if force_deterministic:
                print("Gemini-Prompt-Builder: Deterministic output enabled. Forcing temperature to 0.")

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
            
            print(f"Gemini-Prompt-Builder: Sending request... (Model: {model}, Temp: {final_temperature}, Top-P: {top_p}, Top-K: {top_k})")
            response = gemini_model.generate_content(main_prompt)
            
            if not response.parts:
                try:
                    block_reason = response.prompt_feedback.block_reason.name
                    error_message = f"ERROR: The prompt was blocked by Google's safety filters. Reason: {block_reason}. Please adjust your prompt."
                except Exception:
                    error_message = "ERROR: Received an empty response from Gemini. This is often due to safety filters. Please adjust your prompt."
                
                print(f"Gemini-Prompt-Builder: {error_message}")
                self.last_response = error_message
                return (error_message,)
            
            response_text = response.text
            print("Gemini-Prompt-Builder: Received response successfully.")
            
            self.last_response = response_text

        except Exception as e:
            error_message = f"ERROR: Could not get response from Gemini API. Check console for details. Error: {e}"
            print(f"Gemini-Prompt-Builder: {error_message}")
            self.last_response = error_message
            return (error_message,)

        return (response_text,)


# --- Node Registration ---
NODE_CLASS_MAPPINGS = {
    "GeminiPromptBuilder": GeminiPromptBuilder
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiPromptBuilder": "Gemini-Prompt-Builder"
}