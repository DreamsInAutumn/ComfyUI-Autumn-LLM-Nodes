import requests
import folder_paths
import os
import random
import threading
import time

# --- Dependency Check ---
try:
    from openai import OpenAI
    print("OpenAI library loaded successfully.")
except ImportError:
    print("#####################################################################")
    print("## OpenAI Library Not Found ")
    print("##")
    print("## Please install the required library by running the following command")
    print("## from your ComfyUI virtual environment's command line:")
    print("##")
    print("## pip install openai")
    print("##")
    print("#####################################################################")
    raise ImportError("OpenAI library not found. Please install it using 'pip install openai'")

# --- Node Configuration ---
# Add the IP:Port of your LM Studio server here.
LMSTUDIO_SERVERS = [
    "http://127.0.0.1:5000",
    "http://127.0.0.1:1234",
]

class LLMPromptBuilder:
    """
    A ComfyUI node to connect to an LM Studio backend, send prompts,
    and return the generated text response with seed control.
    This version uses a non-blocking background thread to find available models.
    """
    
    # --- NON-BLOCKING MODEL CACHING ---
    _available_models_cache = None
    _lock = threading.Lock()
    _is_checking_models = False

    @staticmethod
    def _fetch_models_background():
        """
        This function runs in a background thread to fetch models without blocking the UI.
        """
        print("LLM-Prompt-Builder: Starting background check for LM Studio models...")
        
        found_models = []
        for server_url in LMSTUDIO_SERVERS:
            try:
                # Short timeout for quick checks
                response = requests.get(f"{server_url}/v1/models", timeout=2) 
                if response.status_code == 200:
                    models_data = response.json()
                    model_ids = [model['id'] for model in models_data.get('data', [])]
                    if model_ids:
                        print(f"LLM-Prompt-Builder: Found models on {server_url}: {model_ids}")
                        found_models.extend(model_ids)
                        # Optional: break after finding the first server
                        # break 
            except requests.exceptions.RequestException:
                # This is expected if a server is not running, so we don't print an error here
                # to avoid cluttering the console.
                pass
        
        with LLMPromptBuilder._lock:
            if found_models:
                # Use a set to remove duplicates if multiple servers have the same model
                LLMPromptBuilder._available_models_cache = sorted(list(set(found_models)))
            else:
                print("LLM-Prompt-Builder: Could not find any running LM Studio server or models.")
                LLMPromptBuilder._available_models_cache = ["not-found-check-lm-studio-server"]
            
            LLMPromptBuilder._is_checking_models = False
            print(f"LLM-Prompt-Builder: Background check complete. Models available: {LLMPromptBuilder._available_models_cache}")

    @classmethod
    def INPUT_TYPES(s):
        # --- Start the background check if it hasn't been started ---
        with s._lock:
            if s._available_models_cache is None and not s._is_checking_models:
                s._is_checking_models = True
                s._available_models_cache = ["checking..."]
                # Run the check in a daemon thread so it doesn't block ComfyUI exit
                thread = threading.Thread(target=s._fetch_models_background, daemon=True)
                thread.start()
        
        # --- Immediately return the current state of the cache ---
        # This is non-blocking. It will show "checking..." initially, then update
        # on the next interaction with the node (e.g., reconnecting a noodle or refreshing the page).
        current_models = s._available_models_cache if s._available_models_cache is not None else ["checking..."]
        
        return {
            "required": {
                "main_prompt": ("STRING", {"multiline": True, "default": "Describe a mystical forest at twilight."}),
                "server_url": (LMSTUDIO_SERVERS,),
                "model": (current_models,),
                "max_tokens": ("INT", {"default": 256, "min": 1, "max": 8192, "step": 64}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "system_prompt": ("STRING", {
                    "multiline": True, 
                    "default": "You are a helpful and creative assistant who generates vivid descriptions for image generation prompts."
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response_text",)
    FUNCTION = "generate"
    CATEGORY = "Autumn Nodes"
    OUTPUT_NODE = True

    def generate(self, main_prompt, server_url, model, max_tokens, temperature, seed, system_prompt=None):
        # --- Add a check to prevent running if models are not ready ---
        if model in ["not-found-check-lm-studio-server", "checking..."]:
            error_message = f"ERROR: Model not available ('{model}'). Please start your LM Studio server, load a model, and right-click -> 'Refresh' on the ComfyUI browser page."
            print(f"LLM-Prompt-Builder: {error_message}")
            return (error_message,)
        
        print("LLM-Prompt-Builder: Initializing OpenAI client...")
        try:
            client = OpenAI(base_url=f"{server_url}/v1", api_key="lm-studio")
            
            messages = []
            if system_prompt and system_prompt.strip():
                messages.append({"role": "system", "content": system_prompt})
            
            messages.append({"role": "user", "content": main_prompt})
            
            print(f"LLM-Prompt-Builder: Sending request to {server_url} with model {model} and seed {seed}...")
            
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed,
                stream=False
            )
            
            response_text = completion.choices[0].message.content
            print("LLM-Prompt-Builder: Received response successfully.")
            
        except Exception as e:
            error_message = f"ERROR: Could not get response from LM Studio. Check console for details. Error: {e}"
            print(f"LLM-Prompt-Builder: {error_message}")
            return (error_message,)

        return (response_text,)


# --- Node Registration ---
NODE_CLASS_MAPPINGS = {
    "LLMPromptBuilder": LLMPromptBuilder
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LLMPromptBuilder": "LLM-Prompt-Builder"
}