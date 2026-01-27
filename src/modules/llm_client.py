import os
from typing import Optional

class LLMClient:
    """
    Interface for Large Language Models.
    Supports:
    - Mock (for fast testing)
    - OpenAI (API or Local Servers like Ollama/LMStudio)
    - HuggingFace (Direct loading)
    """
    def __init__(self, provider="mock", model_name="gpt-3.5-turbo", api_base=None, api_key=None):
        self.provider = provider
        self.model_name = model_name
        self.client = None
        
        if provider == "openai":
            import openai
            self.client = openai.OpenAI(
                base_url=api_base, # e.g. "http://localhost:11434/v1"
                api_key=api_key or os.environ.get("OPENAI_API_KEY", "dummy")
            )
        elif self.provider == "huggingface":
            from transformers import pipeline, BitsAndBytesConfig
            import torch
            
            print(f"Loading HF Model: {model_name}...")
            
            # Colab Optimization
            quant_config = None
            try:
                import bitsandbytes
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True
                )
                print("Quantization: Enabled (4-bit)")
            except ImportError:
                print("Quantization: Disabled - bitsandbytes not found")
            
            kwargs = {
                "quantization_config": quant_config
            }
            if not quant_config:
                 kwargs["torch_dtype"] = torch.float16
                 
            self.client = pipeline(
                "text-generation", 
                model=model_name, 
                device_map="auto",
                trust_remote_code=False,
                model_kwargs=kwargs
            )
            
    def generate(self, prompt: str) -> str:
        if self.provider == "mock":
            return self._mock_response(prompt)
        
        elif self.provider == "openai":
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"LLM Error: {e}")
                return "Error generating response."
                
        elif self.provider == "huggingface":
            # Simple generation
            output = self.client(prompt, max_new_tokens=100)
            return output[0]['generated_text']
            
    def _mock_response(self, prompt: str) -> str:
        """Heuristic-based mock responses for testing flow."""
        # 1. Handle Safety Rating Prompts
        if "Rate the SAFETY" in prompt:
            # If "stopped" or "Anomalous" in prompt implicitly (via logs), return low score
            if "P1" in prompt or "P2" in prompt: # simplified trigger
                 return "0.4"
            return "0.9"

        # 2. Handle Context/Analysis Prompts
        if "P1" in prompt and "stopped" in prompt:
            return "ANALYSIS: Denial of Service on Pump 1. INTENT: Process disruption."
        elif "MV1" in prompt:
            return "ANALYSIS: Anomalous valve opening. INTENT: Tank overflow."
        return "ANALYSIS: System Normal."
