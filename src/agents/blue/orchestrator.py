import numpy as np
from src.modules.llm_client import LLMClient
from sentence_transformers import SentenceTransformer

class OrchestratorAgent:
    """
    The 'Brain' of the Blue Team.
    1. Aggregates Logs
    2. Prompts LLM
    3. Generates Context Embedding (L_t) for RL Agents
    """
    def __init__(self, provider="mock", model_name="gpt-3.5-turbo", wrapping_dim=16):
        self.llm = LLMClient(provider=provider, model_name=model_name)
        # Use a small, fast model for embedding
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2') 
        self.raw_embedding_dim = 384
        
        # We project the 384-dim vector down to 'wrapping_dim' (e.g. 16) 
        # to keep RL input size manageable, or we assume the RL agent handles 384.
        # For this demo, let's keep the RL input small by doing a random projection or PCA.
        # Or, just return the full 384 and update the RL agent's input dim.
        self.embedding_dim = 384
        
    def act(self, obs_log: list) -> np.ndarray:
        """
        Produce the Context Embedding L_t
        """
        # 1. Summarize Logs
        summary_text = f"System Status: {obs_log[-1:]}. Assess threats."
        
        # 2. Get LLM Reasoning
        reasoning = self.llm.generate(summary_text)
        
        # 3. Convert to Vector
        embedding = self.embedder.encode(reasoning)
        
        return embedding.astype(np.float32)

    def evaluate_safety(self, obs_log: list) -> float:
        """
        Query LLM to rate system safety (0.0 to 1.0).
        Used for Reward Shaping.
        """
        prompt = f"""
        Analyze the following industrial control system logs for safety and stability.
        Logs: {obs_log[-5:]}
        
        Rate the SAFETY of the system on a scale from 0.0 (Critical Failure/Attack) to 1.0 (Normal Operation).
        Return ONLY the number.
        """

        
        response = self.llm.generate(prompt, max_new_tokens=10)
        
        try:
            # Simple parsing of the number
            score = float(response.strip().split()[0]) # naive parse
            return max(0.0, min(1.0, score))
        except:
            # Fallback if LLM output is messy
            return 0.5
