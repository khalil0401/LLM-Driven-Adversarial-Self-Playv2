from src.modules.llm_client import LLMClient
import numpy as np

class ExplainabilityAgent:
    """
    Extension C: Generative Explainability.
    Post-hoc analysis of episode trajectories to explain WHY agents acted.
    """
    def __init__(self, provider="mock"):
        self.llm = LLMClient(provider=provider)
        
    def explain_episode(self, episode_trace: list, metrics: dict) -> str:
        """
        trace: List of (Time, State, Action_Blue, Action_Red, PSI)
        """
        # 1. Identify Critical Events (Low PSI or High Reward Drops)
        critical_moments = []
        for step in episode_trace:
            t, s, a_blue, a_red, psi = step
            if psi < 0.02 or a_blue != 0: # Low stability or Action taken
                critical_moments.append(step)
        
        if not critical_moments:
            return "Episode Summary: Uneventful. System remained stable."
            
        # 2. Construct Prompt
        prompt = "You are a Cyber-Physical Security Analyst. Explain the following sequence:\n"
        for cm in critical_moments[-10:]: # Context window limit
            t, s, a, red, psi = cm
            prompt += f"- T={t}: PSI={psi:.3f}. Blue Action={a}. Potential Attack={red}.\n"
            
        prompt += "\nProvide a brief ROOT CAUSE ANALYSIS and JUSTIFICATION for Blue Team actions."
        
        # 3. Generate
        explanation = self.llm.generate(prompt)
        return explanation
