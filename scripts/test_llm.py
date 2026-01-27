import sys
import os
import numpy as np
sys.path.append(os.getcwd())

from src.agents.blue.orchestrator import OrchestratorAgent

def main():
    print("Initializing Orchestrator with SentenceTransformer...")
    # Wrap dim is ignored in current impl, returns 384
    agent = OrchestratorAgent(provider="mock")
    
    dummy_logs = ["[Event] Pump 1 stopped unexpectedly."]
    print(f"Testing act() with logs: {dummy_logs}")
    
    embedding = agent.act(dummy_logs)
    
    print(f"Embedding Shape: {embedding.shape}")
    print(f"Embedding Mean: {np.mean(embedding):.4f}")
    
    if embedding.shape[0] == 384:
        print("SUCCESS: Embedding dimension is correct (384).")
    else:
        print(f"FAILURE: Expected 384, got {embedding.shape[0]}")

if __name__ == "__main__":
    main()
