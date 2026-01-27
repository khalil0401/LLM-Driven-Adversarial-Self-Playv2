import sys
import os
import torch
import numpy as np
import argparse
from tqdm import tqdm

sys.path.append(os.getcwd())

from src.envs.wrapper import L2MAIDEnv
from src.agents.blue.tactical import MAPPOAgent
from src.agents.blue.orchestrator import OrchestratorAgent
from src.agents.red.learning import LearningRedAgent

def evaluate(checkpoint_path=None, episodes=10):
    env = L2MAIDEnv()
    
    # Initialize Agents (Ideally load from checkpoint)
    blue_agent = MAPPOAgent(obs_dim=390, state_dim=390, action_dim=7)
    orchestrator = OrchestratorAgent(provider="mock")
    # Red agent is present but maybe we want to test against ScriptedRedAgent for standardized benchmarking
    # Or test against the learned one.
    
    # For robust eval, we should test against Scripted (Standard) AND Learned (Adaptive)
    from src.agents.red.scripted import ScriptedRedAgent
    red_agent_scripted = ScriptedRedAgent("attack_1")
    
    metrics = {
        "psi": [],
        "detection_rate": [], # We need to define what counts as 'detection'
        "fpr": []
    }
    
    print(f"Starting Evaluation for {episodes} episodes...")
    
    for ep in tqdm(range(episodes)):
        obs, _ = env.reset()
        episode_psi = []
        
        # We need to track if attack was present and if blue took mitigation
        attack_active_steps = 0
        mitigation_active_steps = 0
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        
        for t in range(200):
            # 1. Orchestrate
            ctx = orchestrator.act([obs])
            full_obs = np.concatenate([obs, ctx])
            
            # 2. Blue Action
            action_blue_idx, _ = blue_agent.get_action(full_obs)
            
            # 3. Red Action (Scripted for Standard Benchmark)
            # Attack 1 happens t=50 to 100
            red_overrides = red_agent_scripted.get_action(t, [])
            is_attack = (t >= 50 and t <= 100)
            
            # Apply Red
            if red_overrides:
                 for k, v in red_overrides.items():
                    if k == 'P1': env.physics.state.pump_states[0] = v
            
            # 4. Step
            next_obs, reward, term, trunc, info = env.step(action_blue_idx)
            episode_psi.append(info['psi'])
            
            # 5. Calculate Classification Metrics (Heuristic)
            # Action > 0 implies "Mitigation/Alert". Action 0 = No-Op.
            is_mitigation = (action_blue_idx > 0)
            
            if is_attack:
                if is_mitigation: tp += 1
                else: fn += 1
            else:
                if is_mitigation: fp += 1
                else: tn += 1
                
            obs = next_obs
            
        # Episode Metrics
        avg_psi = np.mean(episode_psi)
        metrics["psi"].append(avg_psi)
        
        # Avoid division by zero
        dr = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        metrics["detection_rate"].append(dr)
        metrics["fpr"].append(fpr)
        
    print("\n=== Evaluation Results ===")
    print(f"Mean PSI: {np.mean(metrics['psi']):.4f} (Higher is Better)")
    print(f"Detection Rate: {np.mean(metrics['detection_rate'])*100:.2f}%")
    print(f"False Positive Rate: {np.mean(metrics['fpr'])*100:.2f}%")

if __name__ == "__main__":
    evaluate()
