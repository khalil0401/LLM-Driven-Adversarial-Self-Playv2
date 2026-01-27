import sys
import os
import torch
import numpy as np
import argparse
from tqdm import tqdm

sys.path.append(os.getcwd())

from src.envs.wrapper import GenericCPSEnv
from src.agents.blue.tactical import MAPPOAgent
from src.agents.blue.orchestrator import OrchestratorAgent
from src.agents.red.learning import LearningRedAgent

def evaluate(checkpoint_path, episodes=20, provider="mock", model="gpt-3.5-turbo"):
    print(f"Loading environment and agents...")
    env = GenericCPSEnv()
    
    # Initialize Blue Agent
    blue_agent = MAPPOAgent(obs_dim=390, state_dim=390, action_dim=7)
    
    # Initialize Orchestrator
    orchestrator = OrchestratorAgent(provider=provider, model_name=model)
    
    # Initialize Red Agent (Adversarial) - We evaluate against the Learned Attacker to see if Blue can beat it
    red_agent = LearningRedAgent(obs_dim=6, action_dim=5)
    
    # Load Checkpoint
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        blue_agent.actor.load_state_dict(checkpoint['blue_actor'])
        blue_agent.critic.load_state_dict(checkpoint['blue_critic'])
        red_agent.policy.load_state_dict(checkpoint['red_policy'])
    else:
        print("WARNING: No checkpoint found or provided. Evaluating UNTRAINED agents.")
    
    metrics = {
        "psi": [],
        "reward": []
    }
    
    print(f"Starting Evaluation for {episodes} episodes...")
    
    for ep in tqdm(range(episodes)):
        env_obs, _ = env.reset()
        episode_psi = []
        episode_reward = 0
        
        # Init Orchestrator Context
        llm_context = orchestrator.act(env_obs)
        
        for t in range(200):
            # 1. Update Context every 50 steps
            if t % 50 == 0:
                llm_context = orchestrator.act(env_obs)
                
            full_obs = np.concatenate([env_obs, llm_context])
            
            # 2. Blue Action (Deterministic for eval recommended, but using stochastic for now as implemented)
            # ideally blue_agent.get_action(deterministic=True)
            action_blue, _ = blue_agent.get_action(full_obs)
            
            # 3. Red Action (Adversarial)
            action_red_idx, _, _, red_overrides = red_agent.get_action(env_obs)
            
            # 4. Step
            next_env_obs, reward, terminated, truncated, info = env.step(action_blue, attack_dict=red_overrides)
            
            episode_psi.append(info['psi'])
            episode_reward += reward
            env_obs = next_env_obs
            
            if terminated or truncated:
                break
                
        metrics["psi"].append(np.mean(episode_psi))
        metrics["reward"].append(episode_reward)
        
    print("\n=== Final Evaluation Results ===")
    print(f"Average System Stability (PSI): {np.mean(metrics['psi']):.4f} (Target > 0.8)")
    print(f"Average Episode Reward: {np.mean(metrics['reward']):.2f}")
    print(f"Win Rate (PSI > 0.5): {np.mean(np.array(metrics['psi']) > 0.5) * 100:.1f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint.pt")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--provider", type=str, default="mock")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo")
    
    args = parser.parse_args()
    
    evaluate(args.checkpoint, args.episodes, args.provider, args.model)
