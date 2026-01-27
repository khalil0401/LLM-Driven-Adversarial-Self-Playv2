import sys
import os
import torch
import numpy as np
import argparse
from tqdm import tqdm

import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(os.getcwd())

from src.envs.physics_env import GenericCPSPhysicsEnv
from src.envs.data_driven_cps import DataDrivenCPSEnv
from src.agents.blue.tactical import MAPPOAgent
from src.agents.red.learning import LearningRedAgent
from src.agents.blue.orchestrator import OrchestratorAgent
from src.modules.explanation import ExplainabilityAgent
import datetime

# Ensure directories exist
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("results/hybrid_experiment", exist_ok=True)

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # random.seed(seed) if imported

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.obs = []
        self.logprobs = []
        self.rewards = []
    
    def clear(self):
        self.actions = []
        self.states = []
        self.obs = []
        self.logprobs = []
        self.rewards = []

def save_checkpoint(blue_agent, red_agent, filename):
    """Saves model weights."""
    path = f"checkpoints/{filename}"
    torch.save({
        'blue_actor': blue_agent.actor.state_dict(),
        'blue_critic': blue_agent.critic.state_dict(),
        'red_policy': red_agent.policy.state_dict()
    }, path)
    print(f"Checkpoint saved: {path}")

def plot_training_results(metrics, filename="results/training_results.png"):
    """Plots training metrics."""
    df = pd.DataFrame(metrics)
    
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Avg PSI
    axs[0, 0].plot(df['episode'], df['avg_psi'], color='blue', label='PSI')
    axs[0, 0].set_title('Average System Stability (PSI)')
    axs[0, 0].set_ylim(0, 1.1)
    axs[0, 0].grid(True)
    
    # 2. Total Reward
    axs[0, 1].plot(df['episode'], df['total_reward'], color='green', label='Reward')
    axs[0, 1].set_title('Total Episode Reward')
    axs[0, 1].grid(True)
    
    # 3. Blue Loss
    axs[1, 0].plot(df['episode'], df['blue_loss'], color='orange', label='Blue Loss')
    axs[1, 0].set_title('Blue Agent Loss')
    axs[1, 0].grid(True)
    
    # 4. Red Loss
    if 'red_loss' in df.columns and not df['red_loss'].isna().all():
        axs[1, 1].plot(df['episode'], df['red_loss'], color='red', label='Red Loss')
        axs[1, 1].set_title('Red Agent Loss')
        axs[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Results plotted: {filename}")

def run_training_stage(env, blue_agent, red_agent, orchestrator, explainer, episodes, stage_name, mode="adversarial"):
    """Runs a single training stage (Physics or Data)."""
    print(f"\n=== Starting Stage: {stage_name} ({episodes} episodes) ===")
    
    blue_mem = Memory()
    red_mem_logprobs = []
    red_mem_rewards = []
    red_mem_entropies = []
    
    metrics = {
        "episode": [],
        "avg_psi": [],
        "total_reward": [],
        "blue_loss": [],
        "red_loss": []
    }
    
    for ep in tqdm(range(episodes)):
        env_obs, _ = env.reset()
        score = 0
        psi_accum = 0
        episode_trace = []
        
        # Episode Loop
        for t in range(200): # Max steps per episode
            # 1. Orchestrator
            if t % 40 == 0:
                llm_context = orchestrator.act(env_obs)
                llm_safety_score = orchestrator.evaluate_safety(env_obs)
            if t == 0 and 'llm_context' not in locals():
                llm_context = orchestrator.act(env_obs)
                llm_safety_score = orchestrator.evaluate_safety(env_obs)
            
            # 2. Blue Agent
            full_obs = np.concatenate([env_obs, llm_context])
            action_blue, log_prob_blue = blue_agent.get_action(full_obs)
            
            # 3. Red Agent
            if mode == "adversarial":
                action_red_idx, log_prob_red, entropy_red, red_overrides = red_agent.get_action(env_obs)
            else:
                red_overrides = {} 
                action_red_idx = 0
                log_prob_red = torch.tensor(0.0)
                entropy_red = torch.tensor(0.0)
            
            # 4. Step
            next_env_obs, reward_blue, terminated, truncated, info = env.step(action_blue, attack_dict=red_overrides)
            
            episode_trace.append((t, env_obs[:3], action_blue, action_red_idx, info['psi']))
            
            psi = info['psi']
            reward_blue += (0.5 * llm_safety_score)
            reward_red = (1.0 - psi) * 10.0
            
            score += reward_blue
            psi_accum += psi
            
            # Store
            blue_mem.obs.append(full_obs)
            blue_mem.states.append(full_obs)
            blue_mem.actions.append(action_blue)
            blue_mem.logprobs.append(log_prob_blue)
            blue_mem.rewards.append(reward_blue)
            
            if mode == "adversarial":
                red_mem_logprobs.append(log_prob_red)
                red_mem_rewards.append(reward_red)
                red_mem_entropies.append(entropy_red)
            
            env_obs = next_env_obs
            
            if terminated or truncated:
                break
                
        # Update
        loss_blue = blue_agent.update(blue_mem)
        blue_mem.clear()
        
        loss_red = 0
        if mode == "adversarial":
            loss_red = red_agent.update(red_mem_rewards, red_mem_logprobs, red_mem_entropies)
            red_mem_logprobs = []
            red_mem_rewards = []
            red_mem_entropies = []
            
        avg_psi = psi_accum / 200.0
        metrics["episode"].append(ep)
        metrics["avg_psi"].append(avg_psi)
        metrics["total_reward"].append(score)
        metrics["blue_loss"].append(loss_blue)
        metrics["red_loss"].append(loss_red)
            
        if ep % 20 == 0:
            print(f"[{stage_name}] Ep {ep} | Avg PSI: {avg_psi:.3f}")
            plot_training_results(metrics, filename=f"results/hybrid_experiment/results_{stage_name}.png")
            
        if ep % 50 == 0:
            explanation = explainer.explain_episode(episode_trace, metrics)
            with open(f"results/hybrid_experiment/explanations_{stage_name}.txt", "a") as f:
                f.write(f"\n--- Episode {ep} [{datetime.datetime.now()}] ---\n")
                f.write(explanation + "\n")
                
    return metrics

def train(mode="adversarial", episodes=500, provider="mock", model_name="gpt-3.5-turbo", dataset_path=None, hybrid=False):
    set_seed(42)  # IEEE Reproducibility
    
    # Initialize Agents (Shared across stages)
    # Obs Dim: 6 (Env) + 384 (LLM Context) = 390
    blue_agent = MAPPOAgent(obs_dim=390, state_dim=390, action_dim=7)
    red_agent = LearningRedAgent(obs_dim=6, action_dim=5)
    orchestrator = OrchestratorAgent(provider=provider, model_name=model_name)
    explainer = ExplainabilityAgent(provider=provider, model_name=model_name)
    
    if hybrid:
        # STAGE 1: Physics Pretraining
        print("\n>>> STAGE 1: PHYSICS-BASED PRETRAINING <<<")
        env_physics = GenericCPSPhysicsEnv()
        run_training_stage(env_physics, blue_agent, red_agent, orchestrator, explainer, 
                         episodes=episodes//2, stage_name="stage1_physics", mode=mode)
        
        save_checkpoint(blue_agent, red_agent, "checkpoint_stage1_physics.pt")
        
        # STAGE 2: Data-Driven Transfer
        print("\n>>> STAGE 2: DATA-DRIVEN TRANSFER (TON_IoT) <<<")
        env_data = DataDrivenCPSEnv(dataset_path=dataset_path)
        run_training_stage(env_data, blue_agent, red_agent, orchestrator, explainer, 
                         episodes=episodes//2, stage_name="stage2_data", mode=mode)
        
        save_checkpoint(blue_agent, red_agent, "checkpoint_stage2_data.pt")
        
    else:
        # Classical Single Mode
        if dataset_path:
            env = DataDrivenCPSEnv(dataset_path=dataset_path)
            name = "data_only"
        else:
            env = GenericCPSPhysicsEnv()
            name = "physics_only"
            
        run_training_stage(env, blue_agent, red_agent, orchestrator, explainer, 
                         episodes=episodes, stage_name=name, mode=mode)
        save_checkpoint(blue_agent, red_agent, f"checkpoint_{name}.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="adversarial", choices=["curriculum", "adversarial"])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--provider", type=str, default="mock", choices=["mock", "huggingface", "openai"])
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--dataset", type=str, default=None, help="Path to TON_IoT CSV")
    parser.add_argument("--hybrid", action="store_true", help="Enable Two-Stage Hybrid Training")
    
    args = parser.parse_args()
    
    train(mode=args.mode, episodes=args.episodes, provider=args.provider, 
          model_name=args.model, dataset_path=args.dataset, hybrid=args.hybrid)
