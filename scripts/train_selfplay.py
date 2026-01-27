import sys
import os
import torch
import numpy as np
import argparse
from tqdm import tqdm

import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(os.getcwd())

from src.envs.wrapper import GenericCPSEnv
from src.agents.blue.tactical import MAPPOAgent
from src.agents.red.learning import LearningRedAgent
from src.agents.blue.orchestrator import OrchestratorAgent
from src.modules.explanation import ExplainabilityAgent
import datetime

# Ensure directories exist
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("results", exist_ok=True)

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

def save_checkpoint(blue_agent, red_agent, episode):
    """Saves model weights."""
    path = f"checkpoints/checkpoint_ep_{episode}.pt"
    torch.save({
        'blue_actor': blue_agent.actor.state_dict(),
        'blue_critic': blue_agent.critic.state_dict(),
        'red_policy': red_agent.policy.state_dict()
    }, path)
    # print(f"Checkpoint saved: {path}")

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

def train(mode="adversarial", episodes=500, provider="mock", model_name="gpt-3.5-turbo"):
    env = GenericCPSEnv()
    
    # Obs Dim: 6 (Env) + 384 (LLM Context) = 390
    # State Dim (Central Critic): 390
    blue_agent = MAPPOAgent(obs_dim=390, state_dim=390, action_dim=7)
    
    # Red Agent: Obs=6 (Raw Env), Action=5 (No-op + 4 attacks)
    red_agent = LearningRedAgent(obs_dim=6, action_dim=5)
    
    orchestrator = OrchestratorAgent(provider=provider, model_name=model_name)
    explainer = ExplainabilityAgent(provider=provider, model_name=model_name)
    
    blue_mem = Memory()
    red_mem_logprobs = []
    red_mem_rewards = []
    
    metrics = {
        "episode": [],
        "avg_psi": [],
        "total_reward": [],
        "blue_loss": [],
        "red_loss": []
    }
    
    print(f"Starting Training in {mode} mode for {episodes} episodes...")
    
    for ep in tqdm(range(episodes)):
        env_obs, _ = env.reset()
        score = 0
        psi_accum = 0
        episode_trace = []
        
        # Episode Loop
        for t in range(200): # Max steps per episode
            # 1. Orchestrator: Generate Context & Safety Score
            # Speed optimization: Only run every 40 steps or use cached
            if t % 40 == 0:
                llm_context = orchestrator.act(env_obs)
                llm_safety_score = orchestrator.evaluate_safety(env_obs)
            # else use previous values (implied by python scoping in loop, but need init)
            
            if t == 0: # Init for first step
               if 'llm_context' not in locals():
                   llm_context = orchestrator.act(env_obs)
                   llm_safety_score = orchestrator.evaluate_safety(env_obs)
            
            # 2. Blue Agent: Perception & Action
            # Concatenate Env Obs + LLM Context
            full_obs = np.concatenate([env_obs, llm_context])
            
            action_blue, log_prob_blue = blue_agent.get_action(full_obs)
            
            # 3. Red Agent: Attack Selection (if Adversarial)
            if mode == "adversarial":
                action_red_idx, log_prob_red, red_overrides = red_agent.get_action(env_obs)
            else:
                red_overrides = {} # Scripted or No-op
                action_red_idx = 0
                log_prob_red = torch.tensor(0.0)
            
            # 4. Environment Step (With Adversarial Injection)
            next_env_obs, reward_blue, terminated, truncated, info = env.step(action_blue, attack_dict=red_overrides)
            
            # Trace: (Time, State, BlueAct, RedAct, PSI)
            episode_trace.append((t, env_obs[:3], action_blue, action_red_idx, info['psi']))
            
            if t % 50 == 0:
                print(f"  [Ep {ep} Step {t}/200] Safety Score: {llm_safety_score:.2f} | PSI: {info['psi']:.2f}")

            psi = info['psi']
            
            # 5. Reward Engineering
            # Blue wants High PSI. Red wants Low PSI.
            # Reward_Blue = Env_Reward + 0.5 * LLM_Safety_Score (Shaping)
            psi = info['psi']
            reward_blue += (0.5 * llm_safety_score)
            
            # Reward_Red = (1.0 - PSI)
            reward_red = (1.0 - psi) * 10.0 # Scale up
            
            score += reward_blue
            psi_accum += psi
            
            # 6. Store Memories
            blue_mem.obs.append(full_obs)
            blue_mem.states.append(full_obs) # approximating global state
            blue_mem.actions.append(action_blue)
            blue_mem.logprobs.append(log_prob_blue)
            blue_mem.rewards.append(reward_blue)
            
            if mode == "adversarial":
                red_mem_logprobs.append(log_prob_red)
                red_mem_rewards.append(reward_red)
            
            env_obs = next_env_obs
            
            if terminated or truncated:
                break
                
        # Update Agents
        loss_blue = blue_agent.update(blue_mem)
        blue_mem.clear()
        
        loss_red = 0
        if mode == "adversarial":
            loss_red = red_agent.update(red_mem_rewards, red_mem_logprobs)
            red_mem_logprobs = []
            red_mem_rewards = []
            
        # Metrics & Logging
        avg_psi = psi_accum / 200.0 # approx
        metrics["episode"].append(ep)
        metrics["avg_psi"].append(avg_psi)
        metrics["total_reward"].append(score)
        metrics["blue_loss"].append(loss_blue)
        metrics["red_loss"].append(loss_red)
            
        if ep % 20 == 0:
            print(f"Ep {ep} | Avg PSI: {avg_psi:.3f} | Blue Loss: {loss_blue:.4f}")
            save_checkpoint(blue_agent, red_agent, ep)
            plot_training_results(metrics)
            
        if ep % 50 == 0:
            print(f"Generating Explanation for Ep {ep}...")
            explanation = explainer.explain_episode(episode_trace, metrics)
            with open("results/explanations.txt", "a") as f:
                f.write(f"\n--- Episode {ep} [{datetime.datetime.now()}] ---\n")
                f.write(explanation + "\n")
            
    # Final Plot
    save_checkpoint(blue_agent, red_agent, episodes)
    plot_training_results(metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="adversarial", choices=["curriculum", "adversarial"])
    parser.add_argument("--episodes", type=int, default=100)
    
    # New Args for LLM Provider
    parser.add_argument("--provider", type=str, default="mock", choices=["mock", "huggingface", "openai"])
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo")
    
    args = parser.parse_args()
    
    # Pass args to train (need to update train signature/logic next)
    train(mode=args.mode, episodes=args.episodes, provider=args.provider, model_name=args.model)
