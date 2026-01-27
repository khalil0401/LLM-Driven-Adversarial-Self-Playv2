import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical

class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x):
        return self.net(x)

class CentralizedCritic(nn.Module):
    def __init__(self, global_state_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) # Value scalar
        )
        
    def forward(self, state):
        return self.net(state)

class MAPPOAgent:
    """
    Blue Team Tactical Agent.
    Uses PPO with Centralized Critic (state) and Decentralized Actor (observation).
    """
    def __init__(self, obs_dim, state_dim, action_dim, lr=1e-3):
        self.actor = Actor(obs_dim, action_dim)
        self.critic = CentralizedCritic(state_dim) # Global state
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.eps_clip = 0.2
        self.gamma = 0.99
        self.mse_loss = nn.MSELoss()

    def get_action(self, obs):
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs)
            probs = self.actor(obs_t)
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.item(), log_prob

    def evaluate(self, obs, state, action):
        # 1. Actor evaluation
        probs = self.actor(obs)
        dist = Categorical(probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        
        # 2. Critic evaluation (Centralized)
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy

    def update(self, memory):
        # Convert memory to tensors
        old_states = torch.FloatTensor(np.array(memory.states))
        old_obs = torch.FloatTensor(np.array(memory.obs))
        old_actions = torch.LongTensor(np.array(memory.actions))
        old_logprobs = torch.FloatTensor(np.array(memory.logprobs))
        rewards = torch.FloatTensor(np.array(memory.rewards))
        
        # Monte Carlo estimate of state rewards
        with torch.no_grad():
            # Simply use rewards for now, GAE is better but this is MVP
            returns = []
            discounted_sum = 0
            for reward in reversed(rewards):
                discounted_sum = reward + self.gamma * discounted_sum
                returns.insert(0, discounted_sum)
            returns = torch.tensor(returns)
            # Normalize
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        # Optimize for K epochs
        for _ in range(4):
            # Evaluate old actions and values
            logprobs, state_values, dist_entropy = self.evaluate(old_obs, old_states, old_actions)
            
            # Match tensor shapes
            state_values = torch.squeeze(state_values)
            
            # Ratio
            ratios = torch.exp(logprobs - old_logprobs.detach())
            
            # Surrogate Loss
            advantages = returns - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            
            loss = -torch.min(surr1, surr2) + 0.5 * self.mse_loss(state_values, returns) - 0.01 * dist_entropy
            
            # Take gradient step
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            loss.mean().backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
            
        return loss.mean().item()
