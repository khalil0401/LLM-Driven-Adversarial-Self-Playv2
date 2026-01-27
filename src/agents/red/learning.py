import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical

class RedPolicyNetwork(nn.Module):
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

class LearningRedAgent:
    """
    Adversarial Agent (Extension A).
    Learns to select the optimal Attack ID to minimize PSI (Process Stability).
    """
    def __init__(self, obs_dim, action_dim, lr=3e-4):
        self.policy = RedPolicyNetwork(obs_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = 0.99
        self.obs_dim = obs_dim
        
        # Action space: 0=No-Op, 1=Stop P1, 2=Open MV1, 3=Close MV1, etc.
        # 0=No-Op
        # 1=Actuator_0 Off, 2=Actuator_2 Max, etc.
        self.action_map = {
            1: {'Actuator_0': 0.0},       # Dos Node 1 (Stop Input)
            2: {'Actuator_2': 1.0},       # Max Node 2 (Overflow)
            3: {'Actuator_2': 0.0},       # Min Node 2 (Starve)
            4: {'Actuator_1': 0.0, 'Actuator_0': 1.0} # Imbalance
        }
        
    def get_action(self, obs: np.ndarray):
        obs_t = torch.FloatTensor(obs)
        probs = self.policy(obs_t)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        # Map integer to override dict
        overrides = self.action_map.get(action.item(), {})
        
        return action.item(), log_prob, entropy, overrides

    def update(self, rewards, log_probs, entropies):
        """
        Simple REINFORCE update for the Red Agent.
        Maximize Reward = (1.0 - PSI) [Damage]
        """
        R = 0
        policy_loss = []
        returns = []
        
        for r in rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
            
        returns = torch.tensor(returns)
        # Normalize
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        for log_prob, R in zip(log_probs, returns):
            policy_loss.append(-log_prob * R)
            
        self.optimizer.zero_grad()
        
        # Entropy Regularization (IEEE Rigor: Prevent policy collapse)
        entropy_mean = torch.stack(entropies).mean()
        entropy_coef = 0.01 
        
        # Minimize Loss = -(Reward + Entropy)
        loss = torch.stack(policy_loss).sum() - (entropy_coef * entropy_mean)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
