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
        self.action_map = {
            1: {'P1': 0},       # Dos Pump 1
            2: {'MV1': 1},      # Overflow Tank 1
            3: {'MV1': 0},      # Starve Tank 2
            4: {'P2': 0, 'P1': 1} # Imbalance
        }
        
    def get_action(self, obs: np.ndarray):
        obs_t = torch.FloatTensor(obs)
        probs = self.policy(obs_t)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        # Map integer to override dict
        overrides = self.action_map.get(action.item(), {})
        
        return action.item(), log_prob, overrides

    def update(self, rewards, log_probs):
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
        # H(pi) = -sum(p * log(p))
        entropy = dist.entropy().mean()
        entropy_coef = 0.01 
        
        # Minimize Loss = -(Reward + Entropy)
        loss = torch.stack(policy_loss).sum() - (entropy_coef * entropy)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
