import gymnasium as gym
import numpy as np
from gymnasium import spaces
from abc import ABC, abstractmethod

class BaseCPSEnv(gym.Env, ABC):
    """
    Abstract Base Class for L2M-AID CPS Environments.
    Ensures unified observation and action spaces for Policy Transfer.
    """
    def __init__(self):
        super().__init__()
        
        # Unified Observation Space: 6 Dimensions
        # [Node_0, Node_1, Node_2, Actuator_0, Actuator_1, Actuator_2]
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(6,), dtype=np.float32)
        
        # Unified Action Space: 7 Discrete Actions
        # 0: No-Op
        # 1/2: Actuator 0 (+/-)
        # 3/4: Actuator 1 (+/-)
        # 5/6: Actuator 2 (+/-)
        self.action_space = spaces.Discrete(7)
        
    @abstractmethod
    def reset(self, seed=None, options=None):
        pass
        
    @abstractmethod
    def step(self, action, attack_dict=None):
        """
        step must accept 'attack_dict' for Red Team adversarial injections.
        """
        pass
