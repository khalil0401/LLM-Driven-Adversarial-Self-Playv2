import gymnasium as gym
import numpy as np
from gymnasium import spaces
from .generic_cps import GenericCPSSimulator
from .network_sim import NetworkSimulator, Packet

class GenericCPSEnv(gym.Env):
    """
    Generic Cyber-Physical Environment.
    Replaces L2MAIDEnv (SWaT) with abstract CPS dynamics.
    """
    def __init__(self):
        super().__init__()
        self.physics = GenericCPSSimulator(num_nodes=3, num_actuators=3)
        self.network = NetworkSimulator()
        
        # Action Space (Preserved for compatibility): 
        # 0: No-Op
        # 1: Actuator_0 + (Fill)
        # 2: Actuator_0 - (Drain)
        # ...
        self.action_space = spaces.Discrete(7)
        
        # Observation Space:
        # [Node1, Node2, Node3, Act1, Act2, Act3]
        self.observation_space = spaces.Box(low=0, high=1, shape=(6,), dtype=np.float32)
        
        self.step_count = 0
        self.max_steps = 1000

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.physics = GenericCPSSimulator(num_nodes=3, num_actuators=3)
        self.step_count = 0
        return self._get_obs(), {}

    def step(self, action, attack_dict=None):
        # 1. Decode Action (Blue Team)
        # Map discrete 0-6 to Actuator adjustments
        # We define simple logic: Set Actuator to High (0.8) or Low (0.2)
        control_actions = {}
        
        # Mapping for Back-Compat with existing Agents
        # P1 (Pump 1) -> Actuator_0
        # P2 (Pump 2) -> Actuator_1
        # MV1 (Valve 1) -> Actuator_2
        
        if action == 1: control_actions['Actuator_0'] = 0.8
        elif action == 2: control_actions['Actuator_0'] = 0.0
        elif action == 3: control_actions['Actuator_1'] = 0.8
        elif action == 4: control_actions['Actuator_1'] = 0.0
        elif action == 5: control_actions['Actuator_2'] = 0.8
        elif action == 6: control_actions['Actuator_2'] = 0.0
        
        # 1.5 Apply Red Team Overrides (Adversarial Supremacy)
        if attack_dict:
            # Map old keys if necessary or expect new keys
            # For robustness, we handle both if Red Agent isn't fully updated yet
            cleaned_attack = {}
            for k, v in attack_dict.items():
                if k == 'P1': k = 'Actuator_0'
                if k == 'P2': k = 'Actuator_1'
                if k == 'MV1': k = 'Actuator_2'
                cleaned_attack[k] = v
            control_actions.update(cleaned_attack)
        
        # 2. Step Physics
        state = self.physics.step(control_actions)
        
        # 3. Step Network (Process Traffic)
        packets = self.network.process_queue(self.step_count)
        
        # 4. Calculate Reward
        # Reward = w_proc * PSI
        psi = self.physics.get_psi_metric()
        reward = 2.0 * psi 
        
        # 5. Check Termination
        truncated = False
        terminated = False
        if self.step_count >= self.max_steps:
            truncated = True
        
        # Check Safety Violation (Nodes drifting too far)
        for i, val in enumerate(state.node_values):
            if val < 0.05 or val > 0.95:
                reward -= 50 # Safety penalty
                terminated = True
                
        self.step_count += 1
        
        return self._get_obs(), reward, terminated, truncated, {"psi": psi}

    def _get_obs(self):
        # State is already 0-1
        nodes = self.physics.state.node_values
        acts = self.physics.state.actuator_states
        return np.concatenate([nodes, acts], dtype=np.float32)
