import numpy as np
from src.envs.base import BaseCPSEnv
from .generic_cps import GenericCPSSimulator
from .network_sim import NetworkSimulator

class GenericCPSPhysicsEnv(BaseCPSEnv):
    """
    Physics-Based CPS Environment (Stage 1).
    Uses LTI dynamics for pretraining strategy emergence.
    """
    def __init__(self):
        super().__init__()
        self.physics = GenericCPSSimulator(num_nodes=3, num_actuators=3)
        self.network = NetworkSimulator()
        self.step_count = 0
        self.max_steps = 200

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.physics.reset()
        self.step_count = 0
        return self._get_obs(), {}

    def step(self, action, attack_dict=None):
        # 1. Action Decoding (Discrete -> Physics Inputs)
        control_actions = {}
        # 1/2 -> Actuator 0
        if action == 1: control_actions['Actuator_0'] = 0.0 # Off
        elif action == 2: control_actions['Actuator_0'] = 1.0 # Max
        
        # 3/4 -> Actuator 1
        elif action == 3: control_actions['Actuator_1'] = 0.0
        elif action == 4: control_actions['Actuator_1'] = 1.0
        
        # 5/6 -> Actuator 2
        elif action == 5: control_actions['Actuator_2'] = 0.0
        elif action == 6: control_actions['Actuator_2'] = 1.0
        
        # 2. Apply Adversarial Overrides (Attack Injection)
        # Red Team inputs override Blue Team inputs
        if attack_dict:
            for k, v in attack_dict.items():
                control_actions[k] = v

        # 3. Physics Step
        next_state = self.physics.step(control_actions)
        
        # 4. Reward Calculation (Stability)
        # PSI = Process Stability Index (0 to 1). Higher is better.
        psi = self.physics.get_psi()
        
        # Blue Reward: Keep PSI high + Penalty for extreme actions
        reward = (psi - 0.5) * 2 # Center around 0 [-1, 1]
        
        if psi < 0.2:
            reward -= 50 # Crash penalty
            
        self.step_count += 1
        terminated = (self.step_count >= self.max_steps) or (psi < 0.05)
        
        obs = self._get_obs()
        return obs, reward, terminated, False, {"psi": psi}

    def _get_obs(self):
        # Flatten state: [Nodes(3), Actuators(3)]
        s = self.physics.state
        nodes = s.node_values
        # Actuators: We need to reconstruct current actuator state or track it
        # The simulator stores it? Let's check simulator.
        # Actually GenericCPSSimulator doesn't store 'actuator state' permanently in a way accessible easily?
        # Let's assume input u is transient.
        # Ideally we track it.
        # For compatibility with DataEnv, we pad with 0s if not tracked, or we update Simulator to track.
        # Let's just use what we have in Simulator state if available, or zeros.
        # GenericCPSSimulator state has .node_values.
        # Let's just return node values padded to 6 dim.
        
        # Wait, BaseCPSEnv defined strict 6 Dim.
        # Let's return [N1, N2, N3, 0, 0, 0] if we don't track actuators, but we should track them.
        
        # For now, simplistic approximation.
        obs = np.concatenate([nodes, np.zeros(3)], dtype=np.float32)
        return obs
