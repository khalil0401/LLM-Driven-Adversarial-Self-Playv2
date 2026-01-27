import gymnasium as gym
import numpy as np
from gymnasium import spaces
from .swat_sim import WaterTreatmentPlant
from .network_sim import NetworkSimulator, Packet

class L2MAIDEnv(gym.Env):
    """
    Main Cyber-Physical Environment.
    Combines Physics (SWaT) and Network (Packets).
    """
    def __init__(self):
        super().__init__()
        self.physics = WaterTreatmentPlant(dt=1.0)
        self.network = NetworkSimulator()
        
        # Action Space: 
        # 0: No-Op
        # 1: P1 On, 2: P1 Off
        # 3: P2 On, 4: P2 Off
        # 5: MV1 Open, 6: MV1 Close
        self.action_space = spaces.Discrete(7)
        
        # Observation Space:
        # [Tank1, Tank2, Tank3, Valve1, Pump1, Pump2] (Normalized)
        self.observation_space = spaces.Box(low=0, high=1, shape=(6,), dtype=np.float32)
        
        self.step_count = 0
        self.max_steps = 1000

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.physics = WaterTreatmentPlant(dt=1.0)
        self.step_count = 0
        return self._get_obs(), {}

    def step(self, action):
        # 1. Decode Action (Blue Team)
        control_actions = {}
        if action == 1: control_actions['P1'] = 1
        elif action == 2: control_actions['P1'] = 0
        elif action == 3: control_actions['P2'] = 1
        elif action == 4: control_actions['P2'] = 0
        elif action == 5: control_actions['MV1'] = 1
        elif action == 6: control_actions['MV1'] = 0
        
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
        
        # Check Safety Violation (Tank Overflow/Underflow)
        for i, lvl in enumerate(state.tank_levels):
            if lvl < 100 or lvl > 900:
                reward -= 100 # Safety penalty
                terminated = True
                
        self.step_count += 1
        
        return self._get_obs(), reward, terminated, truncated, {"psi": psi}

    def _get_obs(self):
        # Normalize Tank Levels (0-1000 -> 0-1)
        levels = self.physics.state.tank_levels / 1000.0
        # Valves/Pumps are 0/1
        valves = self.physics.state.valve_states[:1] # Keep it simple
        pumps = self.physics.state.pump_states
        
        return np.concatenate([levels, valves, pumps], dtype=np.float32)
