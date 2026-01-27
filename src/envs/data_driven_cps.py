import gymnasium as gym
import numpy as np
from gymnasium import spaces
from src.data_loader import TONIoTLoader

class DataDrivenCPSEnv(gym.Env):
    """
    Data-Driven CPS Environment (TON_IoT).
    Replays real dataset sequences + Supports Red/Blue adversarial modifications.
    """
    def __init__(self, dataset_path=None):
        super().__init__()
        self.loader = TONIoTLoader(filepath=dataset_path)
        
        # State: 3 Nodes (from loader data) + 3 Control States (Virtual)
        self.observation_space = spaces.Box(low=0, high=1, shape=(6,), dtype=np.float32)
        
        # Actions: Same 7 dims for compatibility
        self.action_space = spaces.Discrete(7)
        
        self.current_episode_data = None
        self.current_episode_labels = None
        self.step_idx = 0
        self.max_steps = 200
        
        # Virtual Actuator States (Blue Team Control)
        self.virtual_controls = np.zeros(3) 

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Load a random chunk
        ep_idx = np.random.randint(0, 100)
        self.current_episode_data, self.current_episode_labels = self.loader.get_episode(ep_idx, self.max_steps)
        self.step_idx = 0
        self.virtual_controls = np.zeros(3)
        
        return self._get_obs(), {}

    def step(self, action, attack_dict=None):
        # 1. Base State from Dataset (Ground Truth)
        if self.step_idx >= len(self.current_episode_data):
            # End of data
            return self._get_obs(), 0, True, False, {"psi": 0.5}
            
        base_state = self.current_episode_data[self.step_idx].copy() # [N nodes]
        label = self.current_episode_labels[self.step_idx]
        
        # 2. Apply Blue Team Mitigation (Virtual Controls)
        # Action Map: 
        # 1/2 -> Control Node 0 (+/-)
        # 3/4 -> Control Node 1
        # 5/6 -> Control Node 2
        # Mitigation "fixes" deviations.
        
        if action == 1: self.virtual_controls[0] += 0.1
        elif action == 2: self.virtual_controls[0] -= 0.1
        elif action == 3: self.virtual_controls[1] += 0.1
        elif action == 4: self.virtual_controls[1] -= 0.1
        elif action == 5: self.virtual_controls[2] += 0.1
        elif action == 6: self.virtual_controls[2] -= 0.1
        
        # Decay controls (Mitigation is temporary logic fix)
        self.virtual_controls *= 0.9 
        
        # 3. Apply Red Team Attacks (Direct Feature Perturbation)
        # Attack Dict: {'Actuator_0': val} -> Mapped to Feature 0, 1, 2
        
        attack_vector = np.zeros(3)
        if attack_dict:
            for k, v in attack_dict.items():
                if k == 'Actuator_0': attack_vector[0] = v # Injection on Feat 0
                if k == 'Actuator_1': attack_vector[1] = v
                if k == 'Actuator_2': attack_vector[2] = v
        
        # 4. Final Observation Logic
        # Obs = Base + Attack + Control
        # Attack pushes away from Normal. Control pushes back?
        # Actually: Attack = Perturbation. Control = Correction.
        
        # Effective State = Data + Attack_Perturbation - Blue_Correction
        # Note: In TON_IoT, 'Data' might already be an attack if label=1.
        # But we assume the 'Data' is the baseline flow, and Red adds adversarial noise.
        
        # Simple Model:
        # Red adds noise to hide/create attacks.
        # Blue tries to detect/normalize.
        
        current_state = base_state + attack_vector - self.virtual_controls
        current_state = np.clip(current_state, 0.0, 1.0)
        
        # 5. Reward Calculation (Anomaly Detection / Stability)
        # PSI = 1 - Deviation from "Normal" (0.5 or smoothed baseline)
        # Here we define Normal as the 'Data' (assuming we want to match ground truth flow)
        # OR: We want stability (low variance).
        
        # Let's use: Target = 0.5 (Ideal stable state).
        # Dataset moves around 0.5.
        # Attack drags it to 0 or 1.
        
        error = np.mean(np.abs(current_state - 0.5))
        psi = 1.0 - (error / 0.5)
        psi = np.clip(psi, 0, 1)
        
        reward = psi # Blue wants to keep state close to 0.5 (Stable)
        
        self.step_idx += 1
        terminated = (self.step_idx >= self.max_steps)
        
        # Obs: [State_0, State_1, State_2, Ctl_0, Ctl_1, Ctl_2]
        obs = np.concatenate([current_state, self.virtual_controls], dtype=np.float32)
        
        return obs, reward, terminated, False, {"psi": psi}

    def _get_obs(self):
        # Initial peek
        if self.current_episode_data is None:
            return np.zeros(6, dtype=np.float32)
        
        s = self.current_episode_data[self.step_idx]
        return np.concatenate([s, self.virtual_controls], dtype=np.float32)
