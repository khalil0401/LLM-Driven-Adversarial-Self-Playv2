import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class CPSState:
    node_values: np.ndarray  # [N] Continuous state (e.g. load, pressure)
    actuator_states: np.ndarray # [M] Control inputs (0.0 - 1.0)
    
class GenericCPSSimulator:
    """
    A Generic Cyber-Physical System Simulator.
    Dynamics: Linear Time-Invariant (LTI) system with stable decay.
    dx/dt = -lambda * (x - target) + B * u
    """
    def __init__(self, num_nodes=3, num_actuators=3, dt=1.0):
        self.dt = dt
        self.num_nodes = num_nodes
        self.num_actuators = num_actuators
        
        # Stability Params
        self.decay = 0.1 # Natural stabilizing force
        self.coupling = np.array([ # Simple interaction matrix
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8]
        ])
        
        self.target_state = 0.5 # Ideal normalized state
        
        # Initial State
        self.state = CPSState(
            node_values=np.ones(num_nodes) * 0.5,
            actuator_states=np.zeros(num_actuators)
        )
        self.time = 0.0

    def step(self, control_signals: Dict[str, float]) -> CPSState:
        """
        control_signals: {'Actuator_0': 1.0, ...}
        """
        # 1. Update Actuators
        for k, v in control_signals.items():
            try:
                idx = int(k.split('_')[1])
                if 0 <= idx < self.num_actuators:
                    self.state.actuator_states[idx] = np.clip(v, 0.0, 1.0)
            except:
                pass
                
        # 2. Dynamics (Abstract Physics)
        # u = Effect of actuators
        # We assume Actuator i affects Node i roughly
        u = self.state.actuator_states[:self.num_nodes] 
        if len(u) < self.num_nodes:
            u = np.pad(u, (0, self.num_nodes - len(u)))
            
        # Error from target
        error = self.state.node_values - self.target_state
        
        # Restoration force (control pushes towards 0 error if tuned, or drives basic flow)
        # Here we model: Actuators ADD to the value. Decay removes it.
        # Like filling a tank with a hole.
        
        # dx = -decay * x + input
        # Target is to maintain x around 0.5.
        # If input is 0.5 and decay balances, stable.
        
        inputs = u * 0.2 # Gain
        
        # Diffusion/Coupling
        current_vals = self.state.node_values
        new_vals = np.dot(self.coupling, current_vals)
        
        # Apply change
        # If val > target, decay. If val < target, requires input.
        # Let's simple model:
        # x_new = x_old + dt * (-decay*x_old + input)
        
        d_x = -self.decay * current_vals + inputs
        
        self.state.node_values += d_x * self.dt
        self.state.node_values = np.clip(self.state.node_values, 0.0, 1.0)
        
        self.time += self.dt
        return self.state

    def get_psi_metric(self) -> float:
        """
        Process Stability Index.
        1.0 = Perfect (All nodes at Target).
        0.0 = Chaos (Nodes at bounds).
        """
        error = np.abs(self.state.node_values - self.target_state)
        # Normalize error. Max error is 0.5 (if target is 0.5 and val is 0 or 1)
        mean_error = np.mean(error)
        
        # psi = 1 - (error / max_possible_error)
        psi = 1.0 - (mean_error / 0.5)
        return max(0.0, min(1.0, psi))
