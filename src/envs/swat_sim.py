import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class PhysicalSystemState:
    tank_levels: np.ndarray  # [Tank1, Tank2, Tank3]
    valve_states: np.ndarray # [Valve1, Valve2, Valve3] (0=Closed, 1=Open)
    pump_states: np.ndarray  # [Pump1, Pump2] (0=Off, 1=On)
    flow_rates: np.ndarray   # calculated flow rates

class WaterTreatmentPlant:
    """
    Simulates a simplified 3-stage SWaT process:
    Stage 1: Raw Water Intake (Tank 1, Pump 1)
    Stage 2: Filtration (Tank 2, Valve 2)
    Stage 3: Distribution (Tank 3, Pump 2)
    
    Physics:
    dh/dt = (Qin - Qout) / Area
    """
    def __init__(self, dt: float = 1.0):
        self.dt = dt
        self.num_tanks = 3
        self.max_levels = np.array([1000.0, 1000.0, 1000.0]) # mm
        self.safe_ranges = [(400, 800), (400, 800), (400, 800)]
        self.tank_areas = np.array([5.0, 4.0, 5.0]) # m^2
        
        # Initial State
        self.state = PhysicalSystemState(
            tank_levels=np.array([500.0, 500.0, 500.0]),
            valve_states=np.array([1, 1, 1]),
            pump_states=np.array([1, 0]),
            flow_rates=np.zeros(4)
        )
        self.time = 0.0

    def step(self, action_dict: Dict[str, int]) -> PhysicalSystemState:
        """
        Apply control actions and advance physics.
        action_dict: {'P1': 1, 'MV2': 0, ...}
        """
        # 1. Update Actuators based on Actions (Discrete)
        if 'P1' in action_dict: self.state.pump_states[0] = action_dict['P1']
        if 'P2' in action_dict: self.state.pump_states[1] = action_dict['P2']
        if 'MV1' in action_dict: self.state.valve_states[0] = action_dict['MV1']
        
        # 2. Calculate Flows (Simplified Bernouilli/Linear)
        # Flow 1 (Inlet -> T1) depends on Source Pump
        q_in_1 = 50.0 * self.state.pump_states[0] 
        
        # Flow 1-2 (T1 -> T2) depends on gravity + Valve 1
        head_diff_12 = max(0, self.state.tank_levels[0] - self.state.tank_levels[1])
        q_12 = 10.0 * self.state.valve_states[0] * np.sqrt(head_diff_12 + 1e-3)
        
        # Flow 2-3 (T2 -> T3) - Filtration Pump logic (implicit in gravity for this simplified model)
        head_diff_23 = max(0, self.state.tank_levels[1] - self.state.tank_levels[2])
        q_23 = 8.0 * self.state.valve_states[1] * np.sqrt(head_diff_23 + 1e-3)
        
        # Flow 3-Out (T3 -> Distribution) depends on Output Pump
        q_out_3 = 45.0 * self.state.pump_states[1]

        # 3. Update Levels (Euler Integration)
        # T1
        d_h1 = (q_in_1 - q_12) / self.tank_areas[0] * self.dt
        # T2
        d_h2 = (q_12 - q_23) / self.tank_areas[1] * self.dt
        # T3
        d_h3 = (q_23 - q_out_3) / self.tank_areas[2] * self.dt
        
        self.state.tank_levels += np.array([d_h1, d_h2, d_h3])
        self.state.tank_levels = np.clip(self.state.tank_levels, 0, self.max_levels)
        
        self.state.flow_rates = np.array([q_in_1, q_12, q_23, q_out_3])
        self.time += self.dt
        
        return self.state

    def get_psi_metric(self) -> float:
        """Calculate Process Stability Index inverse error"""
        error_sum = 0
        for i in range(self.num_tanks):
            target = (self.safe_ranges[i][0] + self.safe_ranges[i][1]) / 2
            error_sum += (self.state.tank_levels[i] - target)**2
        
        rmse = np.sqrt(error_sum / self.num_tanks)
        return 1.0 / (rmse + 1e-6) # Higher is better (less error)
