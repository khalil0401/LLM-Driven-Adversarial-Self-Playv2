import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.getcwd())

from src.envs.wrapper import L2MAIDEnv
from src.agents.red.scripted import ScriptedRedAgent

def main():
    print("Initializing L2M-AID Environment...")
    env = L2MAIDEnv()
    red_agent = ScriptedRedAgent(attack_id="attack_1") # P1 OFF at t=50-100
    
    obs, _ = env.reset()
    print("Initial Obs:", obs)
    
    psi_history = []
    
    print("\nStarting Simulation Loop (200 steps)...")
    for t in range(200):
        # 1. Blue Team Action (Random for now)
        # 0 = No-Op
        blue_action = 0 
        
        # 2. Red Team Overrides
        red_overrides = red_agent.get_action(t, [])
        if red_overrides:
            print(f"Step {t}: RED ATTACK! Overrides: {red_overrides}")
            # How to apply overrides in Gym? 
            # In a real MARL, Red is an agent.
            # Here, we hack it by passing it via 'info' or modifying the env directly.
            # Ideally, Env.step() should take a dict of actions.
            # For this test, let's manually apply it to the physics:
            for k, v in red_overrides.items():
                if k == 'P1': env.physics.state.pump_states[0] = v
        
        # 3. Environment Step
        obs, reward, terminated, truncated, info = env.step(blue_action)
        psi = info['psi']
        psi_history.append(psi)
        
        if t % 50 == 0:
            levels = env.physics.state.tank_levels
            print(f"Step {t} | Reward: {reward:.2f} | PSI: {psi:.2f} | Levels: {levels}")
            
    print("\nSimulation Complete.")
    avg_psi_normal = np.mean(psi_history[:40])
    avg_psi_attack = np.mean(psi_history[50:100])
    print(f"Avg PSI (Normal): {avg_psi_normal:.2f}")
    print(f"Avg PSI (Attack): {avg_psi_attack:.2f}")
    
    if avg_psi_attack < avg_psi_normal:
        print("SUCCESS: Attack successfully degraded process stability.")
    else:
        print("WARNING: Attack did not degrade stability significantly.")

if __name__ == "__main__":
    main()
