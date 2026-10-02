import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.env.hvac_env import BuildingThermalEnv
from src.cbf.barrier import ThermalCBF
import numpy as np

def run_simulation():
    env = BuildingThermalEnv()
    cbf = ThermalCBF()
    
    state, _ = env.reset()
    print("Starting Simulation...")
    print("Target PMV range: [-0.5, 0.5]")
    
    # Simulate an RL agent that maliciously or ignorantly tries to save energy 
    # by applying 0 power (turning off HVAC).
    # Since outdoor is 30C, building will heat up rapidly.
    
    for step in range(10):
        raw_action = np.array([0.0], dtype=np.float32) # "Turn off HVAC"
        
        # Apply Safety Filter
        safe_action = cbf.get_safe_action(state, raw_action)
        
        print(f"Step {step}: T_in={state[0]:.2f}C, PMV={state[2]:.2f} | RL Action: {raw_action[0]:.2f} -> CBF Action: {safe_action[0]:.2f}")
        
        state, reward, done, _, info = env.step(safe_action)
        
        if info["cost"] > 0:
            print(f"!!! SAFETY VIOLATION AT STEP {step} !!!")
            
    print("Simulation complete.")

if __name__ == "__main__":
    run_simulation()
