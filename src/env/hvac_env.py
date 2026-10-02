import gymnasium as gym
from gymnasium import spaces
import numpy as np

class BuildingThermalEnv(gym.Env):
    """
    A simplified CMDP Environment for a single-zone building HVAC system.
    State: [Indoor Temp, Outdoor Temp, Solar Radiance, PMV]
    Action: [HVAC Cooling/Heating Power]
    """
    def __init__(self):
        super().__init__()
        # Continuous action: HVAC power (-1 to 1, where -1 is max cooling, 1 is max heating)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # State: [T_in, T_out, PMV]
        self.observation_space = spaces.Box(low=-100, high=100, shape=(3,), dtype=np.float32)
        
        # ASHRAE 55 Parameters
        self.pmv_target = 0.0
        self.pmv_limit = 0.5
        
        # Building Thermal Dynamics (RC Model parameters)
        self.R = 0.5 # Thermal resistance
        self.C = 10.0 # Thermal capacitance
        self.dt = 1.0 # Time step
        
        self.state = None

    def _calculate_pmv(self, t_in):
        # Extremely simplified PMV surrogate: perfectly comfortable at 22C
        return (t_in - 22.0) / 3.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        t_in = 24.0 # Initial warm state
        t_out = 30.0 # Hot outside
        pmv = self._calculate_pmv(t_in)
        self.state = np.array([t_in, t_out, pmv], dtype=np.float32)
        return self.state, {}

    def step(self, action):
        t_in, t_out, _ = self.state
        power = action[0] * 20.0 # Scale action to kW
        
        # Simplified discrete-time RC thermal dynamics
        # T_in(k+1) = T_in(k) + dt/C * ((T_out(k) - T_in(k))/R + Power)
        t_in_next = t_in + (self.dt / self.C) * ((t_out - t_in) / self.R + power)
        
        pmv_next = self._calculate_pmv(t_in_next)
        
        # Reward: minimize energy usage (absolute power)
        reward = -abs(power)
        
        # Cost constraint (CMDP): 1 if PMV violates ASHRAE 55 bounds, 0 otherwise
        cost = 1.0 if abs(pmv_next) > self.pmv_limit else 0.0
        
        self.state = np.array([t_in_next, t_out, pmv_next], dtype=np.float32)
        
        terminated = False
        truncated = False
        info = {"cost": cost, "pmv": pmv_next}
        
        return self.state, reward, terminated, truncated, info
