import numpy as np
from scipy.optimize import minimize

class ThermalCBF:
    """
    Control Barrier Function (CBF) Safety Filter for ASHRAE 55 PMV bounds.
    Ensures the RL agent's action does not push the building state outside safe PMV limits.
    """
    def __init__(self, R=0.5, C=10.0, dt=1.0, pmv_limit=0.5):
        self.R = R
        self.C = C
        self.dt = dt
        self.pmv_limit = pmv_limit
        self.gamma = 0.5 # CBF relaxation parameter (0 < gamma <= 1)

    def h(self, t_in):
        # Barrier function: h(x) >= 0 is the safe set.
        # Safe set: abs((t_in - 22) / 3) <= 0.5  => 20.5 <= t_in <= 23.5
        # We split into two barriers: h_upper and h_lower
        h_upper = 23.5 - t_in
        h_lower = t_in - 20.5
        return h_upper, h_lower

    def get_safe_action(self, state, raw_action):
        t_in, t_out, _ = state
        u_rl = raw_action[0]
        
        # We want to solve a QP:
        # min_u || u - u_rl ||^2
        # s.t. h(f(x, u)) - h(x) >= -gamma * h(x)
        
        def objective(u):
            return (u[0] - u_rl)**2
        
        def cbf_constraint_upper(u):
            # T_next = T_in + dt/C * ((T_out - T_in)/R + u[0]*20.0)
            t_next = t_in + (self.dt / self.C) * ((t_out - t_in) / self.R + u[0]*20.0)
            h_next_upper = 23.5 - t_next
            h_curr_upper = 23.5 - t_in
            return h_next_upper - (1 - self.gamma) * h_curr_upper
            
        def cbf_constraint_lower(u):
            t_next = t_in + (self.dt / self.C) * ((t_out - t_in) / self.R + u[0]*20.0)
            h_next_lower = t_next - 20.5
            h_curr_lower = t_in - 20.5
            return h_next_lower - (1 - self.gamma) * h_curr_lower

        constraints = (
            {'type': 'ineq', 'fun': cbf_constraint_upper},
            {'type': 'ineq', 'fun': cbf_constraint_lower}
        )
        bounds = ((-1.0, 1.0),)
        
        res = minimize(objective, [u_rl], bounds=bounds, constraints=constraints)
        if res.success:
            return np.array([res.x[0]], dtype=np.float32)
        else:
            # Deterministic fall-safe: if infeasible, apply max cooling/heating to recover
            h_u, h_l = self.h(t_in)
            if h_u < 0: return np.array([-1.0], dtype=np.float32) # Too hot, max cool
            if h_l < 0: return np.array([1.0], dtype=np.float32)  # Too cold, max heat
            return raw_action # Should not happen
