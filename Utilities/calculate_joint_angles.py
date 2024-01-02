import numpy as np
from scipy.integrate import solve_ivp


def solve_diff_eq(initial_cond, t_span, I, D, K, T):
    # define the diff eq func
    def dqdt(t, y):
        q = y[0:7]
        qdot = y[7:14]  # Extract q'
        qddot = np.linalg.inv(I) @ (T - D @ qdot - K @ q)
        return np.concatenate((qdot, qddot))

    sol = solve_ivp(dqdt, t_span, initial_cond)

    # Extract joint angle solutions
    joint_angle_array = sol.y[0:7, -1]

    return joint_angle_array

