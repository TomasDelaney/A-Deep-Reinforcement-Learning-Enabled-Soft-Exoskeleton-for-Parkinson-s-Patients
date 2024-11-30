import numpy as np
from scipy.integrate import solve_ivp


def solve_diff_eq(initial_cond, t_span, I, D, K, T):
    # define the diff eq func
    def dqdt(t, y):
        q = y[0:7]  # Joint positions
        qdot = y[7:14]  # Joint velocities

        # Solve for qddot using np.linalg.solve to avoid matrix inversion
        qddot = np.linalg.solve(I, T - D @ qdot - K @ q)

        # Return the concatenated result of [qdot, qddot]
        return np.concatenate((qdot, qddot))

    sol = solve_ivp(dqdt, t_span, initial_cond)

    # Extract joint angle solutions
    joint_angle_array = sol.y[0:7, -1]

    return np.reshape(np.array(joint_angle_array), (7,))
