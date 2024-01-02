import numpy as np

"""
This script holds the Inertia, damping and stiffness matrices that are used in the joint model differential equation
row/col positions:
1: SFE (Shoulder flexion/extension - shoulder y axis
2: SAA (Shoulder abduction/adduction - shoulder x axis
3: SEIR (Shoulder internal/external rotation - shoulder z axis)
4: EFE (Elbow flexion/extension - elbow y axis)
5: FPS (Forearm pronation/supination - elbow z axis)
6: WFE (Wrist flexion/extension)
7: WRUD (Wrist radial-ulnar deviation)
"""


def five_by_five():
    # Inertia matrix
    I = np.array([[0.269, 0, 0, 0.076, 0],
                  [0, 0.196, 0.083, 0, -0.002],
                  [0, 0.083, 0.079, 0, 0],
                  [0.076, 0, 0, 0.076, 0],
                  [0, -0.002, 0, 0, 0.002]])

    # Damping matrix
    D = np.array([[0.756, 0.184, 0.020, 0.187, 0],
                  [0.184, 0.383, 0.267, 0, 0],
                  [0.020, 0.267, 0.524, 0, 0],
                  [0.187, 0, 0, 0.607, 0],
                  [0, 0, 0, 0, 0.021]])

    # Stiffness matrix
    S = np.array([[10.80, 2.626, 0.279, 2.670, 0],
                  [2.626, 5.468, 3.821, 0, 0],
                  [0.279, 3.821, 7.486, 0, 0],
                  [2.670, 0, 0, 8.670, 0],
                  [0, 0, 0, 0, 0.756]])

    return I, D, S


def seven_by_seven():
    # Inertia matrix
    I = np.array([[0.269, 0, 0, 0.076, 0, 0, -0.014],
                  [0, 0.196, 0.083, 0, -0.002, 0.009, 0],
                  [0, 0.083, 0.079, 0, 0, 0.011, 0],
                  [0.076, 0, 0, 0.076, 0, 0, -0.012],
                  [0, -0.002, 0, 0, 0.002, 0, 0],
                  [0, 0.009, 0.011, 0, 0, 0.003, 0],
                  [-0.014, 0, 0, -0.012, 0, 0, 0.003]])

    # Damping matrix
    D = np.array([[0.756, 0.184, 0.020, 0.187, 0, 0, 0],
                  [0.184, 0.383, 0.267, 0, 0, 0, 0],
                  [0.020, 0.267, 0.524, 0, 0, 0, 0],
                  [0.187, 0, 0, 0.607, 0, 0, 0],
                  [0, 0, 0, 0, 0.021, 0.001, 0.008],
                 [0, 0, 0, 0, 0.001, 0.028, -0.003],
                 [0, 0, 0, 0, 0.008, -0.003, 0.082]])

    # Stiffness matrix
    S = np.array([[10.80, 2.626, 0.279, 2.670, 0, 0, 0],
                  [2.626, 5.468, 3.821, 0, 0, 0, 0],
                  [0.279, 3.821, 7.486, 0, 0, 0, 0],
                  [2.670, 0, 0, 8.670, 0, 0, 0],
                  [0, 0, 0, 0, 0.756, 0.018, 0.291],
                  [0, 0, 0, 0, 0.018, 0.992, -0.099],
                  [0, 0, 0, 0, 0.291, -0.099, 2.920]])

    return I, D, S
