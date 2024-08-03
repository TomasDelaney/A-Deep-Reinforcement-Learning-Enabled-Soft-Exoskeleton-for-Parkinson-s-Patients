import numpy as np


def distance_3d(point1, point2):
    return np.linalg.norm(point2 - point1)


def homogeneous_transformation_matrix(alpha, a, d, theta):
    # Homogeneous Transformation Matrix
    return np.array([
        [np.cos(theta), -np.sin(theta) * np.cos(alpha), np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
        [np.sin(theta), np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
        [0, np.sin(alpha), np.cos(alpha), d],
        [0, 0, 0, 1]
    ])


def forward_kinematics(theta_values, L1, L2, L3):
    # Denavit-Hartenberg Parameters
    # L1: length between shoulder and elbow joints aka humerus length
    # L2: length between elbow and wrist joint aka forearm length
    # L3: length between wrist joint and end effector point aka hand length

    # theta 1: SAA (Shoulder abduction/adduction - shoulder x-axis
    # theta 2: SFE (Shoulder flexion/extension - shoulder y-axis
    # theta 3: SEIR (Shoulder internal/external rotation - shoulder z axis)
    # theta 4: EFE (Elbow flexion/extension - elbow y axis)
    # theta 5: FPS (Forearm pronation/supination - elbow z axis)
    # theta 6: WFE (Wrist flexion/extension)
    # theta 7: WRUD (Wrist radial-ulnar deviation)

    # Joint angles
    theta1, theta2, theta3, theta4, theta5, theta6, theta7 = theta_values

    # Homogeneous Transformation Matrices (alpha, a, d, theta)
    A1 = homogeneous_transformation_matrix(np.pi/2, 0, 0, theta1)
    A2 = homogeneous_transformation_matrix(np.pi/2, 0, 0,  theta2)
    A3 = homogeneous_transformation_matrix(-np.pi/2, 0, L1, theta3)
    A4 = homogeneous_transformation_matrix(np.pi/2, 0, 0, theta4)
    A5 = homogeneous_transformation_matrix(np.pi/2, 0, L2, theta5)
    A6 = homogeneous_transformation_matrix(np.pi/2, 0, 0, theta6)
    A7 = homogeneous_transformation_matrix(np.pi/2, 0, 0, theta7)

    # End Effector Transformation
    T_end = A1 @ A2 @ A3 @ A4 @ A5 @ A6 @ A7

    # Extracting position from the transformation matrix
    end_effector_position = T_end[:3, 3]

    return end_effector_position


if __name__ == "__main__":
    # Example joint angles (replace with actual angles)array
    imu_joint_angles = np.radians([0, 0, 0, 0, 0, 0, 0])
    imu_joint_angles2 = np.radians([90, 0, 0, 0, 0, 0, 0])
    imu_joint_angles3 = np.radians([0, 90, 0, 0, 0, 0, 0])
    imu_joint_angles4 = np.radians([0, 0, 90, 0, 0, 0, 0])
    imu_joint_angles5 = np.radians([0, 0, 0, 90, 0, 0, 0])
    joint_angles = np.radians([0, 0, 0, 0, 0, 0, 0])
    joint_angles2 = np.radians([4, 5, 6, 0, 0, 0, 0])

    # arm lengths
    humerus_length = 0.4
    forearm_length = 0.4
    hand_length = 0.05

    # Calculate end effector position
    end_effector_position_joint = forward_kinematics(imu_joint_angles, humerus_length, forearm_length, hand_length)
    end_effector_position_joint2 = forward_kinematics(imu_joint_angles2, humerus_length, forearm_length, hand_length)
    end_effector_position_joint3 = forward_kinematics(imu_joint_angles3, humerus_length, forearm_length, hand_length)
    end_effector_position_joint4 = forward_kinematics(imu_joint_angles4, humerus_length, forearm_length, hand_length)
    end_effector_position_joint5 = forward_kinematics(imu_joint_angles5, humerus_length, forearm_length, hand_length)
    end_effector_position1 = forward_kinematics(joint_angles, humerus_length, forearm_length, hand_length)
    end_effector_position2 = forward_kinematics(joint_angles2, humerus_length, forearm_length, hand_length)
    diff_pos = end_effector_position1 - end_effector_position2

    distance_suppressed = distance_3d(end_effector_position_joint, end_effector_position1)  # Distance from the origin
    distance_unsuppressed = distance_3d(end_effector_position_joint, end_effector_position2)  # Distance from the origin

    # Calculate amplitude difference in percentage
    amplitude_difference_percentage = ((distance_suppressed - distance_unsuppressed) / distance_unsuppressed) * 100

    print("end effector pos: rest pos", end_effector_position_joint)
    print("end effector pos: shoulder add 90", end_effector_position_joint2)
    print("end effector pos: shoulder flex 90", end_effector_position_joint3)
    print("end effector pos: shoulder int 90", end_effector_position_joint4)
    print("end effector pos: elbow 90", end_effector_position_joint5)
    print()
    print("Distance Suppressed:", distance_suppressed)
    print("Distance Unsuppressed:", distance_unsuppressed)
    print("Amplitude Difference Percentage:", amplitude_difference_percentage, "%")
