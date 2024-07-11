import numpy as np
import pybullet_utils.bullet_client as bc
import pybullet as p
import pybullet_data
import gym
from gym.utils import seeding
from gym import spaces
from Utilities.getinertia import calculate_moments_of_inertia, calculate_moments_of_inertia_hand
from Environment.Exoskeleton_sim_pybullet import ExoskeletonSimModel
from Utilities.read_txt_env import define_empty_dict_for_env, read_env_texts
from Utilities.calculate_body_part_mass_ import calculate_body_part_mass
from Utilities.generate_parkinson_tremor import generate_joint_torques_train
from Utilities.differential_eq_matrices import seven_by_seven
from Utilities.calculate_joint_angles import solve_diff_eq
import random

"""
-design consideration the exoskeleton cannot suppress elbow z values for tremors so why consider it
-initial vel and acc value setting for random initialization 0
- wrist joint ang_acc and ang:vel only depend from the tremor
"""


class ExoskeletonEnv_train(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, evaluation, hum_weight, hum_radius, hum_height, forearm_weight, forearm_radius, forearm_height, hand_weight, hand_radius, dummy_shift, use_all_dof,
                 reference_motion_file_num, tremor_sequence, max_force_shoulder, max_force_elbow, client):
        super(ExoskeletonEnv_train, self).__init__()
        # determine if we want to train an agent in the environment or evaluate it: boolean value
        self.dummy_shift = dummy_shift
        self.evaluate = evaluation  # true if we want to evaluate, false if we want to train
        self.file_num = reference_motion_file_num

        # the episode count in the texts and simulation time space
        self.counts = 1

        # define the time steps of the simulation
        self.dt = 1 / 40

        # set the tremor generation mode if all seven DoF-s are used o only the first 4
        self.use_all_dof = use_all_dof
        self.tremor_input_sequence = tremor_sequence

        # counter for which reference motion to load
        self.env_counter = 0

        self.processed_imu_data = None  # dictionary of numpy arrays, contains the target values
        self.max_count = None  # size of a numpy array
        self.ep_state_values = None
        self.tremor_torque_values = None
        self.tremor_torque_places = None  # todo: find on which joints actual tremors are generated
        self.tremor_ang_acc_values = None  # shoulder x,y,z elbow y,z wrist x z

        # human skeleton anatomical properties
        self.humerus_weight = hum_weight
        self.humerus_radius = hum_radius
        self.humerus_height = hum_height
        self.humerus_inertia_x = calculate_moments_of_inertia(hum_weight, hum_radius, hum_height)[0]
        self.humerus_inertia_y = calculate_moments_of_inertia(hum_weight, hum_radius, hum_height)[1]
        self.humerus_inertia_z = calculate_moments_of_inertia(hum_weight, hum_radius, hum_height)[2]

        self.forearm_weight = forearm_weight
        self.forearm_height = forearm_height
        self.forearm_radius = forearm_radius
        self.forearm_inertia_x = calculate_moments_of_inertia(forearm_weight, forearm_radius, forearm_height)[0]
        self.forearm_inertia_y = calculate_moments_of_inertia(forearm_weight, forearm_radius, forearm_height)[1]
        self.forearm_inertia_z = calculate_moments_of_inertia(forearm_weight, forearm_radius, forearm_height)[2]

        self.hand_weight = hand_weight
        self.hand_radius = hand_radius
        self.hand_inertia = calculate_moments_of_inertia_hand(hand_weight, hand_radius)

        # the action space for the environment (0-1) because it will be multiplied by 100
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)

        # max output of the actuators in newton UNIFORM VALUE!!!!!!
        self.min_output = 0
        self.max_output_shoulder = max_force_shoulder
        self.max_output_elbow = max_force_elbow

        # vectors containing the info for the position values of the env: torques, positions
        self.actuator_forces = np.zeros(7)

        # max/min tremor torque values todo: set a calculation for it
        self.min_tremor_torq = -20
        self.max_tremor_torq = 20

        # action history
        self.second_prev_action = np.zeros(7)
        self.prev_action = np.zeros(7)

        # max/min values for actuator position vectors
        self.actuator_pos_min = -2
        self.actuator_pos_max = 2

        # define the observation space for the environment
        # observation space structure: actuator forces, tremor (t-2,t-1,t), act pos(t-1,t)

        low = np.array(
            [self.min_output,
             self.min_output,
             self.min_output,
             self.min_output,
             self.min_output,
             self.min_output,
             self.min_output,
             self.min_output,
             self.min_output,
             self.min_output,
             self.min_output,
             self.min_output,
             self.min_output,
             self.min_output,
             self.min_output,
             self.min_output,
             self.min_output,
             self.min_output,
             self.min_output,
             self.min_output,
             self.min_output,
             self.min_tremor_torq,
             self.min_tremor_torq,
             self.min_tremor_torq,
             self.min_tremor_torq,
             self.min_tremor_torq,
             self.min_tremor_torq,
             self.min_tremor_torq,
             self.min_tremor_torq,
             self.min_tremor_torq,
             self.min_tremor_torq,
             self.min_tremor_torq,
             self.min_tremor_torq,
             self.min_tremor_torq,
             self.min_tremor_torq,
             self.min_tremor_torq,
             self.min_tremor_torq,
             self.min_tremor_torq,
             self.min_tremor_torq,
             self.min_tremor_torq,
             self.min_tremor_torq,
             self.min_tremor_torq,
             self.actuator_pos_min,
             self.actuator_pos_min,
             self.actuator_pos_min,
             self.actuator_pos_min,
             self.actuator_pos_min,
             self.actuator_pos_min,
             self.actuator_pos_min,
             self.actuator_pos_min,
             self.actuator_pos_min,
             self.actuator_pos_min,
             self.actuator_pos_min,
             self.actuator_pos_min,
             self.actuator_pos_min,
             self.actuator_pos_min,
             self.actuator_pos_min,
             self.actuator_pos_min,
             self.actuator_pos_min,
             self.actuator_pos_min,
             self.actuator_pos_min,
             self.actuator_pos_min,
             self.actuator_pos_min,
             self.actuator_pos_min,
             self.actuator_pos_min,
             self.actuator_pos_min,
             self.actuator_pos_min,
             self.actuator_pos_min,
             self.actuator_pos_min,
             self.actuator_pos_min,
             self.actuator_pos_min,
             self.actuator_pos_min,
             self.actuator_pos_min,
             self.actuator_pos_min,
             self.actuator_pos_min,
             self.actuator_pos_min,
             self.actuator_pos_min,
             self.actuator_pos_min,
             self.actuator_pos_min,
             self.actuator_pos_min,
             self.actuator_pos_min,
             self.actuator_pos_min,
             self.actuator_pos_min,
             self.actuator_pos_min,
             self.actuator_pos_min,
             self.actuator_pos_min,
             self.actuator_pos_min,
             self.actuator_pos_min,
             self.actuator_pos_min,
             self.actuator_pos_min,
             self.actuator_pos_min,
             self.actuator_pos_min,
             self.actuator_pos_min,
             self.actuator_pos_min,
             self.actuator_pos_min,
             self.actuator_pos_min,
             ])

        # find the max of the acc values
        high = np.array(
            [self.max_output_shoulder / self.max_output_elbow,
             self.max_output_shoulder / self.max_output_elbow,
             self.max_output_shoulder / self.max_output_shoulder,
             self.max_output_shoulder / self.max_output_shoulder,
             self.max_output_shoulder / self.max_output_shoulder,
             self.max_output_shoulder / self.max_output_shoulder,
             self.max_output_shoulder / self.max_output_shoulder,
             self.max_output_shoulder / self.max_output_elbow,
             self.max_output_shoulder / self.max_output_elbow,
             self.max_output_shoulder / self.max_output_shoulder,
             self.max_output_shoulder / self.max_output_shoulder,
             self.max_output_shoulder / self.max_output_shoulder,
             self.max_output_shoulder / self.max_output_shoulder,
             self.max_output_shoulder / self.max_output_shoulder,
             self.max_output_shoulder / self.max_output_elbow,
             self.max_output_shoulder / self.max_output_elbow,
             self.max_output_shoulder / self.max_output_shoulder,
             self.max_output_shoulder / self.max_output_shoulder,
             self.max_output_shoulder / self.max_output_shoulder,
             self.max_output_shoulder / self.max_output_shoulder,
             self.max_output_shoulder / self.max_output_shoulder,
             self.max_tremor_torq,
             self.max_tremor_torq,
             self.max_tremor_torq,
             self.max_tremor_torq,
             self.max_tremor_torq,
             self.max_tremor_torq,
             self.max_tremor_torq,
             self.max_tremor_torq,
             self.max_tremor_torq,
             self.max_tremor_torq,
             self.max_tremor_torq,
             self.max_tremor_torq,
             self.max_tremor_torq,
             self.max_tremor_torq,
             self.max_tremor_torq,
             self.max_tremor_torq,
             self.max_tremor_torq,
             self.max_tremor_torq,
             self.max_tremor_torq,
             self.max_tremor_torq,
             self.max_tremor_torq,
             self.actuator_pos_max,
             self.actuator_pos_max,
             self.actuator_pos_max,
             self.actuator_pos_max,
             self.actuator_pos_max,
             self.actuator_pos_max,
             self.actuator_pos_max,
             self.actuator_pos_max,
             self.actuator_pos_max,
             self.actuator_pos_max,
             self.actuator_pos_max,
             self.actuator_pos_max,
             self.actuator_pos_max,
             self.actuator_pos_max,
             self.actuator_pos_max,
             self.actuator_pos_max,
             self.actuator_pos_max,
             self.actuator_pos_max,
             self.actuator_pos_max,
             self.actuator_pos_max,
             self.actuator_pos_max,
             self.actuator_pos_max,
             self.actuator_pos_max,
             self.actuator_pos_max,
             self.actuator_pos_max,
             self.actuator_pos_max,
             self.actuator_pos_max,
             self.actuator_pos_max,
             self.actuator_pos_max,
             self.actuator_pos_max,
             self.actuator_pos_max,
             self.actuator_pos_max,
             self.actuator_pos_max,
             self.actuator_pos_max,
             self.actuator_pos_max,
             self.actuator_pos_max,
             self.actuator_pos_max,
             self.actuator_pos_max,
             self.actuator_pos_max,
             self.actuator_pos_max,
             self.actuator_pos_max,
             self.actuator_pos_max,
             self.actuator_pos_max,
             self.actuator_pos_max,
             self.actuator_pos_max,
             self.actuator_pos_max,
             self.actuator_pos_max,
             self.actuator_pos_max,
             self.actuator_pos_max,
             self.actuator_pos_max,
             self.actuator_pos_max,
             self.actuator_pos_max,
             self.actuator_pos_max,
             self.actuator_pos_max,
             ],
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float64)

        # Connect to pybullet
        self.client = client
        client.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.client.resetSimulation()
        self.client.setGravity(0, 0, -9.81)  # todo: for dynamic randomization change this

        # define the patients data
        self.exoskeleton_sim_model = ExoskeletonSimModel(dummy_shift=dummy_shift, client=self.client)

        # normalization values for the tremor torques
        self.shoulder_norm_val = 10
        self.elbow_norm_val = 5
        self.wrist_norm_val = 0.5

        # angle calc
        self.to_rad = np.pi / 180

        # define a seed for the environment
        # self.seed()
        self.state = self.initialize_movement(evaluate=self.evaluate)

        # holds the position values for the actuators
        self.position_vectors = None
        self.prev_position_vectors = None

        # position values for the reference dummies
        self.ref_position_vectors = None
        self.ref_prev_position_vectors = None

        '''variable regarding the rewards'''
        # if a type of reward exceeds this (other than torque reward)-> set the next reward values to zero
        self.early_termination = False

        # matrices for tremor propagation
        self.I, self.D, self.S = seven_by_seven()
        self.T = None  # torque at a given episode
        self.T_act = None  # torque generated by the actuators onto each joints
        self.T_initial_values = np.zeros(14)

        # simulation boundary values
        self.shoulder_z_max_angle = 80
        self.shoulder_z_min_angle = -80
        self.shoulder_y_max_angle = 160.5
        self.shoulder_y_min_angle = -40
        self.shoulder_x_max_angle = 33.5
        self.shoulder_x_min_angle = -151.5
        self.elbow_y_max_angle = 150
        self.elbow_y_min_angle = -10
        self.elbow_z_max_angle = 80
        self.elbow_z_min_angle = -87

        self.low_bounds = np.array([self.shoulder_z_min_angle, self.shoulder_y_min_angle, self.shoulder_x_min_angle, self.elbow_y_min_angle])
        self.high_bounds = np.array([self.shoulder_z_max_angle, self.shoulder_y_max_angle, self.shoulder_x_max_angle, self.elbow_y_max_angle])

    def get_torques(self, force_components):
        radius_vectors = self.exoskeleton_sim_model.get_radius_vectors()
        actuator_torques = {"actuator1": np.cross(force_components["actuator1"], radius_vectors["actuator1"]),
                            "actuator2": np.cross(force_components["actuator2"], radius_vectors["actuator2"]),
                            "actuator3": np.cross(force_components["actuator3"], radius_vectors["actuator3"]),
                            "actuator4": np.cross(force_components["actuator4"], radius_vectors["actuator4"]),
                            "actuator5": np.cross(force_components["actuator5"], radius_vectors["actuator5"]),
                            "actuator6": np.cross(force_components["actuator6"], radius_vectors["actuator6"]),
                            "actuator7": np.cross(force_components["actuator7"], radius_vectors["actuator7"])}

        return actuator_torques

    def seed(self, seed=None):
        np.random, seed = seeding.np_random(seed)
        return [seed]

    def initialize_movement(self, evaluate):
        # load the correct reference motion in
        self.processed_imu_data = read_env_texts("reference_motions/ref_motion_" + self.file_num + ".txt")  # dictionary of numpy arrays, contains the target values
        self.max_count = self.processed_imu_data["elbow_joint_y_positions"].size  # size of a numpy array
        self.ep_state_values = define_empty_dict_for_env(self.max_count)
        self.tremor_torque_values, self.tremor_torque_places = generate_joint_torques_train(episode_length=self.max_count, all_seven=self.use_all_dof,
                                                                                            torque_sequence=self.tremor_input_sequence, dt=self.dt)

        # increment env counter
        self.env_counter += 1

        # define tremor angular acceleration values
        self.tremor_ang_acc_values = np.zeros_like(self.tremor_torque_values)
        self.tremor_ang_acc_values[0, :] = self.tremor_torque_values[0, :] / (self.humerus_inertia_y + self.forearm_inertia_y + self.hand_inertia)
        self.tremor_ang_acc_values[1, :] = self.tremor_torque_values[1, :] / (self.humerus_inertia_x + self.forearm_inertia_x + self.hand_inertia)
        self.tremor_ang_acc_values[2, :] = self.tremor_torque_values[2, :] / (self.humerus_inertia_z + self.forearm_inertia_z + self.hand_inertia)
        self.tremor_ang_acc_values[3, :] = self.tremor_torque_values[3, :] / (self.forearm_inertia_y + self.hand_inertia)
        self.tremor_ang_acc_values[4, :] = self.tremor_torque_values[4, :] / (self.forearm_inertia_z + self.hand_inertia)
        self.tremor_ang_acc_values[5, :] = self.tremor_torque_values[5, :] / self.hand_inertia
        self.tremor_ang_acc_values[6, :] = self.tremor_torque_values[6, :] / self.hand_inertia

        # randomly sample the imu datas and pick randomly a time of the movement
        # first is a boolean value if we want to initialize the movement at the beginning then it is true
        # return a vector in the shape of observation space
        if evaluate:
            # then return the first elements of the processed imu list
            self.counts = 2  # so we do not index out in the reward function

        else:
            random_index = random.randint(2, self.max_count - 1)
            self.counts = random_index

            if self.counts == self.max_count - 1:
                self.counts = 2

        # print(self.counts)

        # set the simulation values to the corresponding values in the target dict
        self.ep_state_values["elbow_joint_y_ang_acc"][0:self.counts] = 0
        self.ep_state_values["elbow_joint_y_ang_vel"][0:self.counts] = 0
        self.ep_state_values["shoulder_joint_x_ang_acc"][0:self.counts] = 0
        self.ep_state_values["shoulder_joint_x_ang_vel"][0:self.counts] = 0
        self.ep_state_values["shoulder_joint_y_ang_acc"][0:self.counts] = 0
        self.ep_state_values["shoulder_joint_y_ang_vel"][0:self.counts] = 0
        self.ep_state_values["shoulder_joint_z_ang_acc"][0:self.counts] = 0
        self.ep_state_values["shoulder_joint_z_ang_vel"][0:self.counts] = 0

        # set the corresponding joint angles--- differential eq initial conditions in the matlab script
        self.ep_state_values["elbow_joint_y_positions"][0:self.counts] = self.processed_imu_data["elbow_joint_y_positions"][0:self.counts]
        self.ep_state_values["elbow_joint_z_positions"][0:self.counts] = self.processed_imu_data["elbow_joint_z_positions"][0:self.counts]
        self.ep_state_values["shoulder_joint_x_positions"][0:self.counts] = self.processed_imu_data["shoulder_joint_x_positions"][0:self.counts]
        self.ep_state_values["shoulder_joint_y_positions"][0:self.counts] = self.processed_imu_data["shoulder_joint_y_positions"][0:self.counts]
        self.ep_state_values["shoulder_joint_z_positions"][0:self.counts] = self.processed_imu_data["shoulder_joint_z_positions"][0:self.counts]

        # set the corresponding actuator forces
        self.ep_state_values["actuator_1_forces"][0:self.counts] = 0.0
        self.ep_state_values["actuator_2_forces"][0:self.counts] = 0.0
        self.ep_state_values["actuator_3_forces"][0:self.counts] = 0.0
        self.ep_state_values["actuator_4_forces"][0:self.counts] = 0.0
        self.ep_state_values["actuator_5_forces"][0:self.counts] = 0.0
        self.ep_state_values["actuator_6_forces"][0:self.counts] = 0.0
        self.ep_state_values["actuator_7_forces"][0:self.counts] = 0.0

        # calculate previous position
        self.exoskeleton_sim_model.set_joint_position([self.ep_state_values["shoulder_joint_z_positions"][self.counts - 1] * self.to_rad,
                                                       self.ep_state_values["shoulder_joint_y_positions"][self.counts - 1] * self.to_rad,
                                                       self.ep_state_values["shoulder_joint_x_positions"][self.counts - 1] * self.to_rad,
                                                       self.ep_state_values["elbow_joint_y_positions"][self.counts - 1] * self.to_rad,
                                                       self.processed_imu_data["elbow_joint_z_positions"][self.counts - 1] * self.to_rad])

        _ = self.exoskeleton_sim_model.get_actuator_positions()
        self.prev_position_vectors = self.exoskeleton_sim_model.get_actuator_pos_vect()
        self.ref_prev_position_vectors = self.exoskeleton_sim_model.return_reference_dummy_pos()

        # set the exoskeleton position into order
        self.exoskeleton_sim_model.set_joint_position([self.ep_state_values["shoulder_joint_z_positions"][self.counts] * self.to_rad,
                                                       self.ep_state_values["shoulder_joint_y_positions"][self.counts] * self.to_rad,
                                                       self.ep_state_values["shoulder_joint_x_positions"][self.counts] * self.to_rad,
                                                       self.ep_state_values["elbow_joint_y_positions"][self.counts] * self.to_rad,
                                                       self.processed_imu_data["elbow_joint_z_positions"][self.counts] * self.to_rad])

        # position vectors
        _ = self.exoskeleton_sim_model.get_actuator_positions()
        self.position_vectors = self.exoskeleton_sim_model.get_actuator_pos_vect()
        self.ref_position_vectors = self.exoskeleton_sim_model.return_reference_dummy_pos()

        # return the target motions values (tremor torque values are defined as zero since tremor cannot be in the first time step)
        return np.array([self.ep_state_values["actuator_1_forces"][self.counts] / self.max_output_elbow,
                         self.ep_state_values["actuator_2_forces"][self.counts] / self.max_output_elbow,
                         self.ep_state_values["actuator_3_forces"][self.counts] / self.max_output_shoulder,
                         self.ep_state_values["actuator_4_forces"][self.counts] / self.max_output_shoulder,
                         self.ep_state_values["actuator_5_forces"][self.counts] / self.max_output_shoulder,
                         self.ep_state_values["actuator_6_forces"][self.counts] / self.max_output_shoulder,
                         self.ep_state_values["actuator_7_forces"][self.counts] / self.max_output_shoulder,
                         self.ep_state_values["actuator_1_forces"][self.counts] / self.max_output_elbow,
                         self.ep_state_values["actuator_2_forces"][self.counts] / self.max_output_elbow,
                         self.ep_state_values["actuator_3_forces"][self.counts] / self.max_output_shoulder,
                         self.ep_state_values["actuator_4_forces"][self.counts] / self.max_output_shoulder,
                         self.ep_state_values["actuator_5_forces"][self.counts] / self.max_output_shoulder,
                         self.ep_state_values["actuator_6_forces"][self.counts] / self.max_output_shoulder,
                         self.ep_state_values["actuator_7_forces"][self.counts] / self.max_output_shoulder,
                         self.ep_state_values["actuator_1_forces"][self.counts] / self.max_output_elbow,
                         self.ep_state_values["actuator_2_forces"][self.counts] / self.max_output_elbow,
                         self.ep_state_values["actuator_3_forces"][self.counts] / self.max_output_shoulder,
                         self.ep_state_values["actuator_4_forces"][self.counts] / self.max_output_shoulder,
                         self.ep_state_values["actuator_5_forces"][self.counts] / self.max_output_shoulder,
                         self.ep_state_values["actuator_6_forces"][self.counts] / self.max_output_shoulder,
                         self.ep_state_values["actuator_7_forces"][self.counts] / self.max_output_shoulder,
                         self.tremor_torque_values[0, self.counts - 2] / self.shoulder_norm_val,
                         self.tremor_torque_values[1, self.counts - 2] / self.shoulder_norm_val,
                         self.tremor_torque_values[2, self.counts - 2] / self.shoulder_norm_val,
                         self.tremor_torque_values[3, self.counts - 2] / self.elbow_norm_val,
                         self.tremor_torque_values[4, self.counts - 2] / self.elbow_norm_val,
                         self.tremor_torque_values[5, self.counts - 2] / self.wrist_norm_val,
                         self.tremor_torque_values[6, self.counts - 2] / self.wrist_norm_val,
                         self.tremor_torque_values[0, self.counts - 1] / self.shoulder_norm_val,
                         self.tremor_torque_values[1, self.counts - 1] / self.shoulder_norm_val,
                         self.tremor_torque_values[2, self.counts - 1] / self.shoulder_norm_val,
                         self.tremor_torque_values[3, self.counts - 1] / self.elbow_norm_val,
                         self.tremor_torque_values[4, self.counts - 1] / self.elbow_norm_val,
                         self.tremor_torque_values[5, self.counts - 1] / self.wrist_norm_val,
                         self.tremor_torque_values[6, self.counts - 1] / self.wrist_norm_val,
                         self.tremor_torque_values[0, self.counts] / self.shoulder_norm_val,
                         self.tremor_torque_values[1, self.counts] / self.shoulder_norm_val,
                         self.tremor_torque_values[2, self.counts] / self.shoulder_norm_val,
                         self.tremor_torque_values[3, self.counts] / self.elbow_norm_val,
                         self.tremor_torque_values[4, self.counts] / self.elbow_norm_val,
                         self.tremor_torque_values[5, self.counts] / self.wrist_norm_val,
                         self.tremor_torque_values[6, self.counts] / self.wrist_norm_val,
                         self.prev_position_vectors["actuator1"][0],
                         self.prev_position_vectors["actuator1"][1],
                         self.prev_position_vectors["actuator1"][2],
                         self.prev_position_vectors["actuator2"][0],
                         self.prev_position_vectors["actuator2"][1],
                         self.prev_position_vectors["actuator2"][2],
                         self.prev_position_vectors["actuator3"][0],
                         self.prev_position_vectors["actuator3"][1],
                         self.prev_position_vectors["actuator3"][2],
                         self.prev_position_vectors["actuator4"][0],
                         self.prev_position_vectors["actuator4"][1],
                         self.prev_position_vectors["actuator4"][2],
                         self.prev_position_vectors["actuator5"][0],
                         self.prev_position_vectors["actuator5"][1],
                         self.prev_position_vectors["actuator5"][2],
                         self.prev_position_vectors["actuator6"][0],
                         self.prev_position_vectors["actuator6"][1],
                         self.prev_position_vectors["actuator6"][2],
                         self.prev_position_vectors["actuator7"][0],
                         self.prev_position_vectors["actuator7"][1],
                         self.prev_position_vectors["actuator7"][2],
                         self.position_vectors["actuator1"][0],
                         self.position_vectors["actuator1"][1],
                         self.position_vectors["actuator1"][2],
                         self.position_vectors["actuator2"][0],
                         self.position_vectors["actuator2"][1],
                         self.position_vectors["actuator2"][2],
                         self.position_vectors["actuator3"][0],
                         self.position_vectors["actuator3"][1],
                         self.position_vectors["actuator3"][2],
                         self.position_vectors["actuator4"][0],
                         self.position_vectors["actuator4"][1],
                         self.position_vectors["actuator4"][2],
                         self.position_vectors["actuator5"][0],
                         self.position_vectors["actuator5"][1],
                         self.position_vectors["actuator5"][2],
                         self.position_vectors["actuator6"][0],
                         self.position_vectors["actuator6"][1],
                         self.position_vectors["actuator6"][2],
                         self.position_vectors["actuator7"][0],
                         self.position_vectors["actuator7"][1],
                         self.position_vectors["actuator7"][2],
                         self.ref_prev_position_vectors[0][0],
                         self.ref_prev_position_vectors[0][1],
                         self.ref_prev_position_vectors[0][2],
                         self.ref_prev_position_vectors[1][0],
                         self.ref_prev_position_vectors[1][1],
                         self.ref_prev_position_vectors[1][2],
                         self.ref_position_vectors[0][0],
                         self.ref_position_vectors[0][1],
                         self.ref_position_vectors[0][2],
                         self.ref_position_vectors[1][0],
                         self.ref_position_vectors[1][1],
                         self.ref_position_vectors[1][2]
                         ])

    def transform_action(self, actions):
        transformed_actions = np.zeros((7,))
        transformed_actions[:2] = ((actions[:2] + 1) / 2) * self.max_output_elbow
        transformed_actions[2:] = ((actions[2:] + 1) / 2) * self.max_output_shoulder

        return transformed_actions

    def step(self, action: np.array):
        action = self.transform_action(actions=action)

        # action is the vector which contains the force values by the actuators
        # Append these to the self.episode_state_values
        self.ep_state_values["actuator_1_forces"][self.counts] = action[0]
        self.ep_state_values["actuator_2_forces"][self.counts] = action[1]
        self.ep_state_values["actuator_3_forces"][self.counts] = action[2]
        self.ep_state_values["actuator_4_forces"][self.counts] = action[3]
        self.ep_state_values["actuator_5_forces"][self.counts] = action[4]
        self.ep_state_values["actuator_6_forces"][self.counts] = action[5]
        self.ep_state_values["actuator_7_forces"][self.counts] = action[6]

        # set the angles for the joints
        _ = self.exoskeleton_sim_model.get_actuator_positions()  # update the actuator positions
        self.ref_prev_position_vectors = self.ref_position_vectors
        force_components = self.exoskeleton_sim_model.get_force_components(forces=action)  # a dict of (3,) vectors
        self.prev_position_vectors = self.position_vectors
        self.position_vectors = self.exoskeleton_sim_model.get_actuator_pos_vect()
        self.ref_position_vectors = self.exoskeleton_sim_model.return_reference_dummy_pos()
        actuator_torques = self.get_torques(force_components)

        # for debugging
        # print("Given force in newtons: ", action)
        # print("Torques applied to shoulder y: ", "actuator 3: ", actuator_torques["actuator3"][1], "actuator 4: ", actuator_torques["actuator4"][1], "actuator 5: ",
        #       actuator_torques["actuator5"][1], "actuator 7: ", actuator_torques["actuator7"][1], "actuator 6: ", actuator_torques["actuator6"][1])
        # print("Torques applied to shoulder x: ", "actuator 3: ", actuator_torques["actuator3"][0], "actuator 4: ", actuator_torques["actuator4"][0], "actuator 5: ",
        #       actuator_torques["actuator5"][0], "actuator 7: ", actuator_torques["actuator7"][0], "actuator 6: ", actuator_torques["actuator6"][0])
        # print("Torques applied to shoulder z: ", "actuator 3: ", actuator_torques["actuator3"][2], "actuator 4: ", actuator_torques["actuator4"][2], "actuator 5: ",
        #       actuator_torques["actuator5"][2], "actuator 7: ", actuator_torques["actuator7"][2], "actuator 6: ", actuator_torques["actuator6"][2])
        # print("Torques applied to elbow y: ", "actuator 1: ", abs(actuator_torques["actuator1"][1]), "actuator 2: ", -abs(actuator_torques["actuator2"][1]))
        # print()

        # which actuator influences which joint: (same order as diff eq matrix shoulder y,x,z elbow y
        self.T_act = np.array([actuator_torques["actuator3"][1] + actuator_torques["actuator4"][1] + actuator_torques["actuator5"][1] + actuator_torques["actuator7"][1] +
                               actuator_torques["actuator6"][1],
                               actuator_torques["actuator3"][0] + actuator_torques["actuator4"][0] + actuator_torques["actuator5"][0] + actuator_torques["actuator7"][0] +
                               actuator_torques["actuator6"][0],
                               actuator_torques["actuator3"][2] + actuator_torques["actuator4"][2] + actuator_torques["actuator5"][2] + actuator_torques["actuator7"][2] +
                               actuator_torques["actuator6"][2],
                               abs(actuator_torques["actuator1"][1]) - abs(actuator_torques["actuator2"][1]),
                               0, 0, 0
                               ])

        # apply  tremor difference here
        self.T = np.array(self.tremor_torque_values[:, self.counts])

        # return the torque values (shoulder y,x,z; elbow y)
        torque_values = self.T + self.T_act
        torque_values = np.array(torque_values)

        # mitigated joint angle values
        act_tremor_joint_angles = solve_diff_eq(self.T_initial_values, [0, self.dt], self.I, self.D, self.S, torque_values)
        act_tremor_joint_angles = np.reshape(np.array(act_tremor_joint_angles), (7,))

        # unmitigated joint angle values
        tremor_joint_angles = solve_diff_eq(self.T_initial_values, [0, self.dt], self.I, self.D, self.S, self.T)
        tremor_joint_angles = np.reshape(np.array(tremor_joint_angles), (7,))

        # convert to degrees
        act_tremor_joint_angles = act_tremor_joint_angles * (180 / np.pi)
        tremor_joint_angles = tremor_joint_angles * (180 / np.pi)

        # angles are made up from the previous angle and the change in person angle, tremor angle and actuator angles
        self.ep_state_values["shoulder_joint_z_positions"][self.counts] = self.processed_imu_data["shoulder_joint_z_positions"][self.counts] + act_tremor_joint_angles[2]

        self.ep_state_values["shoulder_joint_y_positions"][self.counts] = self.processed_imu_data["shoulder_joint_y_positions"][self.counts] + act_tremor_joint_angles[0]

        self.ep_state_values["shoulder_joint_x_positions"][self.counts] = self.processed_imu_data["shoulder_joint_x_positions"][self.counts] + act_tremor_joint_angles[1]

        self.ep_state_values["elbow_joint_y_positions"][self.counts] = self.processed_imu_data["elbow_joint_y_positions"][self.counts] + act_tremor_joint_angles[3]

        # set the simulation angles
        self.exoskeleton_sim_model.set_joint_position([self.ep_state_values["shoulder_joint_z_positions"][self.counts] * self.to_rad,
                                                       self.ep_state_values["shoulder_joint_y_positions"][self.counts] * self.to_rad,
                                                       self.ep_state_values["shoulder_joint_x_positions"][self.counts] * self.to_rad,
                                                       self.ep_state_values["elbow_joint_y_positions"][self.counts] * self.to_rad,
                                                       self.processed_imu_data["elbow_joint_z_positions"][self.counts] * self.to_rad])

        self.client.stepSimulation()

        # angular accelerations
        self.ep_state_values["elbow_joint_y_ang_acc"][self.counts] = (self.processed_imu_data["elbow_joint_y_ang_acc"][self.counts] -
                                                                      self.processed_imu_data["elbow_joint_y_ang_acc"][self.counts - 1]) + \
                                                                     self.tremor_ang_acc_values[3, self.counts] + \
                                                                     (actuator_torques["actuator1"][1] / self.forearm_inertia_y) - \
                                                                     (actuator_torques["actuator2"][1] / self.forearm_inertia_y)

        self.ep_state_values["shoulder_joint_x_ang_acc"][self.counts] = (self.processed_imu_data["shoulder_joint_x_ang_acc"][self.counts] -
                                                                         self.processed_imu_data["shoulder_joint_x_ang_acc"][self.counts - 1]) + \
                                                                        self.tremor_ang_acc_values[1, self.counts] + \
                                                                        (actuator_torques["actuator3"][0] / (self.forearm_inertia_x + self.humerus_inertia_x)) + \
                                                                        (actuator_torques["actuator4"][0] / (self.forearm_inertia_x + self.humerus_inertia_x)) + \
                                                                        (actuator_torques["actuator5"][0] / (self.forearm_inertia_x + self.humerus_inertia_x)) + \
                                                                        (actuator_torques["actuator7"][0] / (self.forearm_inertia_x + self.humerus_inertia_x)) - \
                                                                        (actuator_torques["actuator6"][0] / (self.forearm_inertia_x + self.humerus_inertia_x))

        self.ep_state_values["shoulder_joint_y_ang_acc"][self.counts] = (self.processed_imu_data["shoulder_joint_y_ang_acc"][self.counts] -
                                                                         self.processed_imu_data["shoulder_joint_y_ang_acc"][self.counts - 1]) + \
                                                                        self.tremor_ang_acc_values[0, self.counts] + \
                                                                        (actuator_torques["actuator3"][1] / (self.forearm_inertia_y + self.humerus_inertia_y)) - \
                                                                        (actuator_torques["actuator4"][1] / (self.forearm_inertia_y + self.humerus_inertia_y)) + \
                                                                        (actuator_torques["actuator5"][1] / (self.forearm_inertia_y + self.humerus_inertia_y)) - \
                                                                        (actuator_torques["actuator7"][1] / (self.forearm_inertia_y + self.humerus_inertia_y))

        self.ep_state_values["shoulder_joint_z_ang_acc"][self.counts] = (self.processed_imu_data["shoulder_joint_z_ang_acc"][self.counts] -
                                                                         self.processed_imu_data["shoulder_joint_z_ang_acc"][self.counts - 1]) + \
                                                                        self.tremor_ang_acc_values[2, self.counts] + \
                                                                        (actuator_torques["actuator3"][2] / (self.forearm_inertia_z + self.humerus_inertia_z)) - \
                                                                        (actuator_torques["actuator4"][2] / (self.forearm_inertia_z + self.humerus_inertia_z)) + \
                                                                        (actuator_torques["actuator5"][2] / (self.forearm_inertia_z + self.humerus_inertia_z)) - \
                                                                        (actuator_torques["actuator7"][2] / (self.forearm_inertia_z + self.humerus_inertia_z))

        # angular velocity values
        self.ep_state_values["elbow_joint_y_ang_vel"][self.counts] = self.ep_state_values["elbow_joint_y_ang_vel"][self.counts - 1] + \
                                                                     self.ep_state_values["elbow_joint_y_ang_acc"][self.counts] * self.dt

        self.ep_state_values["shoulder_joint_x_ang_vel"][self.counts] = self.ep_state_values["shoulder_joint_x_ang_vel"][self.counts - 1] + \
                                                                        self.ep_state_values["shoulder_joint_x_ang_acc"][self.counts] * self.dt

        self.ep_state_values["shoulder_joint_y_ang_vel"][self.counts] = self.ep_state_values["shoulder_joint_y_ang_vel"][self.counts - 1] + \
                                                                        self.ep_state_values["shoulder_joint_y_ang_acc"][self.counts] * self.dt

        self.ep_state_values["shoulder_joint_z_ang_vel"][self.counts] = self.ep_state_values["shoulder_joint_z_ang_vel"][self.counts - 1] + \
                                                                        self.ep_state_values["shoulder_joint_z_ang_acc"][self.counts] * self.dt

        # define whether the simulation is done
        done = False

        # check if the new joint positions will not go over the movement boundaries
        if self.shoulder_z_min_angle < self.ep_state_values["shoulder_joint_z_positions"][self.counts] < self.shoulder_z_max_angle and \
                self.shoulder_y_min_angle < self.ep_state_values["shoulder_joint_y_positions"][self.counts] < self.shoulder_y_max_angle and \
                self.shoulder_x_min_angle < self.ep_state_values["shoulder_joint_x_positions"][self.counts] < self.shoulder_x_max_angle and \
                self.elbow_y_min_angle < self.ep_state_values["elbow_joint_y_positions"][self.counts] < self.elbow_y_max_angle:
            extended_boundary = False
        else:
            extended_boundary = True
            if (self.shoulder_z_min_angle < self.ep_state_values["shoulder_joint_z_positions"][self.counts] < self.shoulder_z_max_angle) != 1:
                print("Agent went over the movement boundaries Shoulder z.", self.ep_state_values["shoulder_joint_z_positions"][self.counts])
            elif (self.shoulder_y_min_angle < self.ep_state_values["shoulder_joint_y_positions"][self.counts] < self.shoulder_y_max_angle) != 1:
                print("Agent went over the movement boundaries Shoulder y.", self.ep_state_values["shoulder_joint_y_positions"][self.counts])
            elif (self.shoulder_x_min_angle < self.ep_state_values["shoulder_joint_x_positions"][self.counts] < self.shoulder_x_max_angle) != 1:
                print("Agent went over the movement boundaries Shoulder x.", self.ep_state_values["shoulder_joint_x_positions"][self.counts])
            else:
                print("Agent went over the movement boundaries elbow y.", self.ep_state_values["elbow_joint_y_positions"][self.counts])

        # define the reward function
        reward = 0.0

        # the weight of the reward function components
        weight_axis_reward = 0.5
        weight_torque_reward = 0.9
        weight_actuator_force_magnitude_reward = 0.05
        weight_actuator_smoothness_reward = 0.05
        weight_unwanted_component = 0.4

        if not done and self.early_termination is False:

            # tremor reward values
            # sparse reward enforcing that tremor reduction strategies involve all effected axis
            tremor_torque_reduction = (abs(torque_values) - abs(self.T)) / abs(self.T) * 100
            tremor_torque_reduction = np.nan_to_num(tremor_torque_reduction, nan=0, posinf=0, neginf=0)

            involved_axis = np.sum(tremor_torque_reduction < 0)

            # torque sum calculation
            sum_torque = 0
            sum_unwanted = 0
            for index in range(4):
                if self.tremor_torque_places[index] == 1:  # todo: if no tremor reduction minimize it compared to 0
                    sum_torque += (abs(torque_values[index]) - abs(self.T[index])) / abs(self.T[index]) + 1
                else:
                    sum_unwanted += abs(torque_values[index])

            # actuator force reward part
            sum_actions = np.sum(action)

            # actuator smoothness reward
            sum_smoothness = np.mean((action - 2 * self.prev_action + self.second_prev_action) ** 2)

            # apply e^-x func
            torque_reward = np.exp(-sum_torque + 1e-8)
            smoothness_reward = np.exp(-(sum_smoothness / ((self.max_output_elbow + self.max_output_shoulder)/4)) + 1e-8)  # scaling is applied by division
            actuator_reward = np.exp(-(sum_actions / ((self.max_output_elbow + self.max_output_shoulder)/2)) + 1e-8)  # scaling is applied by division
            unwanted_reward = np.exp(-(sum_unwanted / ((self.max_output_elbow + self.max_output_shoulder)/4)) + 1e-8)  # scaling is applied by division

            # normalization value so reward falls into [0,1]
            max_reward = np.sum(self.tremor_input_sequence) * weight_axis_reward + weight_torque_reward + weight_actuator_force_magnitude_reward \
                         + weight_actuator_smoothness_reward + weight_unwanted_component

            # sum of the reward function
            reward = (weight_axis_reward * involved_axis + weight_torque_reward * torque_reward +
                      weight_actuator_smoothness_reward * smoothness_reward + weight_actuator_force_magnitude_reward * actuator_reward
                      + weight_unwanted_component * unwanted_reward) / max_reward

        # for debugging
        # print(actuator_torques)

        # increment the count of the simulation
        self.counts += 1

        # initialize the state vector
        self.state = np.array([self.second_prev_action[0] / self.max_output_elbow,
                               self.second_prev_action[1] / self.max_output_elbow,
                               self.second_prev_action[2] / self.max_output_shoulder,
                               self.second_prev_action[3] / self.max_output_shoulder,
                               self.second_prev_action[4] / self.max_output_shoulder,
                               self.second_prev_action[5] / self.max_output_shoulder,
                               self.second_prev_action[6] / self.max_output_shoulder,
                               self.prev_action[0] / self.max_output_elbow,
                               self.prev_action[1] / self.max_output_elbow,
                               self.prev_action[2] / self.max_output_shoulder,
                               self.prev_action[3] / self.max_output_shoulder,
                               self.prev_action[4] / self.max_output_shoulder,
                               self.prev_action[5] / self.max_output_shoulder,
                               self.prev_action[6] / self.max_output_shoulder,
                               self.ep_state_values["actuator_1_forces"][self.counts] / self.max_output_elbow,
                               self.ep_state_values["actuator_2_forces"][self.counts] / self.max_output_elbow,
                               self.ep_state_values["actuator_3_forces"][self.counts] / self.max_output_shoulder,
                               self.ep_state_values["actuator_4_forces"][self.counts] / self.max_output_shoulder,
                               self.ep_state_values["actuator_5_forces"][self.counts] / self.max_output_shoulder,
                               self.ep_state_values["actuator_6_forces"][self.counts] / self.max_output_shoulder,
                               self.ep_state_values["actuator_7_forces"][self.counts] / self.max_output_shoulder,
                               self.tremor_torque_values[0, self.counts - 2] / self.shoulder_norm_val,
                               self.tremor_torque_values[1, self.counts - 2] / self.shoulder_norm_val,
                               self.tremor_torque_values[2, self.counts - 2] / self.shoulder_norm_val,
                               self.tremor_torque_values[3, self.counts - 2] / self.elbow_norm_val,
                               self.tremor_torque_values[4, self.counts - 2] / self.elbow_norm_val,
                               self.tremor_torque_values[5, self.counts - 2] / self.wrist_norm_val,
                               self.tremor_torque_values[6, self.counts - 2] / self.wrist_norm_val,
                               self.tremor_torque_values[0, self.counts - 1] / self.shoulder_norm_val,
                               self.tremor_torque_values[1, self.counts - 1] / self.shoulder_norm_val,
                               self.tremor_torque_values[2, self.counts - 1] / self.shoulder_norm_val,
                               self.tremor_torque_values[3, self.counts - 1] / self.elbow_norm_val,
                               self.tremor_torque_values[4, self.counts - 1] / self.elbow_norm_val,
                               self.tremor_torque_values[5, self.counts - 1] / self.wrist_norm_val,
                               self.tremor_torque_values[6, self.counts - 1] / self.wrist_norm_val,
                               self.tremor_torque_values[0, self.counts] / self.shoulder_norm_val,
                               self.tremor_torque_values[1, self.counts] / self.shoulder_norm_val,
                               self.tremor_torque_values[2, self.counts] / self.shoulder_norm_val,
                               self.tremor_torque_values[3, self.counts] / self.elbow_norm_val,
                               self.tremor_torque_values[4, self.counts] / self.elbow_norm_val,
                               self.tremor_torque_values[5, self.counts] / self.wrist_norm_val,
                               self.tremor_torque_values[6, self.counts] / self.wrist_norm_val,
                               self.prev_position_vectors["actuator1"][0],
                               self.prev_position_vectors["actuator1"][1],
                               self.prev_position_vectors["actuator1"][2],
                               self.prev_position_vectors["actuator2"][0],
                               self.prev_position_vectors["actuator2"][1],
                               self.prev_position_vectors["actuator2"][2],
                               self.prev_position_vectors["actuator3"][0],
                               self.prev_position_vectors["actuator3"][1],
                               self.prev_position_vectors["actuator3"][2],
                               self.prev_position_vectors["actuator4"][0],
                               self.prev_position_vectors["actuator4"][1],
                               self.prev_position_vectors["actuator4"][2],
                               self.prev_position_vectors["actuator5"][0],
                               self.prev_position_vectors["actuator5"][1],
                               self.prev_position_vectors["actuator5"][2],
                               self.prev_position_vectors["actuator6"][0],
                               self.prev_position_vectors["actuator6"][1],
                               self.prev_position_vectors["actuator6"][2],
                               self.prev_position_vectors["actuator7"][0],
                               self.prev_position_vectors["actuator7"][1],
                               self.prev_position_vectors["actuator7"][2],
                               self.position_vectors["actuator1"][0],
                               self.position_vectors["actuator1"][1],
                               self.position_vectors["actuator1"][2],
                               self.position_vectors["actuator2"][0],
                               self.position_vectors["actuator2"][1],
                               self.position_vectors["actuator2"][2],
                               self.position_vectors["actuator3"][0],
                               self.position_vectors["actuator3"][1],
                               self.position_vectors["actuator3"][2],
                               self.position_vectors["actuator4"][0],
                               self.position_vectors["actuator4"][1],
                               self.position_vectors["actuator4"][2],
                               self.position_vectors["actuator5"][0],
                               self.position_vectors["actuator5"][1],
                               self.position_vectors["actuator5"][2],
                               self.position_vectors["actuator6"][0],
                               self.position_vectors["actuator6"][1],
                               self.position_vectors["actuator6"][2],
                               self.position_vectors["actuator7"][0],
                               self.position_vectors["actuator7"][1],
                               self.position_vectors["actuator7"][2],
                               self.ref_prev_position_vectors[0][0],
                               self.ref_prev_position_vectors[0][1],
                               self.ref_prev_position_vectors[0][2],
                               self.ref_prev_position_vectors[1][0],
                               self.ref_prev_position_vectors[1][1],
                               self.ref_prev_position_vectors[1][2],
                               self.ref_position_vectors[0][0],
                               self.ref_position_vectors[0][1],
                               self.ref_position_vectors[0][2],
                               self.ref_position_vectors[1][0],
                               self.ref_position_vectors[1][1],
                               self.ref_position_vectors[1][2]
                               ])

        # set the actions to the prev and so forth
        self.second_prev_action = self.prev_action
        self.prev_action = action

        if self.counts >= self.max_count - 1:  # we have reached the end of the measurement in time
            done = True

        # print the angle values
        # print(self.exoskeleton_sim_model.getJointPositions())

        # self.T: Tremor torque, torque_values: all the torques (exo+tremor)
        return np.array(self.state, dtype=np.float32), reward, done, self.T_act, torque_values, act_tremor_joint_angles, self.T, tremor_joint_angles, {}

    def reset(self):
        self.actuator_forces = np.zeros(7)
        # reset the exo
        self.client.resetSimulation()
        self.client.setGravity(0, 0, -9.81)

        # reload the exo
        self.exoskeleton_sim_model.load_again()
        self.state = self.initialize_movement(evaluate=self.evaluate)

        score = self.counts
        self.early_termination = False

        return self.state, score

    def render(self, mode="human"):
        pass

    def close(self):
        self.client.disconnect()
        print('Close the environment')

    def return_generated_tremor_data(self):
        # get the maximum value of each column of the generated tremors
        max_values = self.tremor_torque_values.max(axis=1)
        return self.tremor_torque_places, max_values

    def return_max_length(self):
        return self.max_count

    def return_original_joint_angles(self):
        # returns the IMU measured values
        return [self.processed_imu_data["shoulder_joint_x_positions"][self.counts],
                self.processed_imu_data["shoulder_joint_y_positions"][self.counts],
                self.processed_imu_data["shoulder_joint_z_positions"][self.counts],
                self.processed_imu_data["elbow_joint_y_positions"][self.counts],
                self.processed_imu_data["elbow_joint_z_positions"][self.counts],
                0,
                0]


if __name__ == "__main__":

    # values for the simulation
    mass = 81.5  # in kg
    u_arm_weight, l_arm_weight, h_weight = calculate_body_part_mass(mass)
    sum_rewards = 0
    scores = []
    client = bc.BulletClient(connection_mode=p.DIRECT)

    env = ExoskeletonEnv_train(evaluation=True, dummy_shift=False, hum_weight=u_arm_weight, forearm_weight=l_arm_weight, forearm_height=0.4, forearm_radius=0.15, hum_height=0.4,
                               hum_radius=0.15, hand_weight=h_weight, hand_radius=0.05, use_all_dof=False, reference_motion_file_num=str(7),
                               tremor_sequence=np.array([0, 0, 0, 1, 0, 0, 0]), max_force_shoulder=30, max_force_elbow=18, client=client)

    # test the running of the environment
    for i in range(1):
        observation, score = env.reset()
        done = False
        ep_len = score
        while not done:
            action = np.array([-1, 1, -1, -1, -1, -1, -1])  # random action
            observation_, reward, done, actuator_torques, torq_val, ampl_val, tremor_torq_val, tremor_ampl_val, info = env.step(action)
            score += reward
            ep_len += 1

            # print("Observation values: ", observation_)
            # print("Torque values: ", torq_val)
            # print("Amplitude values: ", ampl_val)
            # print("Reward: ", reward)
        scores.append(score)
        print("Score: ", score, ", Episode length: ", ep_len, ", Score in percent to max: ", score / ep_len * 100, " Average score: ", np.mean(scores))

    env.close()
