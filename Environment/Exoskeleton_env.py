import numpy as np
import pybullet_utils.bullet_client as bc
import pybullet as p
import pybullet_data
import gym
from gym.utils import seeding
from gym import spaces
from Environment.Exoskeleton_sim_pybullet import ExoskeletonSimModel
from Utilities.read_txt_env import define_empty_dict_for_env, read_env_texts
from Utilities.generate_parkinson_tremor import generate_joint_torques_train
from Utilities.differential_eq_matrices import seven_by_seven
from Utilities.calculate_joint_angles import solve_diff_eq
from Utilities.domain_randomization_anatomical_matrices import add_symmetric_noise


def debug_exoskeleton_torque_feedback(action, actuator_torques) -> None:
    """
    Optional evaluation function for debugging. Prints out the exoskeleton torques present at each joint axes.
    :param action: The
    :param actuator_torques:
    :return:
    """
    print("Given force in newtons: ", action)
    print("Torques applied to shoulder y: ", "actuator 3: ", actuator_torques["actuator3"][1], "actuator 4: ", actuator_torques["actuator4"][1], "actuator 5: ",
          actuator_torques["actuator5"][1], "actuator 7: ", actuator_torques["actuator7"][1], "actuator 6: ", actuator_torques["actuator6"][1])
    print("Torques applied to shoulder x: ", "actuator 3: ", actuator_torques["actuator3"][0], "actuator 4: ", actuator_torques["actuator4"][0], "actuator 5: ",
          actuator_torques["actuator5"][0], "actuator 7: ", actuator_torques["actuator7"][0], "actuator 6: ", actuator_torques["actuator6"][0])
    print("Torques applied to shoulder z: ", "actuator 3: ", actuator_torques["actuator3"][2], "actuator 4: ", actuator_torques["actuator4"][2], "actuator 5: ",
          actuator_torques["actuator5"][2], "actuator 7: ", actuator_torques["actuator7"][2], "actuator 6: ", actuator_torques["actuator6"][2])
    print("Torques applied to elbow y: ", "actuator 1: ", abs(actuator_torques["actuator1"][1]), "actuator 2: ", -abs(actuator_torques["actuator2"][1]))
    print()


class ExoskeletonEnv_train(gym.Env):
    """Custom Environment that follows gym interface """
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 reference_motion_file_num: str,
                 tremor_sequence: np.array,
                 tremor_amplitude_range: np.array,
                 first_harmonics_interval: np.array,
                 second_harmonics_interval: np.array,
                 max_force_shoulder: float,
                 max_force_elbow: float,
                 dr_actuator_end_pos_shift: float = 0.02,
                 dr_actuator_range: float = 0.1,
                 matrix_noise_fraction: float = 0.1):
        """"""
        super(ExoskeletonEnv_train, self).__init__()
        # Pybullet simulation parameters
        self.dummy_shift_max_range = dr_actuator_end_pos_shift
        self.file_num = reference_motion_file_num

        # the episode count in the texts and simulation time space
        self.counts: int = 1

        # define the time steps of the simulation
        self.dt: float = 1 / 40

        # reference movement variables
        self.processed_imu_data = read_env_texts("reference_motions/ref_motion_" + self.file_num + ".txt")
        self.max_count = self.processed_imu_data["elbow_joint_y_positions"].size
        self.ep_state_values = define_empty_dict_for_env(self.max_count)

        # tremor variables
        self.tremor_torque_values = np.zeros((7, self.max_count))
        self.tremor_input_sequence = tremor_sequence
        self.tremor_amplitude_range = tremor_amplitude_range
        self.tremor_magnitude: float = 0.1 + (np.random.rand() * 0.9)
        self.tremor_axis_n = sum(self.tremor_input_sequence)

        # the action space for the environment
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)

        # max output of the actuators normalized (somewhat due to possible higher values caused by domain rand)
        self.min_output: float = 0
        self.max_output: float = 1.5
        self.max_output_shoulder_original = max_force_shoulder
        self.max_output_elbow_original = max_force_elbow
        self.max_output_shoulder = max_force_shoulder
        self.max_output_elbow = max_force_elbow
        self.actuator_domain_randomization_range = dr_actuator_range

        # max/min tremor torque values
        self.min_tremor_torque: float = -15
        self.max_tremor_torque: float = 15

        # action history
        self.second_prev_action = np.zeros(7)
        self.prev_action = np.zeros(7)

        # max/min values for actuator position vectors
        self.actuator_pos_min: float = -2
        self.actuator_pos_max: float = 2

        # define the observation space for the environment
        # observation space structure: actuator forces, tremor (t-2,t-1,t), act pos(t-1,t)
        low = np.concatenate([
            np.repeat(self.min_output, 14),
            np.repeat(self.min_tremor_torque, 12),
            np.repeat(self.actuator_pos_min, 54)
        ], axis=0).astype(np.float32)

        high = np.concatenate([
            np.repeat(self.max_output, 14),
            np.repeat(self.max_tremor_torque, 12),
            np.repeat(self.actuator_pos_max, 54)
        ], axis=0).astype(np.float32)

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.client = bc.BulletClient(connection_mode=p.DIRECT, options="--useGPGPU")
        self.client.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.client.resetSimulation()
        self.client.setGravity(0, 0, -9.81)  # for dynamic randomization this can be modified
        self.client.setTimeStep(self.dt)
        self.client.setRealTimeSimulation(0)  # Disable Real-Time Simulation, to speed up simulation

        # define the patients data
        self.exoskeleton_sim_model = ExoskeletonSimModel(client=self.client, dummy_shift_range=self.dummy_shift_max_range)

        # normalization values for the tremor torques
        self.shoulder_norm_val: float = 10
        self.elbow_norm_val: float = 5
        self.wrist_norm_val: float = 0.5

        # holds the position values for the actuators
        self.position_vectors: dict
        self.prev_position_vectors: dict

        # position values for the reference dummies
        self.ref_position_vectors = np.zeros((2, 3))
        self.ref_prev_position_vectors = np.zeros((2, 3))

        # matrices for tremor propagation
        self.I, self.D, self.S = seven_by_seven()
        self.matrix_noise_fraction = matrix_noise_fraction

        # torque values
        self.T_initial_values = np.zeros(14)
        self.first_harmonics_interval = first_harmonics_interval
        self.second_harmonics_interval = second_harmonics_interval

        # simulation boundary values
        self.shoulder_z_max_angle: float = 80
        self.shoulder_z_min_angle: float = -80
        self.shoulder_y_max_angle: float = 160.5
        self.shoulder_y_min_angle: float = -40
        self.shoulder_x_max_angle: float = 33.5
        self.shoulder_x_min_angle: float = -151.5
        self.elbow_y_max_angle: float = 150
        self.elbow_y_min_angle: float = -10
        self.elbow_z_max_angle: float = 80
        self.elbow_z_min_angle: float = -87

        self.low_bounds = np.array([self.shoulder_z_min_angle, self.shoulder_y_min_angle, self.shoulder_x_min_angle, self.elbow_y_min_angle])
        self.high_bounds = np.array([self.shoulder_z_max_angle, self.shoulder_y_max_angle, self.shoulder_x_max_angle, self.elbow_y_max_angle])

        # RL reward component weights
        self.weight_axis_reward: float = 0.5
        self.weight_torque_reward: float = 0.9
        self.weight_actuator_force_magnitude_reward: float = 0.05
        self.weight_actuator_smoothness_reward: float = 0.05
        self.weight_unwanted_component: float = 0.5

        self.max_reward = (np.sum(self.tremor_input_sequence) * self.weight_axis_reward + self.weight_torque_reward
                           + self.weight_actuator_force_magnitude_reward + self.weight_actuator_smoothness_reward +
                           self.weight_unwanted_component)

        # define a seed for the environment
        self.state = self.initialize_movement()

        # forego division by 0
        self.epsilon: float = 1e-10  # Small constant to avoid division by zero

    def get_torques(self, force_components):
        radius_vectors = self.exoskeleton_sim_model.get_radius_vectors()
        calculated_torques = {"actuator1": np.cross(force_components["actuator1"], radius_vectors["actuator1"]),
                              "actuator2": np.cross(force_components["actuator2"], radius_vectors["actuator2"]),
                              "actuator3": np.cross(force_components["actuator3"], radius_vectors["actuator3"]),
                              "actuator4": np.cross(force_components["actuator4"], radius_vectors["actuator4"]),
                              "actuator5": np.cross(force_components["actuator5"], radius_vectors["actuator5"]),
                              "actuator6": np.cross(force_components["actuator6"], radius_vectors["actuator6"]),
                              "actuator7": np.cross(force_components["actuator7"], radius_vectors["actuator7"])}

        return calculated_torques

    def seed(self, seed=None):
        np.random, seed = seeding.np_random(seed)
        return [seed]

    def initialize_movement(self):
        # load the correct reference motion in
        self.ep_state_values = define_empty_dict_for_env(self.max_count)

        # Generate one random value between 0.1 and 1 (domain randomization for amplitude values)
        max_range = self.tremor_amplitude_range[1] - self.tremor_amplitude_range[0]
        self.tremor_magnitude = self.tremor_amplitude_range[0] + (np.random.rand() * max_range)
        self.tremor_torque_values = generate_joint_torques_train(episode_length=self.max_count,
                                                                 torque_sequence=self.tremor_input_sequence,
                                                                 dt=self.dt,
                                                                 tremor_magnitude=self.tremor_magnitude,
                                                                 first_harmonics_interval=self.first_harmonics_interval,
                                                                 second_harmonics_interval=self.second_harmonics_interval)

        # domain randomization for differential equation matrices
        self.I_current = add_symmetric_noise(self.I, noise_fraction=self.matrix_noise_fraction)
        self.D_current = add_symmetric_noise(self.D, noise_fraction=self.matrix_noise_fraction)
        self.S_current = add_symmetric_noise(self.S, noise_fraction=self.matrix_noise_fraction)

        # create domain randomization for the actuator end effector shifts
        self.exoskeleton_sim_model.create_dummy_shift()

        # domain randomization for the actuator forces
        self.max_output_shoulder = self.max_output_shoulder_original * np.random.uniform(1 - self.actuator_domain_randomization_range, 1 + self.actuator_domain_randomization_range)
        self.max_output_elbow = self.max_output_elbow_original * np.random.uniform(1 - self.actuator_domain_randomization_range, 1 + self.actuator_domain_randomization_range)

        # Index in the loaded reference motion data (start from 2 for indexing reasons, and since past data is used in the obs)
        self.counts = 2

        # set the corresponding joint angles--- differential eq initial conditions in the matlab script
        self.ep_state_values["elbow_joint_y_positions"][:self.counts] = self.processed_imu_data["elbow_joint_y_positions"][:self.counts]
        self.ep_state_values["elbow_joint_z_positions"][:self.counts] = self.processed_imu_data["elbow_joint_z_positions"][:self.counts]
        self.ep_state_values["shoulder_joint_x_positions"][:self.counts] = self.processed_imu_data["shoulder_joint_x_positions"][:self.counts]
        self.ep_state_values["shoulder_joint_y_positions"][:self.counts] = self.processed_imu_data["shoulder_joint_y_positions"][:self.counts]
        self.ep_state_values["shoulder_joint_z_positions"][:self.counts] = self.processed_imu_data["shoulder_joint_z_positions"][:self.counts]

        # calculate previous position
        self.exoskeleton_sim_model.set_joint_position([self.ep_state_values["shoulder_joint_z_positions"][self.counts - 1] * (np.pi / 180),
                                                       self.ep_state_values["shoulder_joint_y_positions"][self.counts - 1] * (np.pi / 180),
                                                       self.ep_state_values["shoulder_joint_x_positions"][self.counts - 1] * (np.pi / 180),
                                                       self.ep_state_values["elbow_joint_y_positions"][self.counts - 1] * (np.pi / 180),
                                                       self.processed_imu_data["elbow_joint_z_positions"][self.counts - 1] * (np.pi / 180)])

        self.exoskeleton_sim_model.get_actuator_positions()
        self.prev_position_vectors = self.exoskeleton_sim_model.get_actuator_pos_vect()
        self.ref_prev_position_vectors = self.exoskeleton_sim_model.return_reference_dummy_pos()

        # set the exoskeleton position into order
        self.exoskeleton_sim_model.set_joint_position([self.ep_state_values["shoulder_joint_z_positions"][self.counts] * (np.pi / 180),
                                                       self.ep_state_values["shoulder_joint_y_positions"][self.counts] * (np.pi / 180),
                                                       self.ep_state_values["shoulder_joint_x_positions"][self.counts] * (np.pi / 180),
                                                       self.ep_state_values["elbow_joint_y_positions"][self.counts] * (np.pi / 180),
                                                       self.processed_imu_data["elbow_joint_z_positions"][self.counts] * (np.pi / 180)])

        # position vectors
        self.exoskeleton_sim_model.get_actuator_positions()
        self.position_vectors = self.exoskeleton_sim_model.get_actuator_pos_vect()
        self.ref_position_vectors = self.exoskeleton_sim_model.return_reference_dummy_pos()

        state = self.update_state_vector()
        # return the target motions values (tremor torque values are defined as zero since tremor cannot be in the first time step)
        return state

    def transform_action(self, actions):
        """
        Transforms the agents raw actions to the range of the actuators of the exoskeleton.
        :param actions: Raw actions of the agent actor network
        :return: Actuator forces
        """
        transformed_actions = np.zeros((7,))
        transformed_actions[:2] = ((actions[:2] + 1) / 2) * self.max_output_elbow
        transformed_actions[2:] = ((actions[2:] + 1) / 2) * self.max_output_shoulder

        return transformed_actions

    def control_reward(self, actions):
        """
        Calculates the control cost of the exoskeleton forces.
        Aims to promote control strategy with the least amount of force.
        :param actions: Actions in the range of the exoskeleton actuators
        :return: The actuator control reward
        """
        sum_actions = np.sum(actions)
        actuator_reward = np.exp(-(sum_actions / ((self.max_output_elbow + self.max_output_shoulder) / 2)) + self.epsilon)  # scaling is applied by division
        actuator_reward *= self.weight_actuator_force_magnitude_reward

        return actuator_reward

    def smoothness_reward(self, actions):
        """
        Calculates the smoothness of the exoskeleton forces.
        Aims to reduce jerky motion of the actuator to not damage the equipment and patient.
        :param actions: Actions in the range of the exoskeleton actuators
        :return: The actuator smoothness reward
        """
        sum_smoothness = np.mean((actions - 2 * self.prev_action + self.second_prev_action) ** 2)
        smoothness_reward = np.exp(-(sum_smoothness / ((self.max_output_elbow + self.max_output_shoulder) / 4)) + self.epsilon)  # scaling is applied by division
        reward_smoothness = self.weight_actuator_smoothness_reward * smoothness_reward

        return reward_smoothness

    def axis_reward(self, torque_values, tremor_torque):
        """
        Calculates the amount of axis in the arm where tremor reduction occurred.
        :param torque_values: Torque values present in the joints after the actuator forces have been applied.
        :param tremor_torque: Original torque value of the tremor
        :return: The axis reward
        """
        tremor_torque_reduction = (abs(torque_values) - abs(tremor_torque)) / (abs(tremor_torque) + self.epsilon) * 100
        tremor_torque_reduction = np.nan_to_num(tremor_torque_reduction, nan=0, posinf=0, neginf=0)
        axis_reward = np.sum(tremor_torque_reduction < 0)
        axis_reward *= self.weight_axis_reward

        return axis_reward

    def torque_reward(self, torque_values, tremor_torque):
        """
        Calculates the reward for the overall tremor reduction of the system.
        :param torque_values: Torque values present in the joints after the actuator forces have been applied.
        :param tremor_torque: Original torque value of the tremor.
        :return: tremor torque suppression reward
        """
        sum_torque = 0
        for index in range(4):
            if self.tremor_input_sequence[index] == 1:
                sum_torque += (abs(torque_values[index]) - abs(tremor_torque[index])) / abs(tremor_torque[index]) + 1

        torque_reward = np.exp((-sum_torque + self.epsilon) / self.tremor_axis_n)
        torque_reward *= self.weight_torque_reward

        return torque_reward

    def unwanted_reward(self, torque_values):
        """
        Calculates a reward based on how much the control strategy distorts the original movement.
        :param torque_values: Torque values present in the joints after the actuator forces have been applied.
        :return: The unwanted control reward
        """
        sum_unwanted = 0
        for index in range(4):
            if self.tremor_input_sequence[index] == 0:
                sum_unwanted += abs(torque_values[index])

        unwanted_reward = np.exp(-(sum_unwanted / ((self.max_output_elbow + self.max_output_shoulder) /
                                                   4 / self.tremor_axis_n)) + self.epsilon)
        unwanted_reward *= self.weight_unwanted_component
        return unwanted_reward

    def get_reward(self, action, torque_values, tremor_torque):
        """
        Calculates the reward function.
        :param action: Actions in the range of the exoskeleton actuators
        :param torque_values: Torque values present in the joints after the actuator forces have been applied.
        :param tremor_torque:
        :return: The reward function
        """
        reward_unwanted = self.unwanted_reward(torque_values)
        reward_torque = self.torque_reward(torque_values, tremor_torque)
        reward_axis = self.axis_reward(torque_values, tremor_torque)
        reward_control = self.control_reward(action)
        reward_smoothness = self.smoothness_reward(action)

        # sum of the reward function
        reward = (reward_axis + reward_torque + reward_smoothness + reward_control + reward_unwanted) / self.max_reward

        reward_info = {
            "reward_unwanted": reward_unwanted,
            "reward_torque": reward_torque,
            "reward_axis": reward_axis,
            "reward_control": reward_control,
            "reward_smoothness": reward_smoothness,
        }

        return reward, reward_info

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
        self.exoskeleton_sim_model.get_actuator_positions()  # update the actuator positions
        self.ref_prev_position_vectors = self.ref_position_vectors
        force_components = self.exoskeleton_sim_model.get_force_components(forces=action)  # a dict of (3,) vectors
        self.prev_position_vectors = self.position_vectors
        self.position_vectors = self.exoskeleton_sim_model.get_actuator_pos_vect()
        self.ref_position_vectors = self.exoskeleton_sim_model.return_reference_dummy_pos()
        ac_torque = self.get_torques(force_components)

        # for debugging
        # self.debug_exoskeleton_torque_feedback(action, actuator_torques)

        # which actuator influences which joint: (same order as diff eq matrix shoulder y,x,z elbow y
        actuator_torques = np.array(
            [ac_torque["actuator3"][1] + ac_torque["actuator4"][1] + ac_torque["actuator5"][1] + ac_torque["actuator7"][1] + ac_torque["actuator6"][1],
             ac_torque["actuator3"][0] + ac_torque["actuator4"][0] + ac_torque["actuator5"][0] + ac_torque["actuator7"][0] + ac_torque["actuator6"][0],
             ac_torque["actuator3"][2] + ac_torque["actuator4"][2] + ac_torque["actuator5"][2] + ac_torque["actuator7"][2] + ac_torque["actuator6"][2],
             abs(ac_torque["actuator1"][1]) - abs(ac_torque["actuator2"][1]),
             0, 0, 0
             ])

        # apply  tremor difference here
        tremor_torque = np.array(self.tremor_torque_values[:, self.counts])

        # return the torque values (shoulder y,x,z; elbow y)
        torque_values = np.array(tremor_torque + actuator_torques)

        # mitigated joint angle values
        act_tremor_joint_angles = solve_diff_eq(self.T_initial_values, [0, self.dt], self.I_current,
                                                self.D_current, self.S_current, torque_values)

        # unmitigated joint angle values
        tremor_joint_angles = solve_diff_eq(self.T_initial_values, [0, self.dt], self.I_current,
                                            self.D_current, self.S_current, tremor_torque)

        # convert to degrees
        act_tremor_joint_angles = act_tremor_joint_angles * (180 / np.pi)
        tremor_joint_angles = tremor_joint_angles * (180 / np.pi)

        # angles are made up from the previous angle and the change in person angle, tremor angle and actuator angles
        self.ep_state_values["shoulder_joint_z_positions"][self.counts] = self.processed_imu_data["shoulder_joint_z_positions"][self.counts] + act_tremor_joint_angles[2]
        self.ep_state_values["shoulder_joint_y_positions"][self.counts] = self.processed_imu_data["shoulder_joint_y_positions"][self.counts] + act_tremor_joint_angles[0]
        self.ep_state_values["shoulder_joint_x_positions"][self.counts] = self.processed_imu_data["shoulder_joint_x_positions"][self.counts] + act_tremor_joint_angles[1]
        self.ep_state_values["elbow_joint_y_positions"][self.counts] = self.processed_imu_data["elbow_joint_y_positions"][self.counts] + act_tremor_joint_angles[3]

        # set the simulation angles
        self.exoskeleton_sim_model.set_joint_position([self.ep_state_values["shoulder_joint_z_positions"][self.counts] * (np.pi / 180),
                                                       self.ep_state_values["shoulder_joint_y_positions"][self.counts] * (np.pi / 180),
                                                       self.ep_state_values["shoulder_joint_x_positions"][self.counts] * (np.pi / 180),
                                                       self.ep_state_values["elbow_joint_y_positions"][self.counts] * (np.pi / 180),
                                                       self.processed_imu_data["elbow_joint_z_positions"][self.counts] * (np.pi / 180)])

        self.client.stepSimulation()

        # define whether the simulation is done
        done = False

        # check movement boundaries
        self.check_movement_boundaries()

        # calculate the reward
        reward, reward_info = self.get_reward(action, torque_values, tremor_torque)

        # for debugging
        # print(actuator_torques)

        # increment the count of the simulation
        self.counts += 1

        # initialize the state vector
        state = self.update_state_vector()

        # set the actions to the prev and so forth
        self.second_prev_action = self.prev_action
        self.prev_action = action

        if self.counts >= self.max_count - 1:  # we have reached the end of the measurement in time
            done = True

        # DEBUG LINE: print the angle values
        # print(self.exoskeleton_sim_model.getJointPositions())

        truncated = False  # since we do not truncate the episodes
        info = {"actuator_torques": actuator_torques,
                "torque_val": torque_values,
                "ampl_val": act_tremor_joint_angles,
                "tremor_torque_val": tremor_torque,
                "tremor_ampl_val": tremor_joint_angles,
                **reward_info}

        return state, reward, done, truncated, info

    def reset(self):
        # reload the exo
        self.state = self.initialize_movement()
        score = self.counts

        return self.state, score

    def render(self, mode="human"):
        self.client = bc.BulletClient(connection_mode=p.GUI)  # for visual feedback use p.GUI

    def close(self):
        self.client.disconnect()
        print('Close the environment')

    def update_state_vector(self):
        state = np.array([self.ep_state_values["actuator_1_forces"][self.counts - 2] / self.max_output_elbow_original,
                          self.ep_state_values["actuator_2_forces"][self.counts - 2] / self.max_output_elbow_original,
                          self.ep_state_values["actuator_3_forces"][self.counts - 2] / self.max_output_shoulder_original,
                          self.ep_state_values["actuator_4_forces"][self.counts - 2] / self.max_output_shoulder_original,
                          self.ep_state_values["actuator_5_forces"][self.counts - 2] / self.max_output_shoulder_original,
                          self.ep_state_values["actuator_6_forces"][self.counts - 2] / self.max_output_shoulder_original,
                          self.ep_state_values["actuator_7_forces"][self.counts - 2] / self.max_output_shoulder_original,
                          self.ep_state_values["actuator_1_forces"][self.counts - 1] / self.max_output_elbow_original,
                          self.ep_state_values["actuator_2_forces"][self.counts - 1] / self.max_output_elbow_original,
                          self.ep_state_values["actuator_3_forces"][self.counts - 1] / self.max_output_shoulder_original,
                          self.ep_state_values["actuator_4_forces"][self.counts - 1] / self.max_output_shoulder_original,
                          self.ep_state_values["actuator_5_forces"][self.counts - 1] / self.max_output_shoulder_original,
                          self.ep_state_values["actuator_6_forces"][self.counts - 1] / self.max_output_shoulder_original,
                          self.ep_state_values["actuator_7_forces"][self.counts - 1] / self.max_output_shoulder_original,
                          self.tremor_torque_values[0, self.counts - 2] / self.shoulder_norm_val,
                          self.tremor_torque_values[1, self.counts - 2] / self.shoulder_norm_val,
                          self.tremor_torque_values[2, self.counts - 2] / self.shoulder_norm_val,
                          self.tremor_torque_values[3, self.counts - 2] / self.elbow_norm_val,
                          self.tremor_torque_values[0, self.counts - 1] / self.shoulder_norm_val,
                          self.tremor_torque_values[1, self.counts - 1] / self.shoulder_norm_val,
                          self.tremor_torque_values[2, self.counts - 1] / self.shoulder_norm_val,
                          self.tremor_torque_values[3, self.counts - 1] / self.elbow_norm_val,
                          self.tremor_torque_values[0, self.counts] / self.shoulder_norm_val,
                          self.tremor_torque_values[1, self.counts] / self.shoulder_norm_val,
                          self.tremor_torque_values[2, self.counts] / self.shoulder_norm_val,
                          self.tremor_torque_values[3, self.counts] / self.elbow_norm_val,
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
                          ], dtype=np.float32)

        return state

    def return_generated_tremor_data(self):
        # get the maximum value of each column of the generated tremors
        max_values = self.tremor_torque_values.max(axis=1)
        return self.tremor_input_sequence, max_values

    def return_max_length(self) -> int:
        return self.max_count

    def return_original_joint_angles(self) -> list:
        """
        :return: The original joint axes positions of the reference movement.
         (wrist positions set to 0 since no IMU sensors were used to measure wrist positions)
        """
        # returns the IMU measured values
        return [self.processed_imu_data["shoulder_joint_x_positions"][self.counts],
                self.processed_imu_data["shoulder_joint_y_positions"][self.counts],
                self.processed_imu_data["shoulder_joint_z_positions"][self.counts],
                self.processed_imu_data["elbow_joint_y_positions"][self.counts],
                self.processed_imu_data["elbow_joint_z_positions"][self.counts],
                0,
                0]

    def check_movement_boundaries(self) -> None:
        """
        Function that provides feedback if the agent's action causes an over extension in a joint axis.
        """
        if (self.shoulder_z_min_angle < self.ep_state_values["shoulder_joint_z_positions"][self.counts] < self.shoulder_z_max_angle) != 1:
            print("Former_Agent went over the movement boundaries Shoulder z.", self.ep_state_values["shoulder_joint_z_positions"][self.counts])
        elif (self.shoulder_y_min_angle < self.ep_state_values["shoulder_joint_y_positions"][self.counts] < self.shoulder_y_max_angle) != 1:
            print("Former_Agent went over the movement boundaries Shoulder y.", self.ep_state_values["shoulder_joint_y_positions"][self.counts])
        elif (self.shoulder_x_min_angle < self.ep_state_values["shoulder_joint_x_positions"][self.counts] < self.shoulder_x_max_angle) != 1:
            print("Former_Agent went over the movement boundaries Shoulder x.", self.ep_state_values["shoulder_joint_x_positions"][self.counts])
        elif (self.elbow_y_min_angle < self.ep_state_values["elbow_joint_y_positions"][self.counts] < self.elbow_y_max_angle) != 1:
            print("Former_Agent went over the movement boundaries elbow y.", self.ep_state_values["elbow_joint_y_positions"][self.counts])
