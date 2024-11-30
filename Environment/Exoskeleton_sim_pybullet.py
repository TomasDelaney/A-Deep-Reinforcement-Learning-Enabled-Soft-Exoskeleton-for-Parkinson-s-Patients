import pybullet as p
import math
import numpy as np


class ExoskeletonSimModel:
    def __init__(self, client, dummy_shift_range: float):
        """
        Pybullet simulation model, that models the exoskeleton-human skeleton interactions.
        :param client: The Pybullet client where the simulation is running
        :param dummy_shift_range: The range of possible shift in each coordinate of an actuator end position
        """
        super(self.__class__, self).__init__()
        file_name = "../Simulation/exo_v3.urdf"
        self.file_path = file_name

        self.client = client
        self.exo = self.client.loadURDF(fileName=self.file_path, basePosition=[0, 0, 0.1], physicsClientId=client._client, useFixedBase=True)

        # human skeleton joint handles
        self.shoulder_z_joint_handle = 0
        self.shoulder_y_joint_handle = 1
        self.shoulder_x_joint_handle = 2
        self.elbow_y_joint_handle = 3
        self.elbow_z_joint_handle = 4
        self.human_joints = [self.shoulder_z_joint_handle, self.shoulder_y_joint_handle, self.shoulder_x_joint_handle, self.elbow_y_joint_handle, self.elbow_z_joint_handle]

        # values for the minimum and maximum joint angles
        self.shoulder_z_max_angle = 80
        self.shoulder_z_min_angle = -80
        self.shoulder_y_max_angle = 160.5
        self.shoulder_y_min_angle = -40
        self.shoulder_x_max_angle = 45
        self.shoulder_x_min_angle = -151.5
        self.elbow_y_max_angle = 150
        self.elbow_y_min_angle = -2
        self.elbow_z_max_angle = 80
        self.elbow_z_min_angle = -87

        # human skeleton joint initial toques
        self.shoulder_z_initial_torq = 40
        self.shoulder_y_initial_torq = 80
        self.shoulder_x_initial_torq = 50
        self.elbow_y_initial_torq = 25
        self.elbow_z_initial_torq = 50

        # actuator dummy handles
        self.act11_handle = 9
        self.act12_handle = 5
        self.act21_handle = 12
        self.act22_handle = 6
        self.act31_handle = 15
        self.act32_handle = 8
        self.act41_handle = 17
        self.act42_handle = 11
        self.act51_handle = 14
        self.act52_handle = 7
        self.act61_handle = 18
        self.act62_handle = 13
        self.act71_handle = 16
        self.act72_handle = 10
        self.act_shoulder_handle = 0
        self.act_elbow_handle = 3

        # actuator dummy positions
        self.act11_positions = np.zeros(3, )
        self.act12_positions = np.zeros(3, )
        self.act21_positions = np.zeros(3, )
        self.act22_positions = np.zeros(3, )
        self.act31_positions = np.zeros(3, )
        self.act32_positions = np.zeros(3, )
        self.act41_positions = np.zeros(3, )
        self.act42_positions = np.zeros(3, )
        self.act51_positions = np.zeros(3, )
        self.act52_positions = np.zeros(3, )
        self.act61_positions = np.zeros(3, )
        self.act62_positions = np.zeros(3, )
        self.act71_positions = np.zeros(3, )
        self.act72_positions = np.zeros(3, )
        self.act_shoulder_reference_positions = np.zeros(3, )
        self.act_elbow_reference_positions = np.zeros(3, )

        # human skeleton handles
        self.humerus_handle = None
        self.forearm_handle = None

        # position vectors for the actuators
        self.shoulder_z_joint_pos_vect = np.zeros(3)
        self.shoulder_y_joint_pos_vect = np.zeros(3)
        self.shoulder_x_joint_pos_vect = np.zeros(3)
        self.elbow_y_joint_pos_vect = np.zeros(3)
        self.elbow_z_joint_pos_vect = np.zeros(3)

        # dummy shift coordinates
        self.dummy_shift_range = dummy_shift_range
        self.dummy_shift_coordinates = np.zeros((14, 3))

    def create_dummy_shift(self):
        """
        Adds domain randomization to the shift in the Velcro fixed actuator end positions
        :return: initialises the coordinates of this shift at the beginning of the episode
        """
        self.dummy_shift_coordinates = np.random.uniform(
            low=-self.dummy_shift_range,
            high=self.dummy_shift_range,
            size=(14, 3)
        )

    def set_joint_position(self, position) -> None:
        """
        Function that sets the joint positions in the Pybullet simulation
        :param position:
        """
        # set the joint positions for the human joints
        self.client.setJointMotorControlArray(self.exo, self.human_joints,
                                              controlMode=p.POSITION_CONTROL,
                                              targetPositions=position)  # physicsClientId=self.exo

    def get_actuator_positions(self) -> None:
        """
        Function that updates the actuator dummy positions

        :return:
            np.ndarray: in 14x3 shape, where 14 end effector points of the actuators (2 for each) are described by their
            x, y and z coordinates
        """
        # returns the coordinates of the actuator end point positions
        actuator_positions = np.zeros((14, 3))
        actuator_positions[0] = self.client.getLinkState(self.exo, linkIndex=self.act11_handle)[0]
        actuator_positions[1] = self.client.getLinkState(self.exo, linkIndex=self.act12_handle)[0]
        actuator_positions[2] = self.client.getLinkState(self.exo, linkIndex=self.act21_handle)[0]
        actuator_positions[3] = self.client.getLinkState(self.exo, linkIndex=self.act22_handle)[0]
        actuator_positions[4] = self.client.getLinkState(self.exo, linkIndex=self.act31_handle)[0]
        actuator_positions[5] = self.client.getLinkState(self.exo, linkIndex=self.act32_handle)[0]
        actuator_positions[6] = self.client.getLinkState(self.exo, linkIndex=self.act41_handle)[0]
        actuator_positions[7] = self.client.getLinkState(self.exo, linkIndex=self.act42_handle)[0]
        actuator_positions[8] = self.client.getLinkState(self.exo, linkIndex=self.act51_handle)[0]
        actuator_positions[9] = self.client.getLinkState(self.exo, linkIndex=self.act52_handle)[0]
        actuator_positions[10] = self.client.getLinkState(self.exo, linkIndex=self.act61_handle)[0]
        actuator_positions[11] = self.client.getLinkState(self.exo, linkIndex=self.act62_handle)[0]
        actuator_positions[12] = self.client.getLinkState(self.exo, linkIndex=self.act71_handle)[0]
        actuator_positions[13] = self.client.getLinkState(self.exo, linkIndex=self.act72_handle)[0]

        # set the act positions to the corresponding values
        self.act11_positions = actuator_positions[0] + self.dummy_shift_coordinates[0]
        self.act12_positions = actuator_positions[1] + self.dummy_shift_coordinates[1]
        self.act21_positions = actuator_positions[2] + self.dummy_shift_coordinates[2]
        self.act22_positions = actuator_positions[3] + self.dummy_shift_coordinates[3]
        self.act31_positions = actuator_positions[4] + self.dummy_shift_coordinates[4]
        self.act32_positions = actuator_positions[5] + self.dummy_shift_coordinates[5]
        self.act41_positions = actuator_positions[6] + self.dummy_shift_coordinates[6]
        self.act42_positions = actuator_positions[7] + self.dummy_shift_coordinates[7]
        self.act51_positions = actuator_positions[8] + self.dummy_shift_coordinates[8]
        self.act52_positions = actuator_positions[9] + self.dummy_shift_coordinates[9]
        self.act61_positions = actuator_positions[10] + self.dummy_shift_coordinates[10]
        self.act62_positions = actuator_positions[11] + self.dummy_shift_coordinates[11]
        self.act71_positions = actuator_positions[12] + self.dummy_shift_coordinates[12]
        self.act72_positions = actuator_positions[13] + self.dummy_shift_coordinates[13]

    def get_actuator_pos_vect(self) -> dict:
        return_pos_vectors = {"actuator1": np.array([(self.act12_positions[0] - self.act_elbow_reference_positions[0]),
                                                     (self.act12_positions[1] - self.act_elbow_reference_positions[1]),
                                                     (self.act12_positions[2] - self.act_elbow_reference_positions[2]),
                                                     ]),
                              "actuator2": np.array([(self.act22_positions[0] - self.act_elbow_reference_positions[0]),
                                                     (self.act22_positions[1] - self.act_elbow_reference_positions[1]),
                                                     (self.act22_positions[2] - self.act_elbow_reference_positions[2]),
                                                     ]),
                              "actuator3": np.array([(self.act32_positions[0] - self.act_shoulder_reference_positions[0]),
                                                     (self.act32_positions[1] - self.act_shoulder_reference_positions[1]),
                                                     (self.act32_positions[2] - self.act_shoulder_reference_positions[2]),
                                                     ]),
                              "actuator4": np.array([(self.act42_positions[0] - self.act_shoulder_reference_positions[0]),
                                                     (self.act42_positions[1] - self.act_shoulder_reference_positions[1]),
                                                     (self.act42_positions[2] - self.act_shoulder_reference_positions[2]),
                                                     ]),
                              "actuator5": np.array([(self.act52_positions[0] - self.act_shoulder_reference_positions[0]),
                                                     (self.act52_positions[1] - self.act_shoulder_reference_positions[1]),
                                                     (self.act52_positions[2] - self.act_shoulder_reference_positions[2]),
                                                     ]),
                              "actuator6": np.array([(self.act62_positions[0] - self.act_shoulder_reference_positions[0]),
                                                     (self.act62_positions[1] - self.act_shoulder_reference_positions[1]),
                                                     (self.act62_positions[2] - self.act_shoulder_reference_positions[2]),
                                                     ]),
                              "actuator7": np.array([(self.act72_positions[0] - self.act_shoulder_reference_positions[0]),
                                                     (self.act72_positions[1] - self.act_shoulder_reference_positions[1]),
                                                     (self.act72_positions[2] - self.act_shoulder_reference_positions[2]),
                                                     ]),
                              }
        return return_pos_vectors

    def get_radius_vectors(self) -> dict:
        # update reference dummy positions
        self.act_shoulder_reference_positions = self.client.getLinkState(self.exo, linkIndex=self.act_shoulder_handle)[0]
        self.act_elbow_reference_positions = self.client.getLinkState(self.exo, linkIndex=self.act_elbow_handle)[0]

        radius_vectors = {"actuator1": self.act_elbow_reference_positions - self.act12_positions,
                          "actuator2": self.act_elbow_reference_positions - self.act22_positions,
                          "actuator3": self.act_shoulder_reference_positions - self.act32_positions,
                          "actuator4": self.act_shoulder_reference_positions - self.act42_positions,
                          "actuator5": self.act_shoulder_reference_positions - self.act52_positions,
                          "actuator6": self.act_shoulder_reference_positions - self.act62_positions,
                          "actuator7": self.act_shoulder_reference_positions - self.act72_positions}

        return radius_vectors

    def get_force_components(self, forces: np.array) -> dict:
        # returns the force components of the actuators
        # since actuator 1 and 2 actuate the elbow joint they only return two values: elbow y joint value and elbow z
        # 4 sides are required to get alfa and beta in the triangles.
        # forces is a vector containing the force value each actuator outputs
        shift_value = 5  # value to deal with inconsistent global coordinates
        force_components = {"actuator1": np.zeros(3),
                            "actuator2": np.zeros(3),
                            "actuator3": np.zeros(3),
                            "actuator4": np.zeros(3),
                            "actuator5": np.zeros(3),
                            "actuator6": np.zeros(3),
                            "actuator7": np.zeros(3)}

        # actuator1
        theta_x = math.atan2(((self.act12_positions[1] + shift_value) - (self.act11_positions[1] + shift_value)),
                             ((self.act12_positions[0] + shift_value) - (self.act11_positions[0] + shift_value)))
        theta_y = math.atan2(((self.act12_positions[0] + shift_value) - (self.act11_positions[0] + shift_value)),
                             ((self.act12_positions[1] + shift_value) - (self.act11_positions[1] + shift_value)))
        theta_z = math.atan2(((self.act12_positions[2] + shift_value) - (self.act11_positions[2] + shift_value)),
                             ((self.act12_positions[0] + shift_value) - (self.act11_positions[0] + shift_value)))
        force_components["actuator1"][0] = np.cos(theta_x) * forces[0]
        force_components["actuator1"][1] = np.cos(theta_y) * forces[0]
        force_components["actuator1"][2] = np.cos(theta_z) * forces[0]

        # actuator2
        theta_x = math.atan2(((self.act22_positions[1] + shift_value) - (self.act21_positions[1] + shift_value)),
                             ((self.act22_positions[0] + shift_value) - (self.act21_positions[0] + shift_value)))
        theta_y = math.atan2(((self.act22_positions[0] + shift_value) - (self.act21_positions[0] + shift_value)),
                             ((self.act22_positions[1] + shift_value) - (self.act21_positions[1] + shift_value)))
        theta_z = math.atan2(((self.act22_positions[2] + shift_value) - (self.act21_positions[2] + shift_value)),
                             ((self.act22_positions[0] + shift_value) - (self.act21_positions[0] + shift_value)))
        force_components["actuator2"][0] = np.cos(theta_x) * forces[1]
        force_components["actuator2"][1] = np.cos(theta_y) * forces[1]
        force_components["actuator2"][2] = np.cos(theta_z) * forces[1]

        # actuator 3
        theta_x = math.atan2(((self.act32_positions[1] + shift_value) - (self.act31_positions[1] + shift_value)),
                             ((self.act32_positions[0] + shift_value) - (self.act31_positions[0] + shift_value)))
        theta_y = math.atan2(((self.act32_positions[0] + shift_value) - (self.act31_positions[0] + shift_value)),
                             ((self.act32_positions[1] + shift_value) - (self.act31_positions[1] + shift_value)))
        theta_z = math.atan2(((self.act32_positions[2] + shift_value) - (self.act31_positions[2] + shift_value)),
                             ((self.act32_positions[0] + shift_value) - (self.act31_positions[0] + shift_value)))
        force_components["actuator3"][0] = np.cos(theta_x) * forces[2]
        force_components["actuator3"][1] = np.cos(theta_y) * forces[2]
        force_components["actuator3"][2] = np.cos(theta_z) * forces[2]

        # actuator4
        theta_x = math.atan2(((self.act42_positions[1] + shift_value) - (self.act41_positions[1] + shift_value)),
                             ((self.act42_positions[0] + shift_value) - (self.act41_positions[0] + shift_value)))
        theta_y = math.atan2(((self.act42_positions[0] + shift_value) - (self.act41_positions[0] + shift_value)),
                             ((self.act42_positions[1] + shift_value) - (self.act41_positions[1] + shift_value)))
        theta_z = math.atan2(((self.act42_positions[2] + shift_value) - (self.act41_positions[2] + shift_value)),
                             ((self.act42_positions[0] + shift_value) - (self.act41_positions[0] + shift_value)))
        force_components["actuator4"][0] = np.cos(theta_x) * forces[3]
        force_components["actuator4"][1] = np.cos(theta_y) * forces[3]
        force_components["actuator4"][2] = np.cos(theta_z) * forces[3]

        # actuator 5
        theta_x = math.atan2(((self.act52_positions[1] + shift_value) - (self.act51_positions[1] + shift_value)),
                             ((self.act52_positions[0] + shift_value) - (self.act51_positions[0] + shift_value)))
        theta_y = math.atan2(((self.act52_positions[0] + shift_value) - (self.act51_positions[0] + shift_value)),
                             ((self.act52_positions[1] + shift_value) - (self.act51_positions[1] + shift_value)))
        theta_z = math.atan2(((self.act52_positions[2] + shift_value) - (self.act51_positions[2] + shift_value)),
                             ((self.act52_positions[0] + shift_value) - (self.act51_positions[0] + shift_value)))
        force_components["actuator5"][0] = np.cos(theta_x) * forces[4]
        force_components["actuator5"][1] = np.cos(theta_y) * forces[4]
        force_components["actuator5"][2] = np.cos(theta_z) * forces[4]

        # actuator 6
        theta_x = math.atan2(((self.act62_positions[1] + shift_value) - (self.act61_positions[1] + shift_value)),
                             ((self.act62_positions[0] + shift_value) - (self.act61_positions[0] + shift_value)))
        theta_y = math.atan2(((self.act62_positions[0] + shift_value) - (self.act61_positions[0] + shift_value)),
                             ((self.act62_positions[1] + shift_value) - (self.act61_positions[1] + shift_value)))
        theta_z = math.atan2(((self.act62_positions[2] + shift_value) - (self.act61_positions[2] + shift_value)),
                             ((self.act62_positions[0] + shift_value) - (self.act61_positions[0] + shift_value)))
        force_components["actuator6"][0] = np.cos(theta_x) * forces[5]
        force_components["actuator6"][1] = np.cos(theta_y) * forces[5]
        force_components["actuator6"][2] = np.cos(theta_z) * forces[5]

        # actuator 7
        theta_x = math.atan2(((self.act72_positions[1] + shift_value) - (self.act71_positions[1] + shift_value)),
                             ((self.act72_positions[0] + shift_value) - (self.act71_positions[0] + shift_value)))
        theta_y = math.atan2(((self.act72_positions[0] + shift_value) - (self.act71_positions[0] + shift_value)),
                             ((self.act72_positions[1] + shift_value) - (self.act71_positions[1] + shift_value)))
        theta_z = math.atan2(((self.act72_positions[2] + shift_value) - (self.act71_positions[2] + shift_value)),
                             ((self.act72_positions[0] + shift_value) - (self.act71_positions[0] + shift_value)))
        force_components["actuator7"][0] = np.cos(theta_x) * forces[6]
        force_components["actuator7"][1] = np.cos(theta_y) * forces[6]
        force_components["actuator7"][2] = np.cos(theta_z) * forces[6]

        return force_components

    def get_actuator_pos_vect(self) -> dict:
        """
        Function that returns the r vector used in the observation vector

        :return:
            dict: of the actuators where each actuator has a different np.ndarray with the x,y and z components of the r vector.
        """
        return_pos_vectors = {"actuator1": np.array([(self.act12_positions[0] - self.act_elbow_reference_positions[0]),
                                                     (self.act12_positions[1] - self.act_elbow_reference_positions[1]),
                                                     (self.act12_positions[2] - self.act_elbow_reference_positions[2]),
                                                     ]),
                              "actuator2": np.array([(self.act22_positions[0] - self.act_elbow_reference_positions[0]),
                                                     (self.act22_positions[1] - self.act_elbow_reference_positions[1]),
                                                     (self.act22_positions[2] - self.act_elbow_reference_positions[2]),
                                                     ]),
                              "actuator3": np.array([(self.act32_positions[0] - self.act_shoulder_reference_positions[0]),
                                                     (self.act32_positions[1] - self.act_shoulder_reference_positions[1]),
                                                     (self.act32_positions[2] - self.act_shoulder_reference_positions[2]),
                                                     ]),
                              "actuator4": np.array([(self.act42_positions[0] - self.act_shoulder_reference_positions[0]),
                                                     (self.act42_positions[1] - self.act_shoulder_reference_positions[1]),
                                                     (self.act42_positions[2] - self.act_shoulder_reference_positions[2]),
                                                     ]),
                              "actuator5": np.array([(self.act52_positions[0] - self.act_shoulder_reference_positions[0]),
                                                     (self.act52_positions[1] - self.act_shoulder_reference_positions[1]),
                                                     (self.act52_positions[2] - self.act_shoulder_reference_positions[2]),
                                                     ]),
                              "actuator6": np.array([(self.act62_positions[0] - self.act_shoulder_reference_positions[0]),
                                                     (self.act62_positions[1] - self.act_shoulder_reference_positions[1]),
                                                     (self.act62_positions[2] - self.act_shoulder_reference_positions[2]),
                                                     ]),
                              "actuator7": np.array([(self.act72_positions[0] - self.act_shoulder_reference_positions[0]),
                                                     (self.act72_positions[1] - self.act_shoulder_reference_positions[1]),
                                                     (self.act72_positions[2] - self.act_shoulder_reference_positions[2]),
                                                     ]),
                              }
        return return_pos_vectors

    def return_reference_dummy_pos(self) -> np.ndarray:
        """
        Return the coordinates for the reference dummy vectors, showing the positions of the shoulder and elbow joints.

        Retrieves the current positions of the shoulder and elbow joints from the simulation and returns them as a NumPy array.

        Returns:
            np.ndarray: A 2x3 array where the first row contains the shoulder joint coordinates
                        and the second row contains the elbow joint coordinates.
        """
        self.act_shoulder_reference_positions = self.client.getLinkState(self.exo, linkIndex=self.act_shoulder_handle)[0]
        self.act_elbow_reference_positions = self.client.getLinkState(self.exo, linkIndex=self.act_elbow_handle)[0]

        return np.array([np.array(self.act_shoulder_reference_positions), np.array(self.act_elbow_reference_positions)])

    def load_again(self) -> None:
        """
        Function that reloads the scene of the simulation
        """
        self.exo = self.client.loadURDF(fileName=self.file_path,
                                        basePosition=[0, 0, 0.1],
                                        physicsClientId=self.client._client,
                                        useFixedBase=True)
