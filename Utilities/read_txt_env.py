import numpy as np
import matplotlib.pyplot as plt


def calculate_velocities(accelerations):
    """sample rate is defined as a constant here"""
    sample_rate = 40
    velocity = accelerations[0] * 1 / sample_rate
    vel_values = [velocity]

    for i in range(1, len(accelerations)):
        velocity = velocity + accelerations[i] * 1 / sample_rate
        vel_values.append(velocity)

    return np.array(vel_values)


def plot_txt_positions(text_dictionary):
    """sample rate is defined as a constant here"""
    sample_rate = 40
    d_t = 1 / sample_rate
    length = len(text_dictionary["elbow_joint_y_positions"])

    t = np.linspace(0, length * d_t, length)
    fig, axs = plt.subplots(2, 1, constrained_layout=True)
    fig.suptitle('Positions for the elbow and shoulder joints', fontsize=16)
    fig.set_size_inches(10.5, 10.5)

    # elbow positions
    axs[0].plot(t, text_dictionary["elbow_joint_y_positions"], color='red', linewidth=1, label='Positions around y '
                                                                                               'axis')
    axs[0].plot(t, text_dictionary["elbow_joint_z_positions"], color='blue', linewidth=1, label='Positions around z '
                                                                                                'axis')

    axs[0].set_title('Positions for the elbow joint')
    axs[0].set_xlabel('Time in seconds')
    axs[0].set_ylabel('Positions in deg')
    axs[0].legend()

    # shoulder joint
    axs[1].plot(t, text_dictionary["shoulder_joint_x_positions"], color='red', linewidth=1,
                label='Positions around x axis')
    axs[1].plot(t, text_dictionary["shoulder_joint_y_positions"], color='green', linewidth=1,
                label='Positions around y axis')
    axs[1].plot(t, text_dictionary["shoulder_joint_z_positions"], color='blue', linewidth=1,
                label='Positions around z axis')

    axs[1].set_title('Positions for the shoulder joint')
    axs[1].set_xlabel('Time in seconds')
    axs[1].set_ylabel('Positions in deg')
    axs[1].legend()


def define_empty_dict_for_env(size):
    # todo: define this function with empty numpy arrays
    return_dictionary = {"actuator_1_forces": np.zeros((size,)),
                         "actuator_2_forces": np.zeros((size,)),
                         "actuator_3_forces": np.zeros((size,)),
                         "actuator_4_forces": np.zeros((size,)),
                         "actuator_5_forces": np.zeros((size,)),
                         "actuator_6_forces": np.zeros((size,)),
                         "actuator_7_forces": np.zeros((size,)),
                         "elbow_joint_y_positions": np.zeros((size,)),
                         "elbow_joint_z_positions": np.zeros((size,)),
                         "shoulder_joint_x_positions": np.zeros((size,)),
                         "shoulder_joint_y_positions": np.zeros((size,)),
                         "shoulder_joint_z_positions": np.zeros((size,)),
                         "elbow_joint_x_acc": np.zeros((size,)),
                         "elbow_joint_y_acc": np.zeros((size,)),
                         "elbow_joint_z_acc": np.zeros((size,)),
                         "elbow_joint_x_vel": np.zeros((size,)),
                         "elbow_joint_y_vel": np.zeros((size,)),
                         "elbow_joint_z_vel": np.zeros((size,)),
                         "elbow_joint_x_ang_acc": np.zeros((size,)),
                         "elbow_joint_y_ang_acc": np.zeros((size,)),
                         "elbow_joint_z_ang_acc": np.zeros((size,)),
                         "elbow_joint_x_ang_vel": np.zeros((size,)),
                         "elbow_joint_y_ang_vel": np.zeros((size,)),
                         "elbow_joint_z_ang_vel": np.zeros((size,)),
                         "shoulder_joint_x_acc": np.zeros((size,)),
                         "shoulder_joint_y_acc": np.zeros((size,)),
                         "shoulder_joint_z_acc": np.zeros((size,)),
                         "shoulder_joint_x_vel": np.zeros((size,)),
                         "shoulder_joint_y_vel": np.zeros((size,)),
                         "shoulder_joint_z_vel": np.zeros((size,)),
                         "shoulder_joint_x_ang_acc": np.zeros((size,)),
                         "shoulder_joint_y_ang_acc": np.zeros((size,)),
                         "shoulder_joint_z_ang_acc": np.zeros((size,)),
                         "shoulder_joint_x_ang_vel": np.zeros((size,)),
                         "shoulder_joint_y_ang_vel": np.zeros((size,)),
                         "shoulder_joint_z_ang_vel": np.zeros((size,)),
                         }

    return return_dictionary


def get_min_max_env_values(env_dict):
    return_dict = [
        np.min(env_dict["elbow_joint_y_acc"]), np.max(env_dict["elbow_joint_y_acc"]),
        np.min(env_dict["shoulder_joint_x_acc"]), np.max(env_dict["shoulder_joint_x_acc"]),
        np.min(env_dict["shoulder_joint_y_acc"]), np.max(env_dict["shoulder_joint_y_acc"]),
        np.min(env_dict["shoulder_joint_z_acc"]), np.max(env_dict["shoulder_joint_z_acc"]),
        np.min(env_dict["elbow_joint_y_ang_acc"]), np.max(env_dict["elbow_joint_y_ang_acc"]),
        np.min(env_dict["shoulder_joint_x_ang_acc"]), np.max(env_dict["shoulder_joint_x_ang_acc"]),
        np.min(env_dict["shoulder_joint_y_ang_acc"]), np.max(env_dict["shoulder_joint_y_ang_acc"]),
        np.min(env_dict["shoulder_joint_z_ang_acc"]), np.max(env_dict["shoulder_joint_z_ang_acc"]),
        np.min(env_dict["elbow_joint_y_vel"]), np.max(env_dict["elbow_joint_y_vel"]),
        np.min(env_dict["shoulder_joint_x_vel"]), np.max(env_dict["shoulder_joint_x_vel"]),
        np.min(env_dict["shoulder_joint_y_vel"]), np.max(env_dict["shoulder_joint_y_vel"]),
        np.min(env_dict["shoulder_joint_z_vel"]), np.max(env_dict["shoulder_joint_z_vel"]),
        np.min(env_dict["elbow_joint_y_ang_vel"]), np.max(env_dict["elbow_joint_y_ang_vel"]),
        np.min(env_dict["shoulder_joint_x_ang_vel"]), np.max(env_dict["shoulder_joint_x_ang_vel"]),
        np.min(env_dict["shoulder_joint_y_ang_vel"]), np.max(env_dict["shoulder_joint_y_ang_vel"]),
        np.min(env_dict["shoulder_joint_z_ang_vel"]), np.max(env_dict["shoulder_joint_z_ang_vel"]),
    ]

    return return_dict


def read_env_texts(text_file_name):
    # todo : elbow get relative values aka subtract the shoulder values from the elbow values
    return_dictionary = {"elbow_joint_y_positions": [],
                         "elbow_joint_z_positions": [],
                         "shoulder_joint_x_positions": [],
                         "shoulder_joint_y_positions": [],
                         "shoulder_joint_z_positions": [],
                         "elbow_joint_x_acc": [],
                         "elbow_joint_y_acc": [],
                         "elbow_joint_z_acc": [],
                         "elbow_joint_x_vel": [],
                         "elbow_joint_y_vel": [],
                         "elbow_joint_z_vel": [],
                         "elbow_joint_x_ang_acc": [],
                         "elbow_joint_y_ang_acc": [],
                         "elbow_joint_z_ang_acc": [],
                         "elbow_joint_x_ang_vel": [],
                         "elbow_joint_y_ang_vel": [],
                         "elbow_joint_z_ang_vel": [],
                         "shoulder_joint_x_acc": [],
                         "shoulder_joint_y_acc": [],
                         "shoulder_joint_z_acc": [],
                         "shoulder_joint_x_vel": [],
                         "shoulder_joint_y_vel": [],
                         "shoulder_joint_z_vel": [],
                         "shoulder_joint_x_ang_acc": [],
                         "shoulder_joint_y_ang_acc": [],
                         "shoulder_joint_z_ang_acc": [],
                         "shoulder_joint_x_ang_vel": [],
                         "shoulder_joint_y_ang_vel": [],
                         "shoulder_joint_z_ang_vel": [],
                         }

    # read the lines from the file
    with open(text_file_name, 'r') as f:
        for line in f:
            return_dictionary["elbow_joint_y_positions"].append(float(line.split()[0]))
            return_dictionary["elbow_joint_z_positions"].append(float(line.split()[1]))
            return_dictionary["shoulder_joint_x_positions"].append(float(line.split()[2]))
            return_dictionary["shoulder_joint_y_positions"].append(float(line.split()[3]))
            return_dictionary["shoulder_joint_z_positions"].append(float(line.split()[4]))
            return_dictionary["elbow_joint_x_acc"].append(float(line.split()[5]))
            return_dictionary["elbow_joint_y_acc"].append(float(line.split()[6]))
            return_dictionary["elbow_joint_z_acc"].append(float(line.split()[7]))
            return_dictionary["shoulder_joint_x_acc"].append(float(line.split()[8]))
            return_dictionary["shoulder_joint_y_acc"].append(float(line.split()[9]))
            return_dictionary["shoulder_joint_z_acc"].append(float(line.split()[10]))
            return_dictionary["elbow_joint_x_ang_acc"].append(float(line.split()[11]))
            return_dictionary["elbow_joint_y_ang_acc"].append(float(line.split()[12]))
            return_dictionary["elbow_joint_z_ang_acc"].append(float(line.split()[13]))
            return_dictionary["shoulder_joint_x_ang_acc"].append(float(line.split()[14]))
            return_dictionary["shoulder_joint_y_ang_acc"].append(float(line.split()[15]))
            return_dictionary["shoulder_joint_z_ang_acc"].append(float(line.split()[16]))

    # convert the rest of the python lists to numpy arrays
    return_dictionary["elbow_joint_y_positions"] = np.array(return_dictionary["elbow_joint_y_positions"])
    return_dictionary["elbow_joint_z_positions"] = np.array(return_dictionary["elbow_joint_z_positions"])
    return_dictionary["shoulder_joint_x_positions"] = np.array(return_dictionary["shoulder_joint_x_positions"])
    return_dictionary["shoulder_joint_y_positions"] = np.array(return_dictionary["shoulder_joint_y_positions"])
    return_dictionary["shoulder_joint_z_positions"] = np.array(return_dictionary["shoulder_joint_z_positions"])
    return_dictionary["elbow_joint_x_acc"] = np.array(return_dictionary["elbow_joint_x_acc"])
    return_dictionary["elbow_joint_y_acc"] = np.array(return_dictionary["elbow_joint_y_acc"])
    return_dictionary["elbow_joint_z_acc"] = np.array(return_dictionary["elbow_joint_z_acc"])
    return_dictionary["shoulder_joint_x_acc"] = np.array(return_dictionary["shoulder_joint_x_acc"])
    return_dictionary["shoulder_joint_y_acc"] = np.array(return_dictionary["shoulder_joint_y_acc"])
    return_dictionary["shoulder_joint_z_acc"] = np.array(return_dictionary["shoulder_joint_z_acc"])
    return_dictionary["elbow_joint_x_ang_acc"] = np.array(return_dictionary["elbow_joint_x_ang_acc"])
    return_dictionary["elbow_joint_y_ang_acc"] = np.array(return_dictionary["elbow_joint_y_ang_acc"])
    return_dictionary["elbow_joint_z_ang_acc"] = np.array(return_dictionary["elbow_joint_z_ang_acc"])
    return_dictionary["shoulder_joint_x_ang_acc"] = np.array(return_dictionary["shoulder_joint_x_ang_acc"])
    return_dictionary["shoulder_joint_y_ang_acc"] = np.array(return_dictionary["shoulder_joint_y_ang_acc"])
    return_dictionary["shoulder_joint_z_ang_acc"] = np.array(return_dictionary["shoulder_joint_z_ang_acc"])

    # get the relative elbow values
    return_dictionary["elbow_joint_y_positions"] = np.array(return_dictionary["elbow_joint_y_positions"])
    return_dictionary["elbow_joint_z_positions"] = np.array(return_dictionary["elbow_joint_z_positions"])
    return_dictionary["elbow_joint_x_acc"] = np.array(return_dictionary["elbow_joint_x_acc"])
    return_dictionary["elbow_joint_y_acc"] = np.array(return_dictionary["elbow_joint_y_acc"])
    return_dictionary["elbow_joint_z_acc"] = np.array(return_dictionary["elbow_joint_x_acc"])
    return_dictionary["elbow_joint_x_ang_acc"] = np.array(return_dictionary["elbow_joint_x_ang_acc"])
    return_dictionary["elbow_joint_y_ang_acc"] = np.array(return_dictionary["elbow_joint_y_ang_acc"])
    return_dictionary["elbow_joint_z_ang_acc"] = np.array(return_dictionary["elbow_joint_z_ang_acc"])

    # calculate the velocity values based on the !!!!!THESE ARE ALL NUMPY ARRAYS!!!!!!!
    return_dictionary["elbow_joint_x_vel"] = calculate_velocities(return_dictionary["elbow_joint_x_acc"])
    return_dictionary["elbow_joint_y_vel"] = calculate_velocities(return_dictionary["elbow_joint_y_acc"])
    return_dictionary["elbow_joint_z_vel"] = calculate_velocities(return_dictionary["elbow_joint_z_acc"])
    return_dictionary["elbow_joint_x_ang_vel"] = calculate_velocities(return_dictionary["elbow_joint_x_ang_acc"])
    return_dictionary["elbow_joint_y_ang_vel"] = calculate_velocities(return_dictionary["elbow_joint_y_ang_acc"])
    return_dictionary["elbow_joint_z_ang_vel"] = calculate_velocities(return_dictionary["elbow_joint_z_ang_acc"])
    return_dictionary["shoulder_joint_x_vel"] = calculate_velocities(return_dictionary["shoulder_joint_x_acc"])
    return_dictionary["shoulder_joint_y_vel"] = calculate_velocities(return_dictionary["shoulder_joint_y_acc"])
    return_dictionary["shoulder_joint_z_vel"] = calculate_velocities(return_dictionary["shoulder_joint_z_acc"])
    return_dictionary["shoulder_joint_x_ang_vel"] = calculate_velocities(return_dictionary["shoulder_joint_x_ang_acc"])
    return_dictionary["shoulder_joint_y_ang_vel"] = calculate_velocities(return_dictionary["shoulder_joint_y_ang_acc"])
    return_dictionary["shoulder_joint_z_ang_vel"] = calculate_velocities(return_dictionary["shoulder_joint_z_ang_acc"])

    return return_dictionary


if __name__ == "__main__":
    info = read_env_texts("../measurement_to_env_texts/measurement1_to_env.txt")
    plot_txt_positions(info)
    plt.show()
    print(info)
