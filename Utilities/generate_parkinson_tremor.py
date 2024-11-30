import numpy as np
import matplotlib.pyplot as plt


def generate_tremor_frequency(first_harmonics_interval: np.array,
                              second_harmonics_interval: np.array,
                              episode_length: int,
                              dt: float = 1 / 40):
    # frequency components
    first_frequency = np.random.uniform(low=first_harmonics_interval[0], high=first_harmonics_interval[1])
    second_frequency = np.random.uniform(low=second_harmonics_interval[0], high=second_harmonics_interval[1])
    white_noise = np.random.rand(episode_length) * 0.001

    # generation components
    generated_time_steps = np.linspace(0, episode_length * dt, episode_length)

    # waves
    first_wave = np.sin(2 * np.pi * first_frequency * generated_time_steps)
    second_wave = np.sin(2 * np.pi * second_frequency * generated_time_steps)

    return first_wave, second_wave, white_noise


def generate_tremor_amplitude():
    first_amplitude = 10 ** (np.random.uniform(low=-5, high=0) / 20)
    second_amplitude = 10 ** (np.random.uniform(low=-20, high=-10) / 20)

    return first_amplitude, second_amplitude


def generate_joint_torques_train(episode_length: int,
                                 torque_sequence: np.ndarray,
                                 dt: float,
                                 tremor_magnitude: float,
                                 first_harmonics_interval: np.array,
                                 second_harmonics_interval: np.array,
                                 ):
    """
    calculates the generated torque values at each joint DoF from the generated accelerometer data
    clear proximal-distal increase in the magnitude-> the further away the joint the bigger amplitude it has
    use magnitude ratios to find the corresponding joint torque amplitudes
    assumption frequencies and phases are equal(Davidson and Charles)
    max tremor values(based on: Inverse dynamics modelling of upper-limb tremor... Laurence P. et al. 2014)
        : shoulder z: (-10,10), shoulder y: (-5, 5), shoulder x: (-2.5, 2.5), elbow: (-5,5), wrist: (-0.5, 0.5) Nm
    :param episode_length: int which shows how many time steps an episode has
    :param torque_sequence: the sequence of the involved joint axes in the tremor
    :param dt: time between time steps in the simulation
    :param tremor_magnitude:
    :param first_harmonics_interval:
    :param second_harmonics_interval:
    :return: a numpy array of 7 with the given tremor torque in Nm at the correct time steps
    """
    first_wave, second_wave, noise = generate_tremor_frequency(episode_length=episode_length,
                                                               dt=dt,
                                                               first_harmonics_interval=first_harmonics_interval,
                                                               second_harmonics_interval=second_harmonics_interval)

    torque_seq = []
    joint_max_values = np.array([2.5, 5, 10, 5, 5, 0.5, 0.5]) * tremor_magnitude

    for i, input_element in enumerate(torque_sequence):
        ampl_1, ampl_2 = generate_tremor_amplitude()
        generated_acc = ampl_1 * first_wave + ampl_2 * second_wave + noise
        torque_sequence = generated_acc * input_element
        torque_sequence = ((-1 + 2 * (torque_sequence - min(torque_sequence)) / (max(torque_sequence) - min(torque_sequence)))
                           * joint_max_values[i])
        torque_sequence = np.nan_to_num(torque_sequence, nan=0, posinf=0, neginf=0)

        # also add change in the direction of the tremor, which way does it start
        torque_sequence *= np.random.choice([-1, 1], size=torque_sequence.shape)
        torque_seq.append(torque_sequence)

    return np.array(torque_seq)


if __name__ == "__main__":
    # sample rates
    dt_ = 1 / 40  # 1 / sample rate!

    # generated torque data
    torque_values = generate_joint_torques_train(episode_length=300,
                                                 torque_sequence=np.array([1, 1, 1, 1, 0, 0, 0]),
                                                 dt=dt_,
                                                 first_harmonics_interval=np.array([4, 6]),
                                                 second_harmonics_interval=np.array([8, 10]),
                                                 tremor_magnitude=1)
    print(torque_values.shape)
    for k in range(7):
        plt.subplot(7, 1, k + 1)
        plt.plot(np.linspace(0, torque_values[k].size * dt_, torque_values[k].size), torque_values[k])

    plt.show()
