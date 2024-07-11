import numpy as np
import matplotlib.pyplot as plt


def generate_tremor_frequency(episode_length, dt=1/40):
    # frequency components
    first_frequency = np.random.uniform(low=4, high=6)
    second_frequency = np.random.uniform(low=8, high=10)
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


def generate_joint_torques_train(episode_length, all_seven, torque_sequence, dt):
    # calculates the generated torque values at each joint DoF from the generated accelerometer data
    # clear proximal-distal increase in the magnitude-> the further away the joint the bigger amplitude it has
    # use magnitude ratios to find the corresponding joint torque amplitudes
    # assumption frequencies and phases are equal(Davidson and Charles)
    # max tremor values: shoulder: (-10,10), elbow: (-5,5), wrist: (-0.5, 0.5) Nm
    input_seq = torque_sequence
    first_wave, second_wave, noise = generate_tremor_frequency(episode_length=episode_length, dt=dt)

    # safeguard to handle accidental cases where the last 3 arm joint axes are set to other than 0
    if not all_seven:
        input_seq[-3:] = np.array([0, 0, 0])

    torq_seq = []
    joint_max_values = np.array([10, 5, 2.5, 5, 5, 0.5, 0.5])  # From: Inverse dynamics modelling of upper-limb tremor, with cross-correlation analysis.

    for i, input_element in enumerate(input_seq):
        ampl_1, ampl_2 = generate_tremor_amplitude()
        generated_acc = ampl_1 * first_wave + ampl_2 * second_wave + noise
        torque_sequence = generated_acc * input_element
        torque_sequence = (-1 + 2 * (torque_sequence - min(torque_sequence)) / (max(torque_sequence) - min(torque_sequence))) * joint_max_values[i]
        torque_sequence = np.nan_to_num(torque_sequence, nan=0, posinf=0, neginf=0)
        torq_seq.append(torque_sequence)

    return np.array(torq_seq), input_seq


if __name__ == "__main__":
    # sample rates
    dt = 1 / 40  # 1 / sample rate!

    # generated torque data
    torque_values, _ = generate_joint_torques_train(episode_length=300, all_seven=False, torque_sequence=np.array([1, 1, 1, 1, 0, 0, 0]))
    print(torque_values.shape)
    for i in range(7):
        plt.subplot(7, 1, i + 1)
        plt.plot(np.linspace(0, torque_values[i].size * dt, torque_values[i].size), torque_values[i])

    plt.show()
