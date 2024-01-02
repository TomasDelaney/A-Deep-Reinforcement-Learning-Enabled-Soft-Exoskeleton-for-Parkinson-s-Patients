import numpy as np
import matplotlib.pyplot as plt


def plot_tremor_suppression(tremor_suppression, std, save_path=None, show=False):
    # plots the running average of 100 tremor suppression episodes and its standard deviations with the same sliding window
    time_steps = np.arange(len(tremor_suppression))

    # invert the tremor suppression values since (-) means it actually decreased the tremor
    tremor_suppression = tremor_suppression * -1

    plt.plot(time_steps, tremor_suppression, label='Tremor Suppression')
    plt.fill_between(time_steps, tremor_suppression - std, tremor_suppression + std, color='blue', alpha=0.3, label='Standard Deviation')

    plt.xlabel('Time Steps')
    plt.ylabel('Tremor Suppression (%)')
    plt.title('Average Tremor Suppression with Standard Deviation')
    plt.legend()
    plt.ylim(0, 100)  # Set the y-axis limits from 0 to 100%

    if save_path:
        plt.savefig(save_path, dpi=300)  # Save the plot as a PNG file
        print("Successfully saved the tremor suppression plot")

    if show:
        plt.show()

    plt.close()


def plot_tremor_suppression_episode(tremor_suppression, save_path=None, show=False):
    # find the first non-zero element in the original tremor
    original_start_indices = np.argmax(tremor_suppression[0, 0, :] != 0)

    # invert the tremor suppression values since (-) means it actually decreased the tremor
    tremor_suppression = tremor_suppression * -1

    # average across episodes
    average_tremor_suppression = np.mean(tremor_suppression, axis=0)

    # calculate the std
    std = np.std(tremor_suppression, axis=0)

    # plots the running average of 100 tremor suppression episodes and its standard deviations with the same sliding window
    time_steps = np.arange(len(average_tremor_suppression))

    plt.plot(time_steps, average_tremor_suppression[:, original_start_indices], label='Tremor csökkentés')
    plt.fill_between(time_steps, average_tremor_suppression[:, original_start_indices] - std[:, original_start_indices],
                     average_tremor_suppression[:, original_start_indices] + std[:, original_start_indices], color='blue', alpha=0.3, label='Szórás')

    plt.xlabel('Időpillanat')
    plt.ylabel('Tremor csökkentés (%)')
    plt.title('Tremor csökkentés az egyes időpillanatokban')
    plt.legend()

    plt.ylim(-1000, 100)  # Set y-axis limits

    if save_path:
        plt.savefig(save_path, dpi=300)  # Save the plot as a PNG file
        print("Successfully saved the tremor suppression plot")

    if show:
        plt.show()

    plt.close()


def plot_tremor_suppression_median(tremor_suppression, std, save_path=None, show=False):
    # plots the running average of 100 tremor suppression episodes and its standard deviations with the same sliding window
    time_steps = np.arange(len(tremor_suppression))

    # invert the tremor suppression values since (-) means it actually decreased the tremor
    tremor_suppression = tremor_suppression * -1

    plt.plot(time_steps, tremor_suppression, label='Tremor Suppression')
    plt.fill_between(time_steps, tremor_suppression - std, tremor_suppression + std, color='blue', alpha=0.3, label='Standard Deviation')

    plt.xlabel('Time Steps')
    plt.ylabel('Tremor Suppression (%)')
    plt.title('Median Tremor Suppression with Standard Deviation')
    plt.legend()
    plt.ylim(0, 100)  # Set the y-axis limits from 0 to 100%

    if save_path:
        plt.savefig(save_path, dpi=300)  # Save the plot as a PNG file
        print("Successfully saved the tremor suppression plot")

    if show:
        plt.show()

    plt.close()


def plot_algorithm_rewards(score, std, save_path=None, show=False):
    # plots the running average of 100 tremor suppression episodes and its standard deviations with the same sliding window
    time_steps = np.arange(len(score))

    plt.plot(time_steps, score, label='Tremor Suppression')
    plt.fill_between(time_steps, score - std, score + std, color='blue', alpha=0.3, label='Standard Deviation')

    plt.xlabel('Time Steps')
    plt.ylabel('Score in percent to maximum available score')
    plt.title('Score and Standard Deviation')
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=300)  # Save the plot as a PNG file
        print("Successfully saved the algorithm reward plot")

    if show:
        plt.show()

    plt.close()


def plot_tremor_suppression_histogram(data, save_path=None, display=False):
    # find the tremor suppression place
    original_start_indices = np.argmax(data[0, 0, :] != 0)

    tremor_data = data[:, :, original_start_indices]

    # Flatten the data array to a 1D array
    flattened_data = tremor_data.flatten() * -1

    # Create bins
    bins = np.concatenate(([-np.inf, 0], np.arange(10, 110, 10)))

    # Plot histogram
    plt.hist(flattened_data, bins=bins, edgecolor='black')
    plt.xlabel('Érték')
    plt.ylabel('Darab')
    plt.title('A tremor csökkentések értékei')

    # Compute and display the percentage of elements in each bin
    bin_counts, _ = np.histogram(flattened_data, bins=bins)
    total_count = len(flattened_data)

    # Compute the sum of percentages for bins with value > 0
    # percentage_sum = 0

    for bin_value, count in zip(bins, bin_counts):
        percentage = count / total_count * 100
        # percentage_sum += percentage
        plt.text(bin_value + 5, count, f'{percentage:.1f}%', ha='center')

    # Display bin values less than 0
    for bin_value, count in zip(bins, bin_counts):
        if bin_value < 0:
            plt.text(bin_value + 5, count, f'{bin_value}', ha='center', va='bottom')

    # Display the sum of percentages for bins with value > 0
    # plt.text(0, np.max(bin_counts), f'Effektív százalék: {percentage_sum:.1f}%', ha='left', va='bottom')

    # Save and/or display plot
    if save_path:
        plt.savefig(save_path)
        print("Successfully saved tremor histogram")
    if display:
        plt.show()

    # Close plot
    plt.close()


def plot_torque_difference(suppressed_tremor, original_tremor, save_path=None, display=False):
    # find the first non-zero element in the original tremor
    original_start_indices = np.argmax(original_tremor[0, :] != 0)

    # plot the index of it for the suppressed tremor and the original
    time_steps = np.arange(len(suppressed_tremor[:, original_start_indices]))

    plt.plot(time_steps, suppressed_tremor[:, original_start_indices], label='A csökkentett forgatónyomaték')
    plt.plot(time_steps, original_tremor[:, original_start_indices], label='Az eredeti forgatónyomaték')
    plt.fill_between(time_steps, suppressed_tremor[:, original_start_indices], original_tremor[:, original_start_indices],
                     where=original_tremor[:, original_start_indices] >= suppressed_tremor[:, original_start_indices], interpolate=True, color='skyblue', alpha=0.5)

    # Calculate and display the maximum difference between the two curves
    max_difference = np.max(original_tremor[:, original_start_indices] - suppressed_tremor[:, original_start_indices])
    max_difference_index = np.argmax(original_tremor[:, original_start_indices] - suppressed_tremor[:, original_start_indices])
    plt.annotate(f'Max Diff: {max_difference:.2f}', xy=(max_difference_index, original_tremor[:, original_start_indices][max_difference_index]),
                 xytext=(max_difference_index, original_tremor[:, original_start_indices][max_difference_index] + 0.5), arrowprops=dict(facecolor='black', arrowstyle='->'),
                 fontsize=10)

    # Style the plot with eye-catching features
    plt.xlabel('Időlépés')
    plt.ylabel('Forgatónyomaték különbség (Nm)')
    plt.title('Az eredeti és a csökkentett tremor forgatónyomatékának összehasonlítása')
    plt.legend()

    # Save and/or display plot
    if save_path:
        plt.savefig(save_path, dpi=300)
        print("Successfully plotted torque differences")
    if display:
        plt.show()

    # close plot
    plt.close()


def plot_actuator_forces(actuator_forces, save_path=None, display=False):
    # plot
    fig, axs = plt.subplots(7, 1, constrained_layout=True, figsize=(8, 24))
    fig.suptitle('Aktuátor erők', fontsize=16)

    x = np.linspace(1, actuator_forces.shape[0], actuator_forces.shape[0])
    colors = plt.cm.get_cmap('tab10')  # Choose a colormap

    for k in range(actuator_forces.shape[1]):
        axs[k].plot(x, actuator_forces[:, k], color=colors(k), linewidth=1, marker='o', label="Aktuátor " + str(k + 1) + " Erő")
        axs[k].set_xlabel('Időlépés')
        axs[k].set_ylabel('Erő értékek (N)')
        axs[k].legend()

    # Save and/or display plot
    if save_path:
        plt.savefig(save_path, dpi=300)
        print("Successfully actuator forces")
    if display:
        plt.show()

    # close plot
    plt.close()


if __name__ == "__main__":
    # 1: tremor suppression graph
    tremor_suppression = np.array([0.5, 0.4, 0.7, 0.9, 0.6, 0.8, 1.0, 0.1, 0.569, 0.7])
    tremor_suppression = tremor_suppression * -100
    print(tremor_suppression)

    std_values = []

    for i in range(len(tremor_suppression)):
        subset = tremor_suppression[:i + 1]
        std = np.std(subset)
        std_values.append(std)

    # Call the function to plot the tremor suppression
    plot_tremor_suppression(tremor_suppression, std=std_values)

    # 2: algorithm reward plot
    x = np.linspace(1, 100, 20)  # Generate 20 values evenly spaced between 1 and 100
    y = np.sqrt(x)  # Apply the square root function

    std_values_2 = []
    for i in range(len(y)):
        subset_2 = y[:i + 1]
        std = np.std(subset_2)
        std_values_2.append(std)

    print()

    plot_algorithm_rewards(score=y, std=std_values_2, show=True)

