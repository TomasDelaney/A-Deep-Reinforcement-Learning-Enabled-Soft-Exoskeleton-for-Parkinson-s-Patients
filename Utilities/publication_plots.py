import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot_histogram(tremor_amplitudes, save_path, dpi=400):
    # Define bins for the histogram
    bins = np.arange(-1, 12) * 10

    # flip the values
    tremor_amplitudes = tremor_amplitudes * -1
    tremor_amplitudes = np.where(tremor_amplitudes == 0, -1, tremor_amplitudes)
    print("\n Max tremor suppression values: ", max(tremor_amplitudes), "\n")

    # Set Seaborn style and color palette
    sns.set(style="whitegrid")
    sns.set_context("talk")  # Adjusts the overall visual style
    sns.despine()

    # Create the histogram using Seaborn with specified bin colors
    plt.figure(figsize=(10, 6))  # Adjust the figure size
    sns.histplot(tremor_amplitudes, bins=bins, kde=False, edgecolor='black', alpha=0.7)

    # Add labels and title
    plt.xlabel('Suppression percentage', fontsize=14)
    plt.ylabel('Number of time steps', fontsize=14)
    plt.title('Histogram of Tremor Amplitude Suppression', fontsize=16)

    # Add grid for better readability
    plt.grid(axis='y', alpha=0.5)

    # Customize tick labels
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Save the plot with specified DPI
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0.1)


def plot_joint_torques(suppressed_torques, unsuppressed_torques, save_path, dpi=400):
    num_plots = 4

    num_plots = 4

    # Corresponding joint names
    joint_names = ["SFE", "SAA", "SEIR", "EFE"]

    # Set Seaborn style
    sns.set(style="whitegrid")
    sns.despine()

    for i in range(num_plots):
        # Create a new figure for each plot
        plt.figure(figsize=(8, 4))

        # Plot suppressed torques with blue color
        sns.lineplot(x=range(len(suppressed_torques)), y=suppressed_torques[:, i], label='Suppressed torques')

        # Plot unsuppressed torques with red color
        sns.lineplot(x=range(len(unsuppressed_torques)), y=unsuppressed_torques[:, i], label='Unsuppressed torques')

        # Add a horizontal dotted line at y=0
        plt.axhline(y=0, color='black', linestyle='--', linewidth=1)

        # Customize labels and title
        plt.xlabel('Time steps')
        plt.ylabel('Torque in Newtons')
        plt.title(f'Torque values present on joint {joint_names[i]}')

        # Set legend location to upper right
        plt.legend(loc='upper right')

        # Save the individual plot with specified DPI
        joint_save_path = f"{save_path}_{joint_names[i]}.png"
        plt.savefig(joint_save_path, dpi=dpi)


if __name__ == "__main__":
    # Example usage:
    # Replace 'your_tremor_amplitudes.npy' with the actual file or NumPy array
    tremor_amplitudes = np.random.randint(-100, 10, size=(1000,))
    save_path = 'sexy_tremor_amplitude_histogram_400dpi.png'
    plot_histogram(tremor_amplitudes, save_path, dpi=400)

    suppressed_torques = np.random.uniform(-1, 1, size=(100, 7))
    unsuppressed_torques = np.random.uniform(-1, 1, size=(100, 7))
    save_path = 'torque_plots_400dpi.png'
    plot_joint_torques(suppressed_torques, unsuppressed_torques, save_path, dpi=400)
