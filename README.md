# A-Deep-Reinforcement-Learning-Enabled-Soft-Exoskeleton-for-Parkinson-s-Patients

Official implementation for the code used in: Learning to Suppress Tremors: A Deep Reinforcement Learning-Enabled Soft Exoskeleton for Parkinson's Patients

## Usage:
### Agent:
The agent folder contains the modified version of the replay buffer to store multiple experiences
collected throughout multiple reference movements.
It also contains the TD7 agent with the detailed parameters present in the paper.

### Environment:
The environment contains the folder of the reference movements which hold the 8 distinct
recordings of 4 different dynamic movements.

It also contains the pybullet physical simulation environment used in the reinforcement
learning environment. With the reinforcement learning environment and the upper body / exoskeleton
urdf file.

### Simulation:

The simulation folder contains the code used for the training and evaluation of the exoskeleton
control's performance.

### Trained Agents:

The trained agents' folder contains the actor, critic and encoder neural networks used for
the evaluation of each tremor type.

### Utilities:

The utilities' folder contains various scripts used in the physical simulations joint angle and end
effector position calculations. As well as various scripts used for the evaluation and plotting of
the results.

### Evaluation logs:

The evaluation logs hold the logs used to determine the tremor suppressing effect of the exoskeleton.

## Software
### Results were originally collected with:
* Gym: 0.26.2
* Numpy: 1.23.5
* Pybullet: 3.2.5
* Python: 3.10
* Pytorch: 1.12.1
* SciPy: 1.11.4
