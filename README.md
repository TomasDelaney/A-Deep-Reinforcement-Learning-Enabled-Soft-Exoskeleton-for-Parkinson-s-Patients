<h1 align="center">
  Learning to Suppress Tremors:<br>
  A Deep Reinforcement Learning-Enabled Soft Exoskeleton for Parkinson's Patients
</h1>

<p align="center">
  <b>Tamás Endrei<sup>1,2*</sup>, Sándor Földi<sup>1,2</sup>, Ádám Makk<sup>3</sup>, and György Cserey<sup>1,2*</sup></b><br>
  <sup>1</sup>Faculty of Information Technology and Bionics, Pázmány Péter Catholic University, Budapest, Hungary<br>
  <sup>2</sup>Jedlik Innovation Ltd., Budapest, Hungary<br>
  <sup>3</sup>András Pető Faculty, Semmelweis University, Budapest, Hungary
</p>

Official implementation for the code used in:  
**[Learning to Suppress Tremors: A Deep Reinforcement Learning-Enabled Soft Exoskeleton for Parkinson's Patients](https://www.frontiersin.org/articles/10.3389/frobt.2025.1537470/full)**

---

![Simulation Process](https://github.com/TomasDelaney/A-Deep-Reinforcement-Learning-Enabled-Soft-Exoskeleton-for-Parkinson-s-Patients/raw/f688bda4fd42742b4361807146930cc67efa8a20/Images/Simulation_process.png)

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

## 📚 Citation

If you found this repository useful, please consider citing:

```bibtex
@ARTICLE{endrei2025learning,
  AUTHOR={Endrei, Tamás and Földi, Sándor and Makk, Ádám and Cserey, György},
  TITLE={Learning to suppress tremors: a deep reinforcement learning-enabled soft exoskeleton for Parkinson’s patients},
  JOURNAL={Frontiers in Robotics and AI},
  VOLUME={Volume 12 - 2025},
  YEAR={2025},
  URL={https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2025.1537470},
  DOI={10.3389/frobt.2025.1537470},
  ISSN={2296-9144},
}

