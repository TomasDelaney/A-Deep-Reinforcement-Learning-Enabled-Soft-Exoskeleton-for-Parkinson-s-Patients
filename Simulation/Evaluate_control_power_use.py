import numpy as np
import argparse
import sys
import time
from Environment.Exoskeleton_env import ExoskeletonEnv_train
from Agent.TD7_multi_agent_Pink_noise import Agent
from Utilities.seed_setting_ import set_seeds
from Utilities.Train_logger_ import Logger
from Utilities.measure_script_time_ import seconds_to_hms

if __name__ == "__main__":
    # Record the start time
    start_time = time.time()
    np.set_printoptions(precision=3, floatmode='fixed')

    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name", default="[1,0,1,0]_ac_control")
    # RL
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--num_eval_episodes", default=100, type=int)
    # Physical simulation
    parser.add_argument("--num_reference_motions", default=8, type=int, help="Number of different movement trajectories")
    parser.add_argument("--use_all_dof", default=False, action=argparse.BooleanOptionalAction)
    # Tremor properties
    parser.add_argument("--first_harmonics_interval", default=np.array([3.75, 6.25]), help="Lower and higher end of the first harmonic wave's frequency")
    parser.add_argument("--second_harmonics_interval", default=np.array([7.5, 12.5]), help="Lower and higher end of the second harmonic wave's frequency")
    parser.add_argument("--tremor_amplitude_range", default=np.array([0.95, 1.05]), help="Lower and higher end of the tremor magnitude scaling")
    parser.add_argument("--tremor_sequence", default=np.array([1, 0, 1, 0, 0, 0, 0]), help="In which joint axes are tremors present")
    # Domain randomization properties
    parser.add_argument("--dr_actuator_end_pos_shift", default=0.025, type=float, help="Shift in the actuator end position coordinates in meters")
    parser.add_argument("--dr_actuator_range", default=0.04, type=float, help="Tolerance range for actuator precision")
    parser.add_argument("--dr_anatomical_matrix_noise", default=0.125, type=float, help="Amount of noise in the human anatomical joint matrices")
    # Anatomical properties
    parser.add_argument("--humerus_length", default=0.4, type=float, help="Humerus length in meters")
    parser.add_argument("--humerus_radius", default=0.05, type=float, help="Humerus radius in meters")
    parser.add_argument("--forearm_length", default=0.4, type=float, help="Forearm length in meters")
    parser.add_argument("--forearm_radius", default=0.05, type=float, help="Forearm radius in meters")
    parser.add_argument("--hand_length", default=0.05, type=float, help="Hand length in meters")
    # Exoskeleton properties
    parser.add_argument("--max_force_elbow", default=20, type=float, help="Max force output of actuators 1,2 in Newtons")
    parser.add_argument("--max_force_shoulder", default=40, type=float, help="Max force output of actuators 3,4,5,6,7 in Newtons")
    # Feedback properties
    parser.add_argument("--print_each_env_data", default=False, action=argparse.BooleanOptionalAction, help="Print out the information for each individual movement")
    parser.add_argument("--print_tremor_data", default=False, action=argparse.BooleanOptionalAction, help="Print out the tremor reduction metrics")
    args = parser.parse_args()

    # set the seed
    set_seeds(args.seed)

    # Create an instance of the Logger class
    logger = Logger(f"./{args.file_name}_eval")

    # Redirect the standard output to the logger
    sys.stdout = logger

    # values for the simulation
    envs = []
    for i in range(args.num_reference_motions):
        env = ExoskeletonEnv_train(dr_actuator_end_pos_shift=args.dr_actuator_end_pos_shift,
                                   dr_actuator_range=args.dr_actuator_range,
                                   matrix_noise_fraction=args.dr_anatomical_matrix_noise,
                                   reference_motion_file_num=str(i),
                                   tremor_amplitude_range=args.tremor_amplitude_range,
                                   tremor_sequence=args.tremor_sequence,
                                   max_force_shoulder=args.max_force_shoulder,
                                   max_force_elbow=args.max_force_elbow,
                                   first_harmonics_interval=args.first_harmonics_interval,
                                   second_harmonics_interval=args.second_harmonics_interval)
        envs.append(env)

    # values for training
    steps_count = 0
    eps = 1e-10
    scores = []
    agent_rew = []
    agent = Agent(state_dim=envs[0].observation_space.shape[0], action_dim=envs[0].action_space.shape[0], max_action=1, learning_steps=1)
    total_steps_for_ep = 0
    for i, _ in enumerate(envs):
        total_steps_for_ep += envs[i].return_max_length() - 3
    load = True
    disregard = True

    # values for evaluation of the training
    avg_agent_rew = []
    std_score = []
    max_episode_lengths = [env.return_max_length() for env in envs]

    # load the model
    if load:
        agent.load(f"AGENT_NNS/[1,0,1,0]/[1,0,1,0]")

    # agent train variables
    observation = np.zeros((len(envs), envs[0].observation_space.shape[0]))
    observation_ = np.zeros((len(envs), envs[0].observation_space.shape[0]))
    score = np.zeros(len(envs))
    actions = np.zeros((len(envs), 7))
    torque_places = np.zeros((len(envs), 7))
    torque_maxes = np.zeros((len(envs), 7))

    # actuator forces
    ac_forces = [args.max_force_elbow, args.max_force_elbow, args.max_force_shoulder, args.max_force_shoulder,
                 args.max_force_shoulder, args.max_force_shoulder, args.max_force_shoulder]

    overall_forces = []
    overall_ac = []
    overall_actions = []
    overall_as = []

    # test the running of the environment
    for episode_count in range(args.num_eval_episodes):
        for i, env in enumerate(envs):
            observation[i], score[i] = env.reset()
            torque_places[i], torque_maxes[i] = env.return_generated_tremor_data()
        initial_score = 2
        done = np.full((len(envs)), False, dtype=bool)
        ep_len = np.full((len(envs)), 1, dtype=int)

        # actuator metrics
        all_actions_per_env = [[] for _ in range(len(envs))]
        actuators_action_smoothness = []
        actuators_forces_used = []

        while not np.all(done):
            # choose action
            actions = agent.select_action(observation, use_checkpoint=True, use_exploration=False)

            # take step in the environment and store it
            for i, action in enumerate(actions):
                if not done[i]:
                    observation_[i], reward, done[i], _, info = envs[i].step(action)

                    score[i] += reward
                    all_actions_per_env[i].append((actions[i] + 1) / 2 * ac_forces)
                    ep_len[i] += 1
                    steps_count += 1
                    observation[i] = observation_[i]

        # calculate the metrics
        gotten_score = score - initial_score
        reward_percentage = gotten_score / ep_len * 100
        scores.append(score.copy() / envs[0].max_reward)
        agent_rew.append(reward_percentage)
        avg_score = np.mean(scores, axis=0)
        median_score = np.median(scores, axis=0)
        avg_agent_r = np.mean(agent_rew[-100:], axis=0)
        median_agent_r = np.median(agent_rew[-100:], axis=0)
        avg_agent_rew.append(avg_agent_r)

        # generate console outputs -- local outputs of the envs
        ac_motor_environment = np.zeros((8, 7))
        print("LOCAL ENV OUTPUTS: ")
        for i, env in enumerate(envs):
            print("TREMOR", i, "statistics: ")
            print(' score %.3f' % score[i], 'avg score %.3f' % avg_score[i], 'median score %.3f' % median_score[i], 'max score %.3f' % np.max(np.array(scores)[:, i]),
                  'std of scores %.3f' % np.std(np.array(scores)[-100:, i], axis=0))
            print('Reward achieved by the agent: %.3f, Score in percent to max: %.3f, avg rewards in percent: %.3f, median rewards in percent: %.3f' %
                  (gotten_score[i], reward_percentage[i], avg_agent_r[i], median_agent_r[i]))

            action_traj = np.array(all_actions_per_env[i])  # shape: (ep_len, action_dim)

            # Reshape for smoothness calculation: (1, timesteps, action_dim) for axis compatibility
            action_traj_exp = action_traj[np.newaxis, :, :]

            # Action smoothness: 2nd derivative mean absolute diff across time, then across actions
            action_smoothness = np.mean(np.abs(np.mean(np.diff(action_traj_exp, n=2, axis=1), axis=2)))

            # action smoothness for each actuator
            ac_actuator = np.abs(np.mean(np.diff(all_actions_per_env[i], n=2, axis=0), axis=0))
            ac_motor_environment[i] = ac_actuator

            # Force used: sum of absolute values of all actuator outputs
            force_used = np.sum(np.abs(action_traj))

            actuators_action_smoothness.append(action_smoothness)
            actuators_forces_used.append(force_used)

            print(f"Env {i}: Action Smoothness = {action_smoothness:.3f}, Total Force Used = {force_used:.3f}")

        overall_as.append(np.mean(ac_motor_environment, axis=0))

        # generate global outputs
        print("GLOBAL TRAINING OUTPUTS: ")
        print("Episode count: ", episode_count)
        print("Average reward of the agent %.3f, Average score in percent to max: %.3f, avg rewards in percent: %.3f, median rewards in percent: %.3f" %
              (np.mean(gotten_score), np.mean(reward_percentage), np.mean(avg_agent_r), np.median(avg_agent_r)))
        print("Average Forces used", np.mean(actuators_forces_used))
        print("Average actuator smoothness", np.mean(actuators_action_smoothness))
        print()

        overall_forces.append(np.mean(actuators_forces_used))
        overall_ac.append(np.mean(actuators_action_smoothness))

        # For each environment, calculate the average force for each actuator across the steps
        avg_forces_per_env = [np.mean(env, axis=0) for env in all_actions_per_env]

        # append to global actions
        overall_actions.append(avg_forces_per_env)

    # print out the overall eval data (joint angle reduction on the axis, the % it occurred in, amplitude suppression)
    print()
    print("EVALUATION METRICS:")
    print("Average Forces used", np.mean(overall_forces))
    print("Average actuator smoothness", np.mean(overall_ac))

    # print the actuator wise metrics
    actuator_as = np.mean(np.array(overall_as), axis=0)
    episode_avg_forces = np.mean(np.mean(np.array(overall_actions), axis=1), axis=0)

    print("Average Forces used for each actuator", episode_avg_forces)
    print("Average actuator smoothness for each actuator", actuator_as)

    # agent.save_memory_buffer()
    for env in envs:
        env.close()

    # Record the end time
    end_time = time.time()

    # Calculate the total time taken
    execution_time = end_time - start_time

    # Convert to hours, minutes, and seconds
    hours, minutes, seconds = seconds_to_hms(int(execution_time))

    print(f"Script executed in {hours} hours, {minutes} minutes, and {seconds} seconds.")

    # Restore the standard output and close the logger
    sys.stdout = logger.terminal
    logger.close()
