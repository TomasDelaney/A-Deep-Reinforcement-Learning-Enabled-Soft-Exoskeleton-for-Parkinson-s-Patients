import numpy as np
import argparse
import sys
import time
from Environment.Exoskeleton_env import ExoskeletonEnv_train
from Agent.TD7_multi_agent import Agent
from Utilities.seed_setting_ import set_seeds
from Utilities.evaluation_functions_ import plot_algorithm_rewards
from Utilities.Train_logger_ import Logger
from Utilities.measure_script_time_ import seconds_to_hms
from Utilities.calculate_arm_end_effector_points import distance_3d, forward_kinematics


if __name__ == "__main__":
    # Record the start time
    start_time = time.time()
    np.set_printoptions(precision=3, floatmode='fixed')

    # arguments
    parser = argparse.ArgumentParser()
    # RL
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--n_steps", default=6e6, type=int)
    parser.add_argument("--warmup", default=25e3, type=int)
    # Physical simulation
    parser.add_argument("--num_reference_motions", default=8, type=int, help="Number of different movement trajectories")
    parser.add_argument("--use_all_dof", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--tremor_sequence", default=np.array([0, 1, 0, 1, 0, 0, 0]), help="In which joint axes are tremors present")
    # Tremor properties
    parser.add_argument("--first_harmonics_interval", default=np.array([4, 6]), help="Lower and higher end of the first harmonic wave's frequency")
    parser.add_argument("--second_harmonics_interval", default=np.array([8, 10]), help="Lower and higher end of the second harmonic wave's frequency")
    # Domain randomization properties
    parser.add_argument("--dr_actuator_end_pos_shift", default=0.02, type=float, help="Shift in the actuator end position coordinates in meters")
    parser.add_argument("--dr_actuator_range", default=0.03, type=float, help="Tolerance range for actuator precision")
    parser.add_argument("--dr_anatomical_matrix_noise", default=0.1, type=float, help="Amount of noise in the human anatomical joint matrices")
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
    logger = Logger("output.txt")

    # Redirect the standard output to the logger
    sys.stdout = logger

    # values for the simulation
    envs = []
    for i in range(args.num_reference_motions):
        env = ExoskeletonEnv_train(dr_actuator_end_pos_shift=args.dr_actuator_end_pos_shift,
                                   dr_actuator_range=args.dr_actuator_range,
                                   matrix_noise_fraction=args.dr_anatomical_matrix_noise,
                                   reference_motion_file_num=str(i),
                                   tremor_sequence=args.tremor_sequence,
                                   max_force_shoulder=args.max_force_shoulder,
                                   max_force_elbow=args.max_force_elbow,
                                   first_harmonics_interval=args.first_harmonics_interval,
                                   second_harmonics_interval=args.second_harmonics_interval)
        envs.append(env)

    # values for training
    steps_count = 0
    scores = []
    initial_score = 2
    agent_rew = []
    tremor_torque_suppression_individual_avg = []
    tremor_torque_suppression_individual_median = []
    agent = Agent(state_dim=envs[0].observation_space.shape[0], action_dim=envs[0].action_space.shape[0], max_action=1, learning_steps=args.n_steps,
                  env_num=len(envs))
    total_steps_for_ep = 0
    for i, _ in enumerate(envs):
        total_steps_for_ep += envs[i].return_max_length()
    load = False
    disregard = True
    allow_train = False

    # values for evaluation of the training
    avg_agent_rew = []
    std_score = []
    max_episode_lengths = [env.return_max_length() for env in envs]

    # global training variables
    median_tremor_sup = []

    # load the model if necessary
    if load:
        agent.load("Trained_Agents/[0,0,0,1]/[0,0,0,1]")

    # agent train variables
    observation = np.zeros((len(envs), envs[0].observation_space.shape[0]))
    observation_ = np.zeros((len(envs), envs[0].observation_space.shape[0]))
    score = np.zeros(len(envs))
    actions = np.zeros((len(envs), 7))
    torque_places = np.zeros((len(envs), 7))
    torque_maxes = np.zeros((len(envs), 7))

    # test the running of the environment
    while steps_count < args.n_steps:
        for i, env in enumerate(envs):
            observation[i], score[i] = env.reset()
            torque_places[i], torque_maxes[i] = env.return_generated_tremor_data()
        done = np.full((len(envs)), False, dtype=bool)
        ep_len = np.full((len(envs)), 1, dtype=int)
        tremor_reduction_full_ep = []
        tremor_ampl_reduction_full_ep = []
        tremor_ampl_total_reduction_full_ep = []
        tremor_when_reduction = np.zeros((len(envs), 2))
        tremor_when_ampl_reduction = np.zeros((len(envs), 2))
        tremor_reduction_in_episode = np.zeros((len(envs)))

        while not np.all(done):
            # choose action
            for i, obs in enumerate(observation):
                if not done[i]:
                    if allow_train:
                        actions[i] = agent.select_action(obs, use_checkpoint=False, use_exploration=True)
                    else:
                        actions[i] = np.random.uniform(-1, 1, 7)
                        actions[i] = np.clip(actions[i], -1, 1)

            # declare tremor containing vectors
            tremor_reduction_ep = np.zeros((len(envs), 7))
            tremor_ampl_reduction_ep = np.zeros((len(envs), 7))
            tremor_ampl_total_reduction_ep = np.zeros((len(envs)))

            # take step in the environment and store it
            for i, action in enumerate(actions):
                if not done[i]:
                    observation_[i], reward, done[i], _, info = envs[i].step(action)
                    agent.replay_buffer.add(observation[i], action, observation_[i], reward, done[i], tremor_num=i)

                    score[i] += reward
                    ep_len[i] += 1
                    steps_count += 1
                    observation[i] = observation_[i]

                    # tremor reduction metrics
                    tremor_reduction = (abs(info["torque_val"]) - abs(info["tremor_torque_val"])) / abs(info["tremor_torque_val"]) * 100
                    tremor_reduction = np.nan_to_num(tremor_reduction, nan=0, posinf=0, neginf=0)

                    # measure amplitude suppression in each joint axis angle
                    tremor_reduction_ampl = (abs(info["ampl_val"]) - abs(info["tremor_ampl_val"])) / abs(info["tremor_ampl_val"]) * 100
                    tremor_reduction_ampl = np.nan_to_num(tremor_reduction_ampl, nan=0, posinf=0, neginf=0)

                    # measure total amplitude reduction in the movement | amplitude values to be used as not angles but cm values
                    non_tremor_angle_values = np.radians(envs[i].return_original_joint_angles())
                    suppressed_angles = np.radians(info["ampl_val"]) + non_tremor_angle_values
                    unsuppressed_angles = np.radians(info["tremor_ampl_val"]) + non_tremor_angle_values

                    end_effector_position_joint = forward_kinematics(non_tremor_angle_values, args.humerus_length, args.forearm_length, args.hand_length)
                    end_effector_position1 = forward_kinematics(suppressed_angles, args.humerus_length, args.forearm_length, args.hand_length)
                    end_effector_position2 = forward_kinematics(unsuppressed_angles, args.humerus_length, args.forearm_length, args.hand_length)

                    distance_suppressed = distance_3d(end_effector_position_joint, end_effector_position1)  # Distance from the origin
                    distance_unsuppressed = distance_3d(end_effector_position_joint, end_effector_position2)  # Distance from the origin

                    # Calculate amplitude difference in percentage
                    tremor_reduction_ampl_total = ((distance_suppressed - distance_unsuppressed) / distance_unsuppressed) * 100

                    # count the occurrences of tremor reduction alongside the axis
                    tremor_when_reduction[i, 0] += np.sum(tremor_reduction[:4] >= 0)
                    tremor_when_reduction[i, 1] += np.sum(tremor_reduction[:4] < 0)

                    if np.any(tremor_reduction[:4] < 0):
                        tremor_reduction_in_episode[i] += 1

                    # measure tremor amplitude differences
                    if tremor_reduction_ampl_total < 0:
                        tremor_when_ampl_reduction[i, 1] += 1
                        tremor_ampl_total_reduction_ep[i] = tremor_reduction_ampl_total
                    else:
                        tremor_when_ampl_reduction[i, 0] += 1

                    # disregard none tremor suppression episodes
                    if disregard:
                        tremor_reduction[tremor_reduction > 0] = 0
                        tremor_reduction_ampl[tremor_reduction_ampl > 0] = 0
                        if tremor_reduction_ampl_total > 0:
                            tremor_reduction_ampl_total = 0

                    if len(np.nonzero(tremor_reduction)[0]) > 0:
                        tremor_reduction_ep[i] = tremor_reduction

                    if len(np.nonzero(tremor_reduction_ampl)[0]) > 0:
                        tremor_ampl_reduction_ep[i] = tremor_reduction_ampl
                else:
                    tremor_reduction_ep[i] = np.zeros(7)
                    tremor_ampl_reduction_ep[i] = np.zeros(7)

            # store the tremor suppression values
            tremor_reduction_full_ep.append(tremor_reduction_ep.copy())
            tremor_ampl_reduction_full_ep.append(tremor_ampl_reduction_ep.copy())
            tremor_ampl_total_reduction_full_ep.append(tremor_ampl_total_reduction_ep.copy())

        # using checkpoints so run when each episode terminates
        agent.maybe_train_and_checkpoint(ep_timesteps=round(np.mean(ep_len)), ep_return=np.mean(score))

        if steps_count > args.warmup:
            allow_train = True

        # calculate the metrics
        gotten_score = score - initial_score
        reward_percentage = gotten_score / ep_len * 100
        scores.append(score.copy())
        agent_rew.append(reward_percentage)
        avg_score = np.mean(scores, axis=0)
        median_score = np.median(scores, axis=0)
        avg_agent_r = np.mean(agent_rew[-100:], axis=0)
        median_agent_r = np.median(agent_rew[-100:], axis=0)
        avg_agent_rew.append(avg_agent_r)

        # tremor torque reduction metrics
        tremor_red_np = np.array(tremor_reduction_full_ep)
        full_episode_tremor_red_avg = np.zeros(len(envs))
        full_episode_tremor_red_median = np.zeros(len(envs))

        # tremor reduction amplitude metrics each joint axis
        tremor_red_ampl_np = np.array(tremor_ampl_reduction_full_ep)
        full_episode_tremor_red_ampl_avg = np.zeros(len(envs))
        full_episode_tremor_red_ampl_median = np.zeros(len(envs))

        # tremor reduction amplitude total metrics
        tremor_red_ampl_total_np = np.array(tremor_ampl_total_reduction_full_ep)
        full_episode_tremor_red_ampl_total_avg = np.zeros(len(envs))

        # generate console outputs -- local outputs of the envs
        for i, _ in enumerate(envs):
            # Tremor torque reduction metrics
            tremor_avg = np.mean(tremor_red_np[:max_episode_lengths[i], i], axis=0)
            tremor_median = np.median(tremor_red_np[:max_episode_lengths[i], i], axis=0)
            tremor_std = np.std(tremor_red_np[:max_episode_lengths[i], i], axis=0)
            tremor_occurrence = (tremor_when_reduction[i, 1] / sum(tremor_when_reduction[i])) * 100

            # Tremor amplitude reduction metrics
            tremor_ampl_avg = np.mean(tremor_red_ampl_np[:max_episode_lengths[i], i], axis=0)
            tremor_ampl_median = np.median(tremor_red_ampl_np[:max_episode_lengths[i], i], axis=0)
            tremor_ampl_std = np.std(tremor_red_ampl_np[:max_episode_lengths[i], i], axis=0)
            tremor_ampl_occurrence = (tremor_when_ampl_reduction[i, 1] / sum(tremor_when_ampl_reduction[i])) * 100

            # Overall tremor suppression metrics
            nonzero_tremor_red = tremor_red_np[:max_episode_lengths[i], i][tremor_red_np[:max_episode_lengths[i], i] != 0]
            tremor_sup_avg = np.mean(nonzero_tremor_red)
            tremor_sup_median = np.median(nonzero_tremor_red)

            # Total tremor amplitude suppression
            tremor_ampl_total_avg = np.mean(tremor_red_ampl_total_np[:max_episode_lengths[i], i])

            full_episode_tremor_red_avg[i] = tremor_sup_avg
            full_episode_tremor_red_median[i] = tremor_sup_median
            full_episode_tremor_red_ampl_avg[i] = np.mean(tremor_red_ampl_np[:max_episode_lengths[i], i, :][tremor_red_ampl_np[:max_episode_lengths[i], i, :] != 0], axis=None)
            full_episode_tremor_red_ampl_median = np.median(tremor_red_ampl_np[:max_episode_lengths[i], i, :][tremor_red_ampl_np[:max_episode_lengths[i], i, :] != 0], axis=None)

            # score global tremor suppression values
            tremor_torque_suppression_individual_avg.append(np.mean(tremor_red_np[:max_episode_lengths[i], i, :], axis=0))
            tremor_torque_suppression_individual_median.append((np.median(tremor_red_np[:max_episode_lengths[i], i, :], axis=0)))

            if args.print_each_env_data:
                print(f"\nReference movement {i} statistics:")

                # Basic statistics
                print(
                    f"\n Score: {score[i]:.3f}, Avg: {avg_score[i]:.3f}, Median: {median_score[i]:.3f},"
                    f" Max: {np.max(np.array(scores)[:, i]):.3f}, Std Dev: {np.std(np.array(scores)[-100:, i]):.3f}")
                print(
                    f"Reward achieved by the agent: {gotten_score[i]:.3f}, Score in % of max: {reward_percentage[i]:.3f},"
                    f" Avg rewards in %: {avg_agent_r[i]:.3f}, Median rewards in %: {median_agent_r[i]:.3f}")

                print(f"\nTremor Torque Reduction - Avg: {tremor_avg}, Median: {tremor_median}, Std Dev: {tremor_std}")
                print(f"Tremor reduction occurred in {tremor_occurrence:.2f}% of all time steps")

                print(f"\nTremor Amplitude Reduction - Avg: {tremor_ampl_avg}, Median: {tremor_ampl_median}, Std Dev: {tremor_ampl_std}")
                print(f"Tremor amplitude reduction occurred in {tremor_ampl_occurrence:.2f}% of all time steps")
                print(f"Overall Tremor Suppression - Avg: {tremor_sup_avg:.3f}%, Median: {tremor_sup_median:.3f}%")
                print(f"Total Tremor Amplitude Suppression: {tremor_ampl_total_avg:.3f}%")
                print(f"For torque places: {torque_places[i, :]}, Max Nm: {torque_maxes[i, :]}")

        # save the model
        agent.save("AGENT_NNS/test_agent")

        # generate global outputs
        print("\nGLOBAL TRAINING OUTPUTS: ")
        print("Steps count: ", int(steps_count))
        print("Average reward of the agent %.3f, Average score in percent to max: %.3f, avg rewards in percent: %.3f, median rewards in percent: %.3f" %
              (np.mean(gotten_score), np.mean(reward_percentage), np.mean(avg_agent_r), np.median(avg_agent_r)))
        print()
        print("Tremor torque reduction metrics:")
        print('Tremor reduction occurred in', (np.sum(tremor_when_reduction[:, 1]) / (np.sum(tremor_when_reduction[:, 0]) + np.sum(tremor_when_reduction[:, 1]))) * 100,
              '% of all the generated tremor axis in all time steps')
        print("Some sort of tremor reduction occurred along any axis in",
              np.sum(tremor_reduction_in_episode[:]) / total_steps_for_ep * 100,
              "% of the episode")
        print("Average of the average tremor suppression across all movements", np.mean(tremor_torque_suppression_individual_avg[-100:], axis=0),
              "\nMedian of the average tremor suppression across all movements", np.median(tremor_torque_suppression_individual_avg[-100:], axis=0),
              "\nAverage of the median tremor suppression across all movements", np.mean(median_tremor_sup[-100:], axis=0),
              "\nMedian of the median tremor suppression across all movements", np.median(median_tremor_sup[-100:], axis=0),
              "\nOverall tremor suppression across all movements (avg)", np.mean(full_episode_tremor_red_avg),
              "\nOverall tremor suppression across all movements (median)", np.mean(full_episode_tremor_red_median))
        print()
        print("Tremor amplitude reduction metrics:")
        print("Overall angle tremor suppression across all movements and joint axis (avg)", np.mean(full_episode_tremor_red_ampl_avg), "%",
              "\nOverall angle tremor suppression across all movements and joint axis (median)", np.mean(full_episode_tremor_red_ampl_median), "%")
        print('Tremor amplitude reduction occurred in',
              (np.sum(tremor_when_ampl_reduction[:, 1]) / (np.sum(tremor_when_ampl_reduction[:, 0]) + np.sum(tremor_when_ampl_reduction[:, 1]))) * 100,
              '% of all time steps')
        print("Total average tremor amplitude suppression in the episode", np.mean(tremor_red_ampl_total_np[tremor_red_ampl_total_np[:, :] != 0], axis=None), "%")

        # save the standard deviations
        std_score.append(np.std(avg_agent_rew[-100:], axis=0))
        median_tremor_sup.append(np.mean(full_episode_tremor_red_median))

    # close the envs
    for env in envs:
        env.close()

    # generate evaluations
    for i, _ in enumerate(envs):
        plot_algorithm_rewards(score=np.array(agent_rew[:, i]), std=np.array(std_score[:, i]), save_path="figures/algo_score_env_" + str(i) + ".png")

    # Calculate the total time taken
    end_time = time.time()
    execution_time = end_time - start_time
    hours, minutes, seconds = seconds_to_hms(int(execution_time))
    print(f"Script executed in {hours} hours, {minutes} minutes, and {seconds} seconds.")

    # Restore the standard output and close the logger
    sys.stdout = logger.terminal
    logger.close()
