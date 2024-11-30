import numpy as np
import argparse
import sys
import time
from Environment.Exoskeleton_env import ExoskeletonEnv_train
from Agent.TD7_multi_agent_Pink_noise import Agent
from Utilities.seed_setting_ import set_seeds
from Utilities.Train_logger_ import Logger
from Utilities.measure_script_time_ import seconds_to_hms
from Utilities.calculate_arm_end_effector_points import distance_3d, forward_kinematics
from Utilities.publication_plots import plot_histogram, plot_joint_torques, plot_joint_amplitudes
from Utilities.Violin_plot import tremor_violin_plot


if __name__ == "__main__":
    # Record the start time
    start_time = time.time()
    np.set_printoptions(precision=3, floatmode='fixed')

    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name", default="[0,0,0,1]")
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
    parser.add_argument("--tremor_sequence", default=np.array([0, 0, 0, 1, 0, 0, 0]), help="In which joint axes are tremors present")
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
    tremor_torque_suppression_individual_avg = []
    tremor_torque_suppression_individual_median = []
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

    # global training variables
    median_tremor_sup = []
    median_std_tremor_sup = []

    # load the model
    if load:
        agent.load(f"AGENT_NNS/{args.file_name}/{args.file_name}")

    # agent train variables
    observation = np.zeros((len(envs), envs[0].observation_space.shape[0]))
    observation_ = np.zeros((len(envs), envs[0].observation_space.shape[0]))
    score = np.zeros(len(envs))
    actions = np.zeros((len(envs), 7))
    torque_places = np.zeros((len(envs), 7))
    torque_maxes = np.zeros((len(envs), 7))

    # evaluation holders
    evaluation_percent = np.zeros(args.num_eval_episodes)
    evaluation_amplitude_percent = np.zeros(args.num_eval_episodes)
    evaluation_angle_amplitude_percent = np.zeros((args.num_eval_episodes, 7))
    evaluation_torque_suppression_some = np.zeros(args.num_eval_episodes)
    evaluation_torque_all_axis = np.zeros(args.num_eval_episodes)

    # variables for the histogram and the torque plots
    highest_score = 0
    tremor_amplitude_histogram_values = None
    violin_plot_values = [None, None, None, None]  # sfe, saa, efe, sei
    highest_scores = np.zeros(4)
    tremor_supp_torque_plot_values: np.ndarray
    tremor_unsup_torque_plot_values: np.ndarray
    tremor_supp_ampl_plot_values: np.ndarray
    tremor_unsup_ampl_plot_values: np.ndarray

    # test the running of the environment
    for episode_count in range(args.num_eval_episodes):
        for i, env in enumerate(envs):
            observation[i], score[i] = env.reset()
            torque_places[i], torque_maxes[i] = env.return_generated_tremor_data()
        initial_score = 2
        done = np.full((len(envs)), False, dtype=bool)
        ep_len = np.full((len(envs)), 1, dtype=int)

        tremor_reduction_full_ep = []
        tremor_ampl_reduction_full_ep = []
        tremor_ampl_total_reduction_full_ep = []
        tremor_torque_sup_plot_values_full_ep = []
        tremor_torque_unsup_plot_values_full_ep = []
        tremor_ampl_sup_plot_values_full_ep = []
        tremor_ampl_unsup_plot_values_full_ep = []
        tremor_when_reduction = np.zeros((len(envs), 2))
        tremor_when_ampl_reduction = np.zeros((len(envs), 2))
        tremor_reduction_in_episode = np.zeros((len(envs)))

        # export for mesh creation
        joint_imu_pos_full_ep = []
        joint_tremor_pos_full_ep = []
        joint_suppressed_pos_full_ep = []

        while not np.all(done):
            # choose action
            actions = agent.select_action(observation, use_checkpoint=True, use_exploration=False)

            # declare tremor containing vectors
            torque_values = np.zeros((len(envs), 7))
            tremor_torque_values = np.zeros((len(envs), 7))
            ampl_values = np.zeros((len(envs), 7))
            tremor_ampl_values = np.zeros((len(envs), 7))
            tremor_reduction_ep = np.zeros((len(envs), 7))
            tremor_ampl_reduction_ep = np.zeros((len(envs), 7))
            tremor_ampl_total_reduction_ep = np.zeros((len(envs)))
            tremor_ampl_sup_ep = np.zeros((len(envs), 7))
            tremor_ampl_unsup_ep = np.zeros((len(envs), 7))

            # timestep position values
            joint_imu_pos_ep = np.zeros((len(envs), 3))
            joint_tremor_pos_ep = np.zeros((len(envs), 3))
            joint_suppressed_pos_ep = np.zeros((len(envs), 3))

            # take step in the environment and store it
            for i, action in enumerate(actions):
                if not done[i]:
                    observation_[i], reward, done[i], _, info = envs[i].step(action)

                    score[i] += reward
                    ep_len[i] += 1
                    steps_count += 1
                    torque_values[i] = info["torque_val"]
                    tremor_torque_values[i] = info["tremor_torque_val"]
                    ampl_values[i] = info["ampl_val"]
                    tremor_ampl_values[i] = info["tremor_ampl_val"]
                    observation[i] = observation_[i]

                    # tremor reduction metrics
                    tremor_reduction = (abs(info["torque_val"]) - abs(info["tremor_torque_val"])) / abs(info["tremor_torque_val"] + eps) * 100
                    tremor_reduction = np.nan_to_num(tremor_reduction, nan=0, posinf=0, neginf=0)

                    # measure amplitude suppression in each joint axis angle
                    tremor_reduction_ampl = (abs(info["ampl_val"]) - abs(info["tremor_ampl_val"])) / abs(info["tremor_ampl_val"] + eps) * 100
                    tremor_reduction_ampl = np.nan_to_num(tremor_reduction_ampl, nan=0, posinf=0, neginf=0)

                    # measure total amplitude reduction in the movement | amplitude values to be used as not angles but cm values
                    # change the SAA and SFE angles to accommodate the D-H params
                    non_tremor_angle_values = np.radians(envs[i].return_original_joint_angles())

                    info["ampl_val"][0], info["ampl_val"][1] = info["ampl_val"][1], info["ampl_val"][0]
                    info["tremor_ampl_val"][0], info["tremor_ampl_val"][1] = info["tremor_ampl_val"][1], info["tremor_ampl_val"][0]

                    suppressed_angles = np.radians(info["ampl_val"]) + non_tremor_angle_values
                    unsuppressed_angles = np.radians(info["tremor_ampl_val"]) + non_tremor_angle_values

                    # calculate end effectors positions
                    end_effector_position_joint = forward_kinematics(non_tremor_angle_values, args.humerus_length, args.forearm_length, args.hand_length)
                    end_effector_position1 = forward_kinematics(suppressed_angles, args.humerus_length, args.forearm_length, args.hand_length)
                    end_effector_position2 = forward_kinematics(unsuppressed_angles, args.humerus_length, args.forearm_length, args.hand_length)

                    distance_suppressed = distance_3d(end_effector_position_joint, end_effector_position1)  # Distance from the origin
                    distance_unsuppressed = distance_3d(end_effector_position_joint, end_effector_position2)  # Distance from the origin

                    # save values for mesh creation
                    joint_imu_pos_ep[i] = end_effector_position_joint
                    joint_tremor_pos_ep[i] = end_effector_position2
                    joint_suppressed_pos_ep[i] = end_effector_position1

                    # Calculate amplitude difference in percentage
                    tremor_reduction_ampl_total = ((distance_suppressed - distance_unsuppressed) / distance_unsuppressed) * 100

                    # count the occurrences of tremor reduction alongside the axis # todo: get the cases where the involved axis are 0
                    # Select the relevant tremor axes as defined by args.tremor_sequence
                    selected_axes = tremor_reduction[args.tremor_sequence == 1]

                    # Check if suppression (tremor_reduction >= 0) occurred in *all* selected axes for each instance
                    all_axes_suppressed = np.all(selected_axes <= 0)
                    any_axes_suppressed = np.any(selected_axes <= 0)

                    # Update the tremor_when_reduction counter for this case
                    tremor_when_reduction[i, 1] += np.sum(all_axes_suppressed)

                    tremor_reduction_in_episode[i] += np.sum(any_axes_suppressed)

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
                    tremor_ampl_sup_ep[i] = np.zeros(7)
                    tremor_ampl_unsup_ep[i] = np.zeros(7)

            # store the tremor suppression values
            tremor_reduction_full_ep.append(tremor_reduction_ep.copy())
            tremor_ampl_reduction_full_ep.append(tremor_ampl_reduction_ep.copy())
            tremor_ampl_total_reduction_full_ep.append(tremor_ampl_total_reduction_ep.copy())
            tremor_torque_sup_plot_values_full_ep.append(torque_values.copy())
            tremor_torque_unsup_plot_values_full_ep.append(tremor_torque_values.copy())
            tremor_ampl_sup_plot_values_full_ep.append(ampl_values.copy())
            tremor_ampl_unsup_plot_values_full_ep.append(tremor_ampl_values.copy())

            # store values for mesh creation
            joint_imu_pos_full_ep.append(joint_imu_pos_ep)
            joint_tremor_pos_full_ep.append(joint_tremor_pos_ep)
            joint_suppressed_pos_full_ep.append(joint_suppressed_pos_ep)

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

        # tremor torque reduction metrics
        tremor_red_np = np.array(tremor_reduction_full_ep)
        full_episode_tremor_red_avg = np.zeros(len(envs))
        full_episode_tremor_red_median = np.zeros(len(envs))
        tremor_torque_sup_plot_values_full_ep = np.array(tremor_torque_sup_plot_values_full_ep)
        tremor_torque_unsup_plot_values_full_ep = np.array(tremor_torque_unsup_plot_values_full_ep)
        tremor_ampl_sup_plot_values_full_ep = np.array(tremor_ampl_sup_plot_values_full_ep)
        tremor_ampl_unsup_plot_values_full_ep = np.array(tremor_ampl_unsup_plot_values_full_ep)

        # tremor reduction amplitude metrics each joint axis
        tremor_red_ampl_np = np.array(tremor_ampl_reduction_full_ep)
        full_episode_tremor_red_ampl_avg = np.zeros(len(envs))
        full_episode_tremor_red_ampl_median = np.zeros(len(envs))

        # tremor reduction amplitude total metrics
        tremor_red_ampl_total_np = np.array(tremor_ampl_total_reduction_full_ep)
        eval_angle_sup = np.zeros((len(envs), 7))

        # violin plots
        for i, _ in enumerate(envs):
            index = i // 2
            if avg_score[i] > highest_scores[index]:
                highest_scores[index] = avg_score[i]
                violin_plot_values[index] = tremor_red_ampl_total_np[:max_episode_lengths[i], i] * -1

        # generate console outputs -- local outputs of the envs
        print("LOCAL ENV OUTPUTS: ")
        for i, env in enumerate(envs):
            print("TREMOR", i, "statistics: ")
            print(' score %.3f' % score[i], 'avg score %.3f' % avg_score[i], 'median score %.3f' % median_score[i], 'max score %.3f' % np.max(np.array(scores)[:, i]),
                  'std of scores %.3f' % np.std(np.array(scores)[-100:, i], axis=0))
            print('Reward achieved by the agent: %.3f, Score in percent to max: %.3f, avg rewards in percent: %.3f, median rewards in percent: %.3f' %
                  (gotten_score[i], reward_percentage[i], avg_agent_r[i], median_agent_r[i]))
            print("Tremor torque reduction metrics:")
            print(' Tremor reduction ep avg', np.mean(tremor_red_np[:max_episode_lengths[i], i, :], axis=0), '\n Tremor reduction ep median',
                  np.median(tremor_red_np[:max_episode_lengths[i], i, :], axis=0), '\n Tremor reduction ep std',
                  np.std(tremor_red_np[:max_episode_lengths[i], i, :], axis=0))
            print('Tremor reduction occurred in', (tremor_when_reduction[i, 1] / (max_episode_lengths[i] - 3)) * 100,
                  '% of all the generated tremor axis in all time steps')
            tremor_sup_avg = np.mean(tremor_red_np[:max_episode_lengths[i], i, :][tremor_red_np[:max_episode_lengths[i], i, :] != 0], axis=None)
            tremor_sup_median = np.median(tremor_red_np[:max_episode_lengths[i], i, :][tremor_red_np[:max_episode_lengths[i], i, :] != 0], axis=None)
            print("Some sort of tremor reduction occurred along any axis in",
                  tremor_reduction_in_episode[i] / (envs[i].return_max_length() - 3) * 100,
                  "% of the episode")
            print("Tremor amplitude reduction metrics:")
            print(' Tremor reduction ep avg', np.mean(tremor_red_ampl_np[:max_episode_lengths[i], i, :], axis=0), '\n Tremor reduction ep median',
                  np.median(tremor_red_ampl_np[:max_episode_lengths[i], i, :], axis=0), '\n Tremor reduction ep std',
                  np.std(tremor_red_ampl_np[:max_episode_lengths[i], i, :], axis=0))
            print('Tremor amplitude reduction occurred in', (tremor_when_ampl_reduction[i, 1] / max_episode_lengths[i]) * 100,
                  '% of all time steps')
            print("Overall tremor suppression avg in the angle axis: ", tremor_sup_avg, "%")
            print("Overall tremor suppression median in the angle axis: ", tremor_sup_median, "%")
            print("Total tremor amplitude suppression", np.mean(tremor_red_ampl_total_np[:max_episode_lengths[i], i]), "%")
            print(f"For torque places: {torque_places[i, :]}, With maximum Nm of: {torque_maxes[i, :]}")
            print()
            full_episode_tremor_red_avg[i] = tremor_sup_avg
            full_episode_tremor_red_median[i] = tremor_sup_median
            full_episode_tremor_red_ampl_avg[i] = np.mean(tremor_red_ampl_np[:max_episode_lengths[i], i, :][tremor_red_ampl_np[:max_episode_lengths[i], i, :] != 0], axis=None)
            full_episode_tremor_red_ampl_median = np.median(tremor_red_ampl_np[:max_episode_lengths[i], i, :][tremor_red_ampl_np[:max_episode_lengths[i], i, :] != 0], axis=None)

            # score global tremor suppression values
            tremor_torque_suppression_individual_avg.append(np.mean(tremor_red_np[:max_episode_lengths[i], i, :], axis=0))
            tremor_torque_suppression_individual_median.append((np.median(tremor_red_np[:max_episode_lengths[i], i, :], axis=0)))

            # store the local angle suppression values
            eval_angle_sup[i] = np.mean(tremor_red_ampl_np[:max_episode_lengths[i], i, :], axis=0)

            # store valuables for the plot
            if avg_score[i] > highest_score:
                # torque values
                highest_score = avg_score[i]
                tremor_amplitude_histogram_values = tremor_red_ampl_total_np[:max_episode_lengths[i], i]
                tremor_supp_torque_plot_values = tremor_torque_sup_plot_values_full_ep[:max_episode_lengths[i], i]
                tremor_unsup_torque_plot_values = tremor_torque_unsup_plot_values_full_ep[:max_episode_lengths[i], i]

                # amplitude values
                tremor_supp_ampl_plot_values = tremor_ampl_sup_plot_values_full_ep[:max_episode_lengths[i], i]
                tremor_unsup_ampl_plot_values = tremor_ampl_unsup_plot_values_full_ep[:max_episode_lengths[i], i]

                # export the position values
                # convert to np array
                joint_imu_pos_full_ep = np.array(joint_imu_pos_full_ep)
                joint_tremor_pos_full_ep = np.array(joint_tremor_pos_full_ep)
                joint_suppressed_pos_full_ep = np.array(joint_suppressed_pos_full_ep)

                # transform the values for the simulator
                joint_imu_pos_full_ep[:, :, 0] += 0.01
                joint_tremor_pos_full_ep[:, :, 0] += 0.01
                joint_suppressed_pos_full_ep[:, :, 0] += 0.01

                joint_imu_pos_full_ep[:, :, 1] += -0.475
                joint_tremor_pos_full_ep[:, :, 1] += -0.475
                joint_suppressed_pos_full_ep[:, :, 1] += -0.475

                joint_imu_pos_full_ep[:, :, 2] += 1.2
                joint_tremor_pos_full_ep[:, :, 2] += 1.2
                joint_suppressed_pos_full_ep[:, :, 2] += 1.2

                print("\n Tremor Plot values have been saved for reference movement: " + str(i) + "\n")
                np.savetxt("mesh_data/end_effector_positions_" + str(i) + ".txt", joint_imu_pos_full_ep[:max_episode_lengths[i]//2, i, :], delimiter=',')
                np.savetxt("mesh_data/tremor_cause_positions_" + str(i) + ".txt", joint_tremor_pos_full_ep[:max_episode_lengths[i]//2, i], delimiter=',')
                np.savetxt("mesh_data/suppressed_positions_" + str(i) + ".txt", joint_suppressed_pos_full_ep[:max_episode_lengths[i]//2, i], delimiter=',')

        # save the standard deviations
        std_score.append(np.std(avg_agent_rew[-100:], axis=0))
        median_tremor_sup.append(np.mean(full_episode_tremor_red_median))
        median_std_tremor_sup.append(np.std(median_tremor_sup[-100:]))

        # generate global outputs
        print("GLOBAL TRAINING OUTPUTS: ")
        print("Episode count: ", episode_count)
        print("Average reward of the agent %.3f, Average score in percent to max: %.3f, avg rewards in percent: %.3f, median rewards in percent: %.3f" %
              (np.mean(gotten_score), np.mean(reward_percentage), np.mean(avg_agent_r), np.median(avg_agent_r)))
        print()
        print("Tremor torque reduction metrics:")
        print('Tremor reduction occurred in', (np.sum(tremor_when_reduction[:, 1]) / total_steps_for_ep) * 100,
              '% of all the generated tremor axis in all time steps')
        print("Some sort of tremor reduction occurred along any axis in",
              np.sum(tremor_reduction_in_episode[:]) / total_steps_for_ep * 100,
              "% of the episode")
        print("Average of the average tremor suppression across all movements", np.mean(tremor_torque_suppression_individual_avg[-100:], axis=0),
              "\nSTD of it", np.std(tremor_torque_suppression_individual_avg[-100:], axis=0),
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
        print("Total average angle suppression across joints", np.mean(eval_angle_sup, axis=0))

        # store the global values
        evaluation_percent[episode_count] = (np.sum(tremor_when_ampl_reduction[:, 1]) / (np.sum(tremor_when_ampl_reduction[:, 0]) + np.sum(tremor_when_ampl_reduction[:, 1]))) * 100
        evaluation_angle_amplitude_percent[episode_count] = np.mean(eval_angle_sup, axis=0)
        evaluation_amplitude_percent[episode_count] = np.mean(tremor_red_ampl_total_np[tremor_red_ampl_total_np[:, :] != 0], axis=None)

        evaluation_torque_suppression_some[episode_count] = np.sum(tremor_reduction_in_episode[:]) / total_steps_for_ep * 100
        evaluation_torque_all_axis[episode_count] = (np.sum(tremor_when_reduction[:, 1]) / total_steps_for_ep) * 100

    # print out the overall eval data (joint angle reduction on the axis, the % it occurred in, amplitude suppression)
    print()
    print("EVALUATION METRICS:")
    print("Tremor amplitude suppression occurred in", np.mean(evaluation_percent), "% of all steps")
    print("Tremor amplitude suppression occurred in std", np.std(evaluation_percent), "% of all steps")
    print("Tremor amplitude suppression across all angles", np.mean(evaluation_angle_amplitude_percent, axis=0), "%")
    print("Tremor amplitude suppression across all angles stds", np.std(evaluation_angle_amplitude_percent, axis=0), "%")
    print("Tremor total amplitude suppression", np.mean(evaluation_amplitude_percent), "%")
    print("Tremor total amplitude suppression std", np.std(evaluation_amplitude_percent), "%")
    print("\n Tremor torque suppression in any angle avg", np.mean(evaluation_torque_suppression_some))
    print("Tremor torque suppression in any angle std", np.std(evaluation_torque_suppression_some))
    print("Tremor torque suppression in all angle avg", np.mean(evaluation_torque_all_axis))
    print("Tremor torque suppression in all angle std", np.std(evaluation_torque_all_axis))

    # plot the histogram and torque values
    plot_histogram(tremor_amplitude_histogram_values, "figures/Tremor_amplitude_sup_histogram")
    plot_joint_torques(tremor_supp_torque_plot_values, tremor_unsup_torque_plot_values, "figures/Tremor_torque_plot")
    plot_joint_amplitudes(tremor_supp_ampl_plot_values, tremor_unsup_ampl_plot_values, "figures/Tremor_ampl_plot")
    tremor_violin_plot(sfe=violin_plot_values[0], saa=violin_plot_values[1], efe=violin_plot_values[2], sei=violin_plot_values[3])

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
