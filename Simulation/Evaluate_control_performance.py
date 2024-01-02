import numpy as np
import sys
import time
from Environment.Exoskeleton_env import ExoskeletonEnv_train
from Agent.TD7_multi_agent import Agent
from Utilities.calculate_body_part_mass_ import calculate_body_part_mass
from Utilities.seed_setting_ import set_seeds
from Utilities.Train_logger_ import Logger
from Utilities.measure_script_time_ import seconds_to_hms
from Utilities.calculate_arm_end_effector_points import distance_3d, forward_kinematics
from Utilities.publication_plots import plot_histogram, plot_joint_torques
import pybullet as p
import pybullet_utils.bullet_client as bc

if __name__ == "__main__":
    # Record the start time
    start_time = time.time()
    np.set_printoptions(precision=3, floatmode='fixed')

    # set the seed
    set_seeds(0)  # 17,31,42,69,81 are the used seeds   ---31. gives best results

    # Create an instance of the Logger class
    logger = Logger("output.txt")

    # Redirect the standard output to the logger
    sys.stdout = logger

    # values for the simulation
    mass = 81.5  # in kg
    u_arm_weight, l_arm_weight, h_weight = calculate_body_part_mass(mass)
    humerus_length = 0.4
    forearm_length = 0.4
    humerus_radius = 0.05
    forearm_radius = 0.05
    hand_length = 0.05

    # tremor
    tremor_sequence = np.array([False, True, False, True, False, False, False])

    envs = []
    for i in range(8):
        client = bc.BulletClient(connection_mode=p.DIRECT)
        env = ExoskeletonEnv_train(evaluation=True, dummy_shift=False, hum_weight=u_arm_weight, forearm_weight=l_arm_weight, forearm_height=forearm_length,
                                   forearm_radius=forearm_radius, hum_height=humerus_length, hum_radius=humerus_radius, hand_weight=h_weight, hand_radius=hand_length,
                                   mode=3, use_all_dof=False, tremor_axis=1, reference_motion_file_num=str(i), tremor_sequence=[0, 1, 0, 1, 0, 0, 0], max_force_shoulder=40,
                                   max_force_elbow=20, client=client)
        envs.append(env)
    # delete the [0,0,0,0] env or not?

    # values for training
    num_eval_episodes = 100
    mode_steps = 0
    steps_count = 0
    prev_step_count = 0
    prev_buffer_save = 0
    empty_list = []
    scores = []
    agent_rew = []
    tremor_torque_suppression_individual_avg = []
    tremor_suppression_individual_avg_std = []
    tremor_torque_suppression_individual_median = []
    tremor_suppression_individual_median_std = []
    agent = Agent(state_dim=envs[0].observation_space.shape[0], action_dim=envs[0].action_space.shape[0], max_action=1, learning_steps=1)
    best_agent_performance = 0
    since_saved = 0
    total_steps_for_ep = 0
    for i, _ in enumerate(envs):
        total_steps_for_ep += envs[i].return_max_length() - 3
    load = True
    disregard = True

    # values for evaluation of the training
    avg_agent_rew = []
    median_agent_sup = [empty_list for _, _ in enumerate(envs)]
    std_score = []
    std_median_sup = [empty_list for _, count in enumerate(envs)]
    max_episode_lengths = [env.return_max_length() for env in envs]

    # global training variables
    avg_agent_rew_global = []
    std_score_global = []
    avg_tremor_sup = []
    avg_std_tremor = []
    median_tremor_sup = []
    median_std_tremor_sup = []

    # load the model if necessary
    if load:
        agent.load("Trained_Agents/[0,1,0,1]/[0,1,0,1]")

    # agent train variables
    observation = np.zeros((len(envs), envs[0].observation_space.shape[0]))
    observation_ = np.zeros((len(envs), envs[0].observation_space.shape[0]))
    score = np.zeros(len(envs))
    actions = np.zeros((len(envs), 7))
    torque_places = np.zeros((len(envs), 7))
    torque_maxes = np.zeros((len(envs), 7))

    # evaluation holders
    evaluation_percent = np.zeros(num_eval_episodes)
    evaluation_amplitude_percent = np.zeros(num_eval_episodes)
    evaluation_angle_amplitude_percent = np.zeros((num_eval_episodes, 7))
    evaluation_torque_suppression_some = np.zeros(num_eval_episodes)
    evaluation_torque_all_axis = np.zeros(num_eval_episodes)

    # variables for the histogram and the torque plots
    highest_score = 0
    tremor_amplitude_histogram_values = None
    tremor_supp_torque_plot_values = None
    tremor_unsup_torque_plot_values = None

    # test the running of the environment
    for k in range(num_eval_episodes):
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
        tremor_when_reduction = np.zeros((len(envs), 2))
        tremor_when_ampl_reduction = np.zeros((len(envs), 2))
        tremor_reduction_in_episode = np.zeros((len(envs)))

        # export for mesh creation
        joint_imu_pos_full_ep = []
        joint_tremor_pos_full_ep = []
        joint_suppressed_pos_full_ep = []

        # variables for obs storing todo: implement for the varying length envs
        env_counter = 0

        while not np.all(done):
            # choose action
            for i, obs in enumerate(observation):
                if not done[i]:
                    actions[i] = agent.select_action(obs, use_checkpoint=True, use_exploration=False)

            # declare tremor containing vectors
            torque_values = np.zeros((len(envs), 7))
            tremor_torque_values = np.zeros((len(envs), 7))
            tremor_reduction_ep = np.zeros((len(envs), 7))
            tremor_ampl_reduction_ep = np.zeros((len(envs), 7))
            tremor_ampl_total_reduction_ep = np.zeros((len(envs)))

            # timestep position values
            joint_imu_pos_ep = np.zeros((len(envs), 3))
            joint_tremor_pos_ep = np.zeros((len(envs), 3))
            joint_suppressed_pos_ep = np.zeros((len(envs), 3))

            # take step in the environment and store it
            for i, action in enumerate(actions):
                if not done[i]:
                    observation_[i], reward, done[i], actuator_torques, torq_val, ampl_val, tremor_torq_val, tremor_ampl_val, info = envs[i].step(action)

                    score[i] += reward
                    ep_len[i] += 1
                    steps_count += 1
                    mode_steps += 1
                    torque_values[i] = torq_val
                    tremor_torque_values[i] = tremor_torq_val
                    observation[i] = observation_[i]

                    # tremor reduction metrics
                    tremor_reduction = (abs(torq_val) - abs(tremor_torq_val)) / abs(tremor_torq_val) * 100
                    tremor_reduction = np.nan_to_num(tremor_reduction, nan=0, posinf=0, neginf=0)

                    # measure amplitude suppression in each joint axis angle
                    tremor_reduction_ampl = (abs(ampl_val) - abs(tremor_ampl_val)) / abs(tremor_ampl_val) * 100
                    tremor_reduction_ampl = np.nan_to_num(tremor_reduction_ampl, nan=0, posinf=0, neginf=0)

                    # measure total amplitude reduction in the movement | amplitude values to be used as not angles but cm values
                    # change the SAA and SFE angles to accommodate the D-H params
                    non_tremor_angle_values = np.radians(envs[i].return_original_joint_angles())

                    ampl_val[0], ampl_val[1] = ampl_val[1], ampl_val[0]
                    tremor_ampl_val[0], tremor_ampl_val[1] = tremor_ampl_val[1], tremor_ampl_val[0]

                    suppressed_angles = np.radians(ampl_val) + non_tremor_angle_values
                    unsuppressed_angles = np.radians(tremor_ampl_val) + non_tremor_angle_values

                    # calculate end effectors positions
                    end_effector_position_joint = forward_kinematics(non_tremor_angle_values, humerus_length, forearm_length, hand_length)
                    end_effector_position1 = forward_kinematics(suppressed_angles, humerus_length, forearm_length, hand_length)
                    end_effector_position2 = forward_kinematics(unsuppressed_angles, humerus_length, forearm_length, hand_length)

                    distance_suppressed = distance_3d(end_effector_position_joint, end_effector_position1)  # Distance from the origin
                    distance_unsuppressed = distance_3d(end_effector_position_joint, end_effector_position2)  # Distance from the origin

                    # save values for mesh creation
                    joint_imu_pos_ep[i] = end_effector_position_joint
                    joint_tremor_pos_ep[i] = end_effector_position2
                    joint_suppressed_pos_ep[i] = end_effector_position1

                    # Calculate amplitude difference in percentage
                    tremor_reduction_ampl_total = ((distance_suppressed - distance_unsuppressed) / distance_unsuppressed) * 100

                    # count the occurrences of tremor reduction alongside the axis # todo: get the cases where the involved axis are 0
                    tremor_when_reduction[i, 0] += np.sum(tremor_reduction[tremor_sequence] >= 0)
                    tremor_when_reduction[i, 1] += np.sum(tremor_reduction[tremor_sequence] < 0)

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
            tremor_torque_sup_plot_values_full_ep.append(torque_values.copy())
            tremor_torque_unsup_plot_values_full_ep.append(tremor_torque_values.copy())

            # store values for mesh creation
            joint_imu_pos_full_ep.append(joint_imu_pos_ep)
            joint_tremor_pos_full_ep.append(joint_tremor_pos_ep)
            joint_suppressed_pos_full_ep.append(joint_suppressed_pos_ep)

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
        tremor_torque_sup_plot_values_full_ep = np.array(tremor_torque_sup_plot_values_full_ep)
        tremor_torque_unsup_plot_values_full_ep = np.array(tremor_torque_unsup_plot_values_full_ep)

        # tremor reduction amplitude metrics each joint axis
        tremor_red_ampl_np = np.array(tremor_ampl_reduction_full_ep)
        full_episode_tremor_red_ampl_avg = np.zeros(len(envs))
        full_episode_tremor_red_ampl_median = np.zeros(len(envs))

        # tremor reduction amplitude total metrics
        tremor_red_ampl_total_np = np.array(tremor_ampl_total_reduction_full_ep)
        full_episode_tremor_red_ampl_total_avg = np.zeros(len(envs))
        eval_angle_sup = np.zeros((len(envs), 7))

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
            print('Tremor reduction occurred in', (tremor_when_reduction[i, 1] / (tremor_when_reduction[i, 0] + tremor_when_reduction[i, 1])) * 100,
                  '% of all the generated tremor axis in all timesteps')
            tremor_sup_avg = np.mean(tremor_red_np[:max_episode_lengths[i], i, :][tremor_red_np[:max_episode_lengths[i], i, :] != 0], axis=None)
            tremor_sup_median = np.median(tremor_red_np[:max_episode_lengths[i], i, :][tremor_red_np[:max_episode_lengths[i], i, :] != 0], axis=None)
            print("Some sort of tremor reduction occurred along any axis in",
                  tremor_reduction_in_episode[i] / (envs[i].return_max_length() - 3) * 100,
                  "% of the episode")
            print("Tremor amplitude reduction metrics:")
            print(' Tremor reduction ep avg', np.mean(tremor_red_ampl_np[:max_episode_lengths[i], i, :], axis=0), '\n Tremor reduction ep median',
                  np.median(tremor_red_ampl_np[:max_episode_lengths[i], i, :], axis=0), '\n Tremor reduction ep std',
                  np.std(tremor_red_ampl_np[:max_episode_lengths[i], i, :], axis=0))
            print('Tremor amplitude reduction occurred in', (tremor_when_ampl_reduction[i, 1] / (tremor_when_ampl_reduction[i, 0] + tremor_when_ampl_reduction[i, 1])) * 100,
                  '% of all timesteps')
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
                # increment or reset values
                highest_score = avg_score[i]
                tremor_amplitude_histogram_values = tremor_red_ampl_total_np[:max_episode_lengths[i], i]
                tremor_supp_torque_plot_values = tremor_torque_sup_plot_values_full_ep[:max_episode_lengths[i], i]
                tremor_unsup_torque_plot_values = tremor_torque_unsup_plot_values_full_ep[:max_episode_lengths[i], i]

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

        # store global score values
        avg_agent_rew_global.append(np.mean(reward_percentage))
        std_score_global.append(np.std(avg_agent_rew_global))

        # generate global outputs
        print("GLOBAL TRAINING OUTPUTS: ")
        print("Steps count: ", steps_count)
        print("Average reward of the agent %.3f, Average score in percent to max: %.3f, avg rewards in percent: %.3f, median rewards in percent: %.3f" %
              (np.mean(gotten_score), np.mean(reward_percentage), np.mean(avg_agent_r), np.median(avg_agent_r)))
        print()
        print("Tremor torque reduction metrics:")
        print('Tremor reduction occurred in', (np.sum(tremor_when_reduction[:, 1]) / (np.sum(tremor_when_reduction[:, :]))) * 100,
              '% of all the generated tremor axis in all timesteps')
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
              '% of all timesteps')
        print("Total average tremor amplitude suppression in the episode", np.mean(tremor_red_ampl_total_np[tremor_red_ampl_total_np[:, :] != 0], axis=None), "%")
        print("Total average angle suppression across joints", np.mean(eval_angle_sup, axis=0))

        # save the standard deviations
        last_100_rows = avg_agent_rew[-100:]
        std_score.append(np.std(last_100_rows, axis=0))
        avg_tremor_sup.append(np.mean(full_episode_tremor_red_avg))
        median_tremor_sup.append(np.mean(full_episode_tremor_red_median))
        avg_std_tremor.append(np.std(avg_tremor_sup[-100:]))
        median_std_tremor_sup.append(np.std(median_tremor_sup[-100:]))

        # store the global values
        evaluation_percent[k] = (np.sum(tremor_when_ampl_reduction[:, 1]) / (np.sum(tremor_when_ampl_reduction[:, 0]) + np.sum(tremor_when_ampl_reduction[:, 1]))) * 100
        evaluation_angle_amplitude_percent[k] = np.mean(eval_angle_sup, axis=0)
        evaluation_amplitude_percent[k] = np.mean(tremor_red_ampl_total_np[tremor_red_ampl_total_np[:, :] != 0], axis=None)

        evaluation_torque_suppression_some[k] = np.sum(tremor_reduction_in_episode[:]) / total_steps_for_ep * 100
        evaluation_torque_all_axis[k] = (np.sum(tremor_when_reduction[:, 1]) / (np.sum(tremor_when_reduction[:, :]))) * 100

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
