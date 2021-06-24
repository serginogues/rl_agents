"""
DDPG-HER agent based on https://github.com/hemilpanchiwala/Hindsight-Experience-Replay

ENVIRONMENT CONSTRAINTS
- env.action_space.n    returns action space size
- env.reset()           returns current state
- env.goal              returns current goal
- env.step(action)      returns next_state, reward, done, info and type(action) = int

Define SAVE_PATH before running
"""
import os
import time
import matplotlib.pyplot as plt
from ddpg_her.DDPGHerAgent import Agent
import random
import numpy as np
import gym

# import BitFlipEnv


# path where the networks get saved
# todo: update before every training!
SAVE_PATH = 'trained_agents/'
SAVE_NAME = 'albert0'

# -------------------- #
# Sergi0 trained agent #
# -------------------- #
# SAVE_PATH = 'trained_agents/sergi0'
# SAVE_NAME = 'sergi0'

try:
    os.mkdir(SAVE_PATH)
except OSError:
    pass

# TYPE
GAZEBO = False  # whether to run with gazebo compatibility
TRAIN = True
TEST = False
RELOAD = False
RELOAD_EPOCH = 0

# TOLERANCE
DYNAMIC_TOLERANCE = False
TOLERANCE = 0.5 if DYNAMIC_TOLERANCE else 0.15  # distance from end effector to target to be considered sufficiently close
MIN_TOLERANCE = 0.05  # minimum value of tolerance
TOLERANCE_DECREASE = 0.1  # amount to decrease tolerance by when limit is surpassed
DECREASE_TOLERANCE_LIMIT = 20  # percentage at which to decrease the tolerance

# TRAIN
EPOCHS = 100  # 250 each EPOCH consists of CYCLES
CYCLES = 50  # 50
EPISODES = 16  # 16
STEPS = 70  # 20
OPTIMIZATION_STEPS = 80  # 40

# AGENT HYPERPARAMS
BATCH_SIZE = 256  # 256
MEMORY_SIZE = 5e5
DECAY_EPS = 5e-5  # 0.95
LR_ACTOR = 0.0001
LR_CRITIC = 0.001
DISCOUNT_FACTOR = 0.98
EPSILON = 0.9  # 0.9
n_goals = 4
MIN_EPSILON = 0.1
ACTION_SPACE = 69 * 6  # 61 -> range of desired joint / target resolution + 1
REPLACE_NETWORK_INTERVAL = 10000
ACTIONS = 7
TAU = 0.01  # 0.005
IS_OBSERVATION_NORM = False
HER = True
ENV = 'iiwaEnv-v0'
MODEL_NAME = 'DDPG_{}_HER_{}'.format('WITH' if HER else 'WITHOUT', ENV)

# NON SPARSE REWARD
SMALL = 1
MEDIUM = 10
LARGE = 100
REWARDS = [SMALL, MEDIUM, LARGE]

DISTANCE_AS_REWARD = False
SPARSE_REWARDS = True
DONE_ON_COLLISION = True

PRINT_REWARD_PER_EPOCH = False  # print reward every step
TRACK_BEST_REWARD = True  # save best reward every episode
PRINT_TERMINAL = False  # print distance to target, collisions, target reached
PRINT_LOSS_EPSILON = True
PRINT_FIRST_SECOND_TARGET = True

TRAJECTORY_PLOT = False
SAVE_TRAJECTORY_PLOT = False
SHOW_TRAJECTORY_PLOT = True
if TRAJECTORY_PLOT:
    import plots


def test_dqn_her(env):
    state_size = env.reset().shape[0]
    checkpoint_dir = os.path.join(os.getcwd(), SAVE_PATH)
    # Initializes the DDPG agent with  Hindsight Experience Replay
    agent = Agent(actor_learning_rate=LR_ACTOR, critic_learning_rate=LR_CRITIC,
                  n_actions=ACTIONS, input_dims=3, n_goal=3,
                  gamma=DISCOUNT_FACTOR, memory_size=int(MEMORY_SIZE), tau=TAU,
                  batch_size=BATCH_SIZE, epsilon=EPSILON, dec_epsilon=DECAY_EPS, min_epsilon=MIN_EPSILON,
                  soft_update_step=REPLACE_NETWORK_INTERVAL, is_obs_norm=IS_OBSERVATION_NORM, model_name=MODEL_NAME)
    agent.load_model()
    success = 0
    target1_reached = 0
    for episode in range(EPISODES):
        stored_states = np.zeros((STEPS + 1, 3))
        state = env.reset()
        # print("STATE: ", state, "TARGETS: ", env.targets)
        for step in range(STEPS):
            goal = env.target
            stored_states[step] = (np.asarray(state))
            action, theta_vector = agent.choose_action(state, goal)
            next_state, reward, done, info = env.step(action)
            end = True if done else False
            done = env.check_terminal_state(done=done, reward=reward, HER=False)
            if end:
                if done:
                    print("Episode finished after {} timesteps".format(step + 1))
                    success += 1
                break
            state = next_state
        if env.first_target_reached:
            target1_reached += 1

        plots.trajectory_plot(stored_states, episode, save=SAVE_TRAJECTORY_PLOT, show=SHOW_TRAJECTORY_PLOT,
                              path=SAVE_PATH)

    print("Win percentage of ", success, " out of ", EPISODES)
    print("First target reached in ", target1_reached, "out of ", EPISODES, "episodes")
    env.close()


def train_dqn_her(env):
    global STEPS
    print("**************CONFIG*****************")
    print('GAZEBO = ', GAZEBO)
    print('TRAIN = ', TRAIN)
    print('TEST = ', TEST)
    print('RELOAD = ', RELOAD)
    print('RELOAD_EPOCH = ', RELOAD_EPOCH)
    print('TOLERANCE = ', TOLERANCE)
    print('EPOCHS = ', EPOCHS)
    print('CYCLES = ', CYCLES)
    print('EPISODES = ', EPISODES)
    print('STEPS = ', STEPS)
    print('OPTIMIZATION_STEPS = ', OPTIMIZATION_STEPS)
    print('BATCH_SIZE = ', BATCH_SIZE)
    print('MEMORY_SIZE = ', MEMORY_SIZE)
    print('DECAY_EPS = ', DECAY_EPS)
    print('LR_ACTOR = ', LR_ACTOR)
    print('LR_CRITIC = ', LR_CRITIC)
    print('DISCOUNT_FACTOR = ', DISCOUNT_FACTOR)
    print('EPSILON = ', EPSILON)
    print('TAU = ', TAU)
    print('IS_OBSERVATION_NORM = ', IS_OBSERVATION_NORM)
    print('n_goals = ', n_goals)
    print('MIN_EPSILON = ', MIN_EPSILON)
    print('ACTION_SPACE = ', ACTION_SPACE)
    print('REPLACE_NETWORK_INTERVAL = ', REPLACE_NETWORK_INTERVAL)
    print('ACTIONS = ', ACTIONS)
    print('SMALL = ', SMALL)
    print('MEDIUM = ', MEDIUM)
    print('LARGE = ', LARGE)
    print('REWARDS = ', REWARDS)

    state_size = env.reset().shape[0]
    epsilon_history = []
    epochs = []

    win_percent = []
    win_first_target = []
    win_epsilon = []
    win_loss_actor = []
    win_loss_critic = []
    win_D1 = []
    win_D2 = []
    win_collisions = []
    win_mean_reward = []
    var_counter = 0
    counter_limit = 5
    var_list = np.zeros((counter_limit, 3))
    min_var = 999.
    success = 0
    target1_reached = 0

    checkpoint_dir = os.path.join(os.getcwd(), SAVE_PATH)

    # Initializes the DDPG agent with  Hindsight Experience Replay
    agent = Agent(actor_learning_rate=LR_ACTOR, critic_learning_rate=LR_CRITIC,
                  n_actions=ACTIONS, input_dims=3, n_goal=3,
                  gamma=DISCOUNT_FACTOR, memory_size=int(MEMORY_SIZE), tau=TAU,
                  batch_size=BATCH_SIZE, epsilon=EPSILON, dec_epsilon=DECAY_EPS, min_epsilon=MIN_EPSILON,
                  soft_update_step=REPLACE_NETWORK_INTERVAL, is_obs_norm=IS_OBSERVATION_NORM, model_name=MODEL_NAME)
    best_reward = -1000
    if RELOAD:
        start = RELOAD_EPOCH
        agent.load_model()
    else:
        start = 0

    for epoch in range(start, EPOCHS):
        steps = 0
        collisions = 0
        closest_approach = 9999.
        closest_approach_to_second = 9999.
        start_time = time.time()
        epoch_reward_mean = []
        mean_loss_actor = []
        mean_loss_critic = []
        mean_steps = []
        var_counter = 0
        for cycle in range(CYCLES):

            # for episode = 1, M do
            for episode in range(EPISODES):

                # Sample a goal g and an initial state s0
                state = env.reset()
                episode_transitions = []
                total_reward_episode = 0

                # for t = 0, T − 1 do
                for p in range(STEPS):
                    steps += 1
                    goal = env.target
                    # Sample an action at using the behavioral policy from A
                    action = agent.choose_action(state, goal)
                    # Execute the action at and observe a new state st+1
                    next_state, reward, done, info = env.step(action)

                    # visuals
                    var_list[var_counter % counter_limit] = np.asarray(next_state)
                    var_counter += 1

                    # check terminal state
                    end = True if done else False
                    done = env.check_terminal_state(HER=False)

                    # visuals
                    distance_to_target = env.dist_to_target
                    distance_to_second_target = env.dist_to_second_target
                    if closest_approach > distance_to_target:
                        closest_approach = distance_to_target
                    if closest_approach_to_second > distance_to_second_target:
                        closest_approach_to_second = distance_to_second_target
                    collisions += 1 if end and not done else 0

                    if done:
                        success += 1

                    if TRACK_BEST_REWARD:
                        total_reward_episode += reward
                        if reward > best_reward:
                            best_reward = reward
                        if PRINT_REWARD_PER_EPOCH:
                            print("Best reward {}".format(best_reward))

                    # Store the transition (st||g, at, rt, st+1||g) in R
                    # standard experience replay
                    agent.store_experience(state, action, reward, next_state, done, goal)
                    episode_transitions.append((state, action, reward, next_state, env.target_id))

                    state = next_state

                    if end:
                        if PRINT_REWARD_PER_EPOCH:
                            print("Reward at epoch {}, cycle {}, episode {}, step {} : reward {}".format(epoch, cycle,
                                                                                                         episode, p,
                                                                                                         best_reward))
                        break
                if var_counter > 4:
                    temp_var = np.var(var_list)
                    if temp_var < min_var:
                        min_var = temp_var
                if env.first_target_reached:
                    target1_reached += 1

                # for t = 0, T − 1 do
                for current_state_idx, transition in enumerate(episode_transitions):
                    # rt := r(st, at, g)
                    state_, action_, new_reward, next_state_, info_ = transition

                    # Sample a set of additional goals for replay G := S(current episode)
                    # for g' ∈ G do
                    for _ in range(n_goals):
                        transition = random.choice(episode_transitions[current_state_idx:])
                        new_goal = transition[0]  # set state as new goal
                        # r':= r(st, at, g')

                        if not all(new_goal == state_):
                            # Store the transition (st||g', at, r', st+1||g') in R
                            new_reward, new_done, new_target = env.evaluate(state_, new_goal, info_)
                            new_done = env.check_terminal_state(done=new_done, reward=new_reward, HER=True)
                            agent.store_experience(state_, action_, new_reward, next_state_, new_done, new_goal)

                epoch_reward_mean.append(total_reward_episode)

            for s in range(OPTIMIZATION_STEPS):
                agent.learn()
                mean_loss_actor.append(agent.actor_loss)
                mean_loss_critic.append(agent.critic_loss)
            agent.soft_update_all()

        seconds = time.time() - start_time
        print("--- Epoch finished in %s seconds ---" % seconds)

        win_D1.append(np.round(closest_approach, 3) * 100)
        if closest_approach_to_second != np.inf:
            win_D2.append(np.round(closest_approach_to_second, 3) * 100)
        else:
            win_D2.append(100)
        win_mean_reward.append(np.mean(epoch_reward_mean))
        win_collisions.append(np.round(collisions, 0))

        if PRINT_LOSS_EPSILON:
            print("EPOCH:", epoch)
            print(" - Loss Actor: ", np.round(np.mean(mean_loss_actor), 3))
            print(" - Loss Critic: ", np.round(np.mean(mean_loss_critic), 3))
            print(" - Epsilon: ", np.round(agent.epsilon, 3))
            print(" - Steps: ", np.round(steps, 0))
            print(" - Collisions: ", np.round(collisions, 0))
            print(" - D1: ", np.round(closest_approach, 3))
            if closest_approach_to_second < 9999.:
                print(" - D2: ", np.round(closest_approach_to_second, 3))
            print(" - Postion Variance: ", np.round(min_var, 3))
            print(" - Tolerance:", env.tolerance)

            print(" - Mean reward: ", np.mean(epoch_reward_mean))

        print("Summary for last 800 episodes, after ", (epoch + 1) * CYCLES * EPISODES, " with max steps ",
              STEPS * CYCLES * EPISODES)
        print('Second Target success rate (task completed):', success / 8, "%")
        print("First target success rate:", target1_reached / 8, "%")
        if DYNAMIC_TOLERANCE and target1_reached / 8 > DECREASE_TOLERANCE_LIMIT:
            env.tolerance -= TOLERANCE_DECREASE if env.tolerance > MIN_TOLERANCE else 0

        # if len(win_percent) > 0 and (success / 8) > win_percent[len(win_percent) - 1]:
        agent.save_model()  # save the two Q-Networks
        epochs.append(epoch)

        win_percent.append(success / 8)
        win_first_target.append(target1_reached / 8)
        win_loss_actor.append(np.round(agent.actor_loss * 1000, 3))
        win_loss_critic.append(np.round(agent.critic_loss * 1000, 3))
        win_epsilon.append(agent.epsilon * 100)

        if success / 8 > 89 and agent.epsilon < 0.4:
            break
        success = 0
        target1_reached = 0

        plot_training_history(epochs, win_percent, win_first_target, [win_loss_actor, win_loss_critic], win_epsilon, win_D1, win_mean_reward,
                              win_collisions, True)

    print('Episodes: ', epochs)
    print('Win percentage: ', win_percent)
    print("Win First target: ", win_first_target)
    print("Epsilon history: ", win_epsilon)

    plot_training_history(epochs, win_percent, win_first_target, [win_loss_actor, win_loss_critic], win_epsilon, win_D1, win_mean_reward,
                          win_collisions)


def plot_training_history(epochs, win_percent, win_first_target, win_loss, win_epsilon, win_D1, win_mean_reward,
                          collisions, display=False):
    fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(18, 12))

    # figure = plt.figure()
    ax[0, 0].plot(epochs, win_percent)
    ax[0, 0].plot(epochs, win_first_target)
    ax[0, 0].plot(epochs, win_epsilon)

    ax[0, 0].legend(["Success %", "First target reached %", "Epsilon*100"])

    ax[0, 0].set_title('DDPG with HER')
    ax[0, 0].set_ylabel('%')
    ax[0, 0].set_xlabel('Number of Epochs')
    ax[0, 0].set_ylim([0, 100])
    # if display:
    #     plt.show()
    #
    # plt.savefig(os.path.join(os.getcwd(), SAVE_PATH) + "winrate.png")
    # figure = plt.figure()

    # subplot 2
    ax[0, 1].plot(epochs, win_mean_reward)
    ax[0, 1].set_title('DDPG with HER')
    ax[0, 1].set_ylabel('Average reward')
    ax[0, 1].set_xlabel('Number of Epochs')

    # subplot 3
    ax[1, 0].plot(epochs, win_loss[0])
    ax[1, 0].plot(epochs, win_loss[1])
    ax[1, 0].plot(epochs, win_D1)

    ax[1, 0].legend(["Actor Loss*1000", "Critic Loss*1000", "D1*100"])

    ax[1, 0].set_title('DDPG with HER')
    ax[1, 0].set_ylabel('%')
    ax[1, 0].set_xlabel('Number of Epochs')
    # ax[1, 0].set_ylim([0, 100])

    # subplot 4
    ax[1, 1].plot(epochs, collisions)
    ax[1, 1].set_title('DDPG with HER')
    ax[1, 1].set_ylabel('Number of collisions')
    ax[1, 1].set_xlabel('Number of Epochs')

    plt.tight_layout()

    if display:
        plt.show()
    plt.savefig(os.path.join(os.getcwd(), SAVE_PATH) + "reward.png")


if __name__ == '__main__':
    # n_bits = 12
    # env = BitFlipEnv.BitFlipEnv(n_bits)
    # env = gym.make('gym_robot_arm:robot-arm-v0')

    if GAZEBO:
        from robot_arm import RobotArm
        import rospy

        rospy.init_node('python_robot_driver')  # needed if running on Gazebo sim
        arm = RobotArm()  # needed if running on Gazebo sim
        env = gym.make('iiwaEnv-v0', tolerance=TOLERANCE, robot=arm, discrete=False, discretized_values=ACTION_SPACE,
                       reward_space=REWARDS, distance_as_reward=DISTANCE_AS_REWARD, fully_sparse_rewards=SPARSE_REWARDS,
                       done_on_collision=DONE_ON_COLLISION, PRINT_TERMINAL=PRINT_TERMINAL)

    else:
        env = gym.make('iiwaEnv-v0', tolerance=TOLERANCE, robot=None, discrete=False,
                       discretized_values=ACTION_SPACE,
                       reward_space=REWARDS, distance_as_reward=DISTANCE_AS_REWARD,
                       fully_sparse_rewards=SPARSE_REWARDS,
                       done_on_collision=DONE_ON_COLLISION, print_terminal=PRINT_TERMINAL)

    # -------------------- #
    # to train the network #
    # -------------------- #
    if TRAIN:
        train_dqn_her(env)

    # ------------------- #
    # to test the network #
    # ------------------- #
    if TEST:
        test_dqn_her(env)
