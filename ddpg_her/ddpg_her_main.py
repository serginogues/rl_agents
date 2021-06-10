"""
Based on https://github.com/hemilpanchiwala/Hindsight-Experience-Replay

Set in SAVE_PATH the desired directory for the trained agent
Set load_checkpoint = True to Test trained agent
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from ddpg_her import DDPGAgent

SAVE_PATH = '/UM/RL/trained_agents/'


def train_ddpg_her(env_, state_space=4):
    """
    i.e. env_ = gym.make('gym_robot_arm:robot-arm-v0')
    """
    env = env_

    n_episodes = 10000
    print(n_episodes)
    episodes = []
    win_percent = []
    success = 0
    load_checkpoint = False
    checkpoint_dir = os.path.join(os.getcwd(), SAVE_PATH)

    # Initializes the DDPG agent
    agent = DDPGAgent.DDPGAgent(actor_learning_rate=0.0001, critic_learning_rate=0.001, n_actions=env.action_space.n,
                                input_dims=state_space, gamma=0.99,
                                memory_size=10000, batch_size=64,
                                checkpoint_dir=checkpoint_dir)

    if load_checkpoint:
        agent.load_model()

    # Iterate through the episodes
    for episode in range(n_episodes):
        state = env.reset()
        goal = env.goal
        done = False
        transitions = []

        for p in range(10):
            if not done:
                action = agent.choose_action(state, goal)
                next_state, reward, done, info = env.step(np.argmax(action), goal)
                if not load_checkpoint:
                    agent.store_experience(state, action, reward, next_state, done, goal)
                    transitions.append((state, action, reward, next_state))
                    agent.learn()
                state = next_state

                if done:
                    success += 1

        if not done:
            # The new goal is the current state
            new_goal = np.copy(state)
            if not np.array_equal(new_goal, goal):
                for q in range(4):
                    # state, action, reward, next_state
                    transition = transitions[q]

                    # check if done WITH NEW GOAL
                    observation_, reward_, done_, info_ = env.step(np.argmax(transition[1]), new_goal)

                    if done_:
                        agent.store_experience(transition[0], transition[1], transition[2], transition[3], True, new_goal)
                    else:
                        agent.store_experience(transition[0], transition[1], transition[2], transition[3], False, new_goal)
                    agent.learn()

        # Average over last 100 episodes to avoid spikes
        if episode > 0 and episode % 100 == 0:
            print('success rate for last 100 episodes after', episode, ':', success)
            if len(win_percent) > 0 and (success / 100) > win_percent[len(win_percent) - 1]:
                agent.save_model()
            episodes.append(episode)
            win_percent.append(success / 100)
            success = 0

    print('Episodes:', episodes)
    print('Win percentage:', win_percent)

    figure = plt.figure()
    plt.plot(episodes, win_percent)

    plt.title('DDPG with HER')
    plt.ylabel('Win Percentage')
    plt.xlabel('Number of Episodes')
    plt.ylim([0, 1])

    plt.savefig(os.path.join(os.getcwd(), SAVE_PATH))
