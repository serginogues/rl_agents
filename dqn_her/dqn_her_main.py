"""
DQN-HER agent based on https://github.com/hemilpanchiwala/Hindsight-Experience-Replay

ENVIRONMENT CONSTRAINTS
- env.action_space.n    returns action space size
- env.reset()           returns current state
- env.goal              returns current goal
- env.step(action)      returns next_state, reward, done, info and type(action) = int

Define SAVE_PATH before running
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from dqn_her import DQNAgentWithHER as dqnHER

SAVE_PATH = '/UM/RL/trained_agents/'


def train_dqn_her(env_, state_space=4):

    env = env_
    n_episodes = 30000
    epsilon_history = []
    episodes = []
    win_percent = []
    success = 0
    load_checkpoint = False
    checkpoint_dir = os.path.join(os.getcwd(), SAVE_PATH)

    # Initializes the DQN agent with  Hindsight Experience Replay
    agent = dqnHER.DQNAgentWithHER(learning_rate=0.0001, n_actions=env.action_space.n,
                                   input_dims=state_space, gamma=0.99,
                                   epsilon=0.9, batch_size=64, memory_size=10000,
                                   replace_network_count=50,
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
                next_state, reward, done, info = env.step(action)
                goal = env.goal
                if not load_checkpoint:
                    agent.store_experience(state, action, reward, next_state, done, goal)
                    transitions.append((state, action, reward, next_state))
                    agent.learn()
                state = next_state

                if done:
                    success += 1
                    break

        if not done:
            new_goal = np.copy(state)
            if not np.array_equal(new_goal, goal):
                for p in range(10):
                    transition = transitions[p]

                    # check if done WITH NEW GOAL
                    observation_, reward_, done_, info_ = env.step(transition[1])

                    if done_:
                        agent.store_experience(transition[0], transition[1], transition[2], transition[3], True,
                                               new_goal)
                    else:
                        agent.store_experience(transition[0], transition[1], transition[2], transition[3], False,
                                               new_goal)
                    agent.learn()

        # Average over last 500 episodes to avoid spikes
        if episode % 500 == 0:
            print('success rate for last 500 episodes after', episode, ':', success / 5)
            if len(win_percent) > 0 and (success / 500) > win_percent[len(win_percent) - 1]:
                agent.save_model()
            epsilon_history.append(agent.epsilon)
            episodes.append(episode)
            win_percent.append(success / 500.0)
            success = 0

    print('Epsilon History:', epsilon_history)
    print('Episodes:', episodes)
    print('Win percentage:', win_percent)

    figure = plt.figure()
    plt.plot(episodes, win_percent)

    plt.title('DQN with HER')
    plt.ylabel('Win Percentage')
    plt.xlabel('Number of Episodes')
    plt.ylim([0, 1])

    plt.savefig(os.path.join(os.getcwd(), SAVE_PATH))
