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
from dqn_her import DQNAgentWithHER as dqnHER
import random
import numpy as np
import gym
import BitFlipEnv

SAVE_PATH = '/UM/RL/trained_agents/'
EPOCHS = 80  # each EPOCH consists of CYCLES
CYCLES = 50
EPISODES = 16
STEPS = 50
OPTIMIZATION_STEPS = 40
BATCH_SIZE = 128
MEMORY_SIZE = 1e6
DECAY_EPS = 0.95
LR = 0.0001
DISCOUNT_FACTOR = 0.98
EPSILON = 0.9
n_goals = 4


def train_dqn_her(env_):
    env = env_
    state_size = env.reset().shape[0]
    STEPS = state_size #TODO: delete this line if you are not using the BitFlip env
    epsilon_history = []
    epochs = []

    win_percent = []
    success = 0

    load_checkpoint = False
    checkpoint_dir = os.path.join(os.getcwd(), SAVE_PATH)

    # Initializes the DQN agent with  Hindsight Experience Replay
    agent = dqnHER.DQNAgentWithHER(learning_rate=LR, n_actions=env.action_space.n,
                                   input_dims=state_size, gamma=DISCOUNT_FACTOR,
                                   epsilon=EPSILON, batch_size=BATCH_SIZE, memory_size=int(MEMORY_SIZE),
                                   replace_network_count=50,
                                   checkpoint_dir=checkpoint_dir)

    if load_checkpoint:
        agent.load_model()

    for epoch in range(EPOCHS):
        for cycle in range(CYCLES):
            # for episode = 1, M do
            for episode in range(EPISODES):

                # Sample a goal g and an initial state s0
                state = env.reset()
                goal = env.goal
                done = False
                episode_transitions = []

                # for t = 0, T − 1 do
                for p in range(STEPS):
                    if not done:
                        # Sample an action at using the behavioral policy from A
                        action = agent.choose_action(state, goal)
                        # Execute the action at and observe a new state st+1
                        next_state, reward, done, info = env.step(action)
                        if not load_checkpoint:
                            # Store the transition (st||g, at, rt, st+1||g) in R
                            # standard experience replay
                            agent.store_experience(state, action, reward, next_state, done, goal)
                            episode_transitions.append((state, action, reward, next_state, info))

                        state = next_state

                        if done:
                            success += 1
                            break

                if not done:
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

                            #TODO: need to define evaluate() for each environment you try (see BitFlipEnv)
                            new_reward, new_done = env.evaluate(action_, state_, new_goal)

                            # Store the transition (st||g', at, r', st+1||g') in R
                            agent.store_experience(state, action_, new_reward, next_state_, new_done, new_goal)

                for s in range(OPTIMIZATION_STEPS):
                    agent.learn()

        # Average during EPOCH
        #if episode > 0 and episode % 512 == 0:
        print("Epoch", epoch)
        print('success rate for last 800 episodes after', (epoch+1)*CYCLES*EPISODES, ':', success / 8, ", current epsilon: ", agent.epsilon)
        if len(win_percent) > 0 and (success / 800) > win_percent[len(win_percent) - 1]:
            agent.save_model()
        epsilon_history.append(agent.epsilon )
        epochs.append(episode)
        win_percent.append(success / 800)
        success = 0

    print('Epsilon History:', epsilon_history)
    print('Episodes:', epochs)
    print('Win percentage:', win_percent)

    figure = plt.figure()
    plt.plot(epochs, win_percent)

    plt.title('DQN with HER')
    plt.ylabel('Win Percentage')
    plt.xlabel('Number of Episodes')
    plt.ylim([0, 1])

    plt.savefig(os.path.join(os.getcwd(), SAVE_PATH))


if __name__ == '__main__':
    n_bits = 4
    env = BitFlipEnv.BitFlipEnv(n_bits)
    # env = gym.make('gym_robot_arm:robot-arm-v0')
    # env = gym.make('CartPole-v0')
    train_dqn_her(env)