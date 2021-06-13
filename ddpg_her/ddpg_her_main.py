"""
Based on https://github.com/hemilpanchiwala/Hindsight-Experience-Replay
"""
import os
import gym
import matplotlib.pyplot as plt
import numpy as np
from ddpg_her.DDPGHerAgent import Agent
import random

SAVE_PATH = '/UM/RL/trained_agents/'
EPOCHS = 200  # each EPOCH consists of CYCLES
CYCLES = 50
EPISODES = 16
STEPS = 600
OPTIMIZATION_STEPS = 40
BATCH_SIZE = 128
MEMORY_SIZE = 100000
DECAY_EPS = 1e-5  # 0.95
LR_actor = 0.001
LR_critic = 0.001
DISCOUNT_FACTOR = 0.99
EPSILON = 0.9
SOFT_UPDATE_STEP = 900
n_goals = 10


def train_ddpg_her(env):
    """
    i.e. env_ = gym.make('gym_robot_arm:robot-arm-v0')
    """

    state_size = env.reset().shape[0]
    action_size = env.action_space.shape[0]
    epochs = []
    win_percent = []
    success = 0

    # Initializes the DDPG agent
    agent = Agent(actor_learning_rate=LR_actor, critic_learning_rate=LR_critic,
                  n_actions=action_size, input_dims=state_size,
                  gamma=DISCOUNT_FACTOR, memory_size=int(MEMORY_SIZE),
                  batch_size=BATCH_SIZE, dec_epsilon=DECAY_EPS, soft_update_step=SOFT_UPDATE_STEP)

    for epoch in range(EPOCHS):
        for cycle in range(CYCLES):
            # for episode = 1, M do
            for episode in range(EPISODES):
                # Sample a goal g and an initial state s0
                state = env.reset()
                goal = env.goal
                done = False
                episode_transitions = []

                for p in range(STEPS):
                    if not done:
                        # Sample an action at using the behavioral policy from A
                        action = agent.choose_action(state, goal)

                        # Execute the action at and observe a new state st+1
                        if "gym_robot_arm:robot-arm-v1" in str(env):
                            next_state, reward, done, info = env.step(agent.action_to_discrete(state, action))
                        else:
                            next_state, reward, done, info = env.step(action)
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

                            # TODO: need to define evaluate() for each environment you try (see BitFlipEnv)
                            if "BitFlipEnv" in str(env):
                                new_reward, new_done = env.evaluate(action_, state_, new_goal)
                            elif "RobotArmEnv" in str(env):
                                new_reward, new_done = env.evaluate(new_goal, info_)
                            else:
                                new_reward, new_done = env.evaluate(action_, state_, new_goal)

                            # Store the transition (st||g', at, r', st+1||g') in R
                            agent.store_experience(state, action_, new_reward, next_state_, new_done, new_goal)

            for s in range(OPTIMIZATION_STEPS):
                agent.learn()

        print("Epoch", epoch)
        print('success rate for last 800 episodes after', (epoch + 1) * CYCLES * EPISODES, ':', success / 8, ", current epsilon: ", agent.epsilon)
        if len(win_percent) > 0 and (success / 800) > win_percent[len(win_percent) - 1]:
            agent.save_model()
        epochs.append(epoch)
        win_percent.append(success / 800)
        success = 0

    print('Episodes:', epochs)
    print('Win percentage:', win_percent)

    figure = plt.figure()
    plt.plot(epochs, win_percent)

    plt.title('DDPG with HER')
    plt.ylabel('Win Percentage')
    plt.xlabel('Number of Episodes')
    plt.ylim([0, 1])

    plt.savefig(os.path.join(os.getcwd(), SAVE_PATH))


if __name__ == '__main__':
    # env = gym.make('gym_robot_arm:robot-arm-v1')
    env = gym.make('MountainCarContinuous-v0')
    train_ddpg_her(env)