"""
Test repo on different RL agents and environments
"""
import gym
from ddpg_her.ddpg_her_main import train_ddpg_her
from dqn_her.dqn_her_main import train_dqn_her
import numpy as np


def test_robot2d_env():

    env = gym.make('gym_robot_arm:robot-arm-v0')

    for i_episode in range(20):
        observation = env.reset()
        goal = env.goal
        for t in range(100):
            env.render()
            print(observation)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(np.argmax(action), goal)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
    env.close()


if __name__ == '__main__':
    # env = gym.make('gym_robot_arm:robot-arm-v0')
    env = gym.make('CartPole-v0')
    #train_ddpg_her(env)
    train_dqn_her(env)
    # test_robot2d_env()
