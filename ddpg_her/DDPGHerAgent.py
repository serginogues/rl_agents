import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3.common.running_mean_std import RunningMeanStd

import ddpg_her.HERMemory as her
import ddpg_her.OUNoise as noise


def normalize(x, stats, clip_range):
    if stats is None:
        return x
    return np.clip((x - stats.mean) / (np.sqrt(stats.var) + 1e-8), -clip_range, clip_range)


class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, checkpoint_dir='../weights',
                 filename='ddpg_actor_weights'):
        """
        :param input_size: state size
        :param hidden_size: hidden size
        :param output_size: action size
        :param output_range: output range
        """
        super(Actor, self).__init__()
        if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.checkpoint_file = os.path.join(checkpoint_dir, filename)

        self.linear1 = nn.Linear(input_size, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
        self.norm3 = nn.LayerNorm(output_size)

        self.device_type = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.device_type)
        self.to(self.device)
        print(self.device)

    def forward(self, s):
        x = torch.tanh(self.linear1(s))
        x = torch.tanh(self.linear2(x))
        x = torch.tanh(self.linear3(x))
        return x

    def save_checkpoint(self):
        # print('... Save checkpoint ...')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        # print('... Load checkpoint ...')
        self.load_state_dict(torch.load(self.checkpoint_file))


class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, checkpoint_dir='../weights',
                 filename='ddpg_critic_weights'):
        super().__init__()
        if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.checkpoint_file = os.path.join(checkpoint_dir, filename)

        self.linear1 = nn.Linear(input_size, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

        self.device_type = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.device_type)
        self.to(self.device)
        print(self.device)

    def forward(self, s, a):
        # print(s.shape, a.shape)
        x = torch.cat([s, a], 1)
        x = torch.tanh(self.linear1(x))
        x = torch.tanh(self.linear2(x))
        x = self.linear3(x)
        return x

    def save_checkpoint(self):
        # print('... Save checkpoint ...')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        # print('... Load checkpoint ...')
        self.load_state_dict(torch.load(self.checkpoint_file))


class Agent(object):
    def __init__(self, actor_learning_rate, critic_learning_rate, n_actions,
                 input_dims, n_goal, gamma, memory_size, batch_size, tau=0.001,
                 epsilon=0.9, dec_epsilon=1e-7, min_epsilon=0.2, soft_update_step=900, is_obs_norm=True, model_name='ddpg'):

        self.s_dim = input_dims
        self.a_dim = n_actions
        self.g_dim = n_goal

        self.min_epsilon = min_epsilon
        self.epsilon = epsilon
        self.dec_epsilon = dec_epsilon
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.soft_update_interval = soft_update_step
        self.model_name = model_name
        self.K = 1 / self.epsilon if self.epsilon > 0.1 else 10

        self.actor = Actor(input_size=self.s_dim + self.g_dim, hidden_size=64, output_size=self.a_dim, filename='{}_actor.pth'.format(model_name))
        self.actor_target = Actor(input_size=self.s_dim + self.g_dim, hidden_size=64, output_size=self.a_dim, filename='{}_actor_target.pth'.format(model_name))
        self.critic = Critic(input_size=self.s_dim + self.g_dim + self.a_dim, hidden_size=64, output_size=1, filename='{}_critic.pth'.format(model_name))
        self.critic_target = Critic(input_size=self.s_dim + self.g_dim + self.a_dim, hidden_size=64, output_size=1, filename='{}_critic_target.pth'.format(model_name))
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)
        self.memory = her.HindsightExperienceReplayMemory(memory_size=memory_size,
                                                          input_dims=self.s_dim, n_actions=self.a_dim, n_goal=self.g_dim)

        self.ou_noise = noise.OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.a_dim))

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.counter_soft_update = 0

        self.actor_loss = 0
        self.critic_loss = 0

        self.is_obs_norm = is_obs_norm
        self.obs_norm_clip_range = 5
        self.goal_norm_clip_range = 5
        self.checkpoint_file = os.path.join('../weights', self.model_name)

        if self.is_obs_norm:
            self.obs_rms = RunningMeanStd(shape=self.s_dim)
            self.goal_rms = RunningMeanStd(shape=self.g_dim)
        else:
            self.obs_rms = None
            self.goal_rms = None

    def store_experience(self, state, action, reward, next_state, done, goal, norm_update=True):
        """
        Saves the experience to the hindsight replay memory
        """
        self.memory.add_experience(state=state, action=action,
                                   reward=reward, next_state=next_state,
                                   done=done, goal=goal)
        if self.is_obs_norm:
            if norm_update:
                self.obs_rms.update(state)
                self.goal_rms.update(goal)

    def get_sample_experience(self):
        """
        Gives a sample experience from the hindsight replay memory
        """
        state, action, reward, next_state, done, goal = self.memory.get_random_experience(
            self.batch_size)

        t_state = torch.tensor(state).to(self.actor.device)
        t_action = torch.tensor(action).to(self.actor.device)
        t_reward = torch.tensor(reward).to(self.actor.device)
        t_next_state = torch.tensor(next_state).to(self.actor.device)
        t_done = torch.tensor(done).to(self.actor.device)
        t_goal = torch.tensor(goal).to(self.actor.device)

        return t_state, t_action, t_reward, t_next_state, t_done, t_goal

    def choose_action(self, observation, goal):
        # s0 = torch.tensor(s0, dtype=torch.float).unsqueeze(0)
        # a0 = self.actor(s0).squeeze(0).detach().numpy()
        # return a0

        if np.random.random() > self.epsilon:
            # Exploitation
            if self.is_obs_norm:
                observation = normalize(observation, self.obs_rms, self.obs_norm_clip_range)
                goal = normalize(goal, self.goal_rms, self.goal_norm_clip_range)
            state = torch.tensor([np.concatenate([observation, goal])], dtype=torch.float).unsqueeze(0).to(self.actor.device)
            action = (self.actor.forward(state) + torch.tensor(self.ou_noise(), dtype=torch.float).to(self.actor.device)).cpu().squeeze(0).detach().numpy()[0]
        else:
            # Exploration
            action = np.random.random(self.a_dim) * 2 - 1
        return action

    @staticmethod
    def action_to_discrete(state, continuous_action):
        #TODO: this line is exclusive for gym_robot_arm:robot-arm-v1
        action = 4 * state[2:] / np.pi - 1 + continuous_action * 0.2
        return action

    def decrement_epsilon(self):
        """
        Decrements the epsilon after each step till it reaches minimum epsilon (0.1)
        epsilon = epsilon - decrement (default is 1e-6)
        """
        # self.epsilon = self.epsilon - self.dec_epsilon if self.epsilon > self.min_epsilon \
        #     else self.min_epsilon
        # self.epsilon = self.epsilon * self.dec_epsilon if self.epsilon > self.min_epsilon else self.min_epsilon

        self.epsilon = 1 / self.K if self.epsilon > self.min_epsilon else self.min_epsilon

    def learn(self):
        if self.memory.counter < self.batch_size:
            return

        observation, action, reward, next_observation, done, goal = self.get_sample_experience()
        if self.is_obs_norm:
            observation = normalize(observation.cpu(), self.obs_rms, self.obs_norm_clip_range).to(self.actor.device)
            next_observation = normalize(next_observation.cpu(), self.obs_rms, self.obs_norm_clip_range).to(self.actor.device)
            goal = normalize(goal.cpu(), self.goal_rms, self.goal_norm_clip_range).to(self.actor.device)
        concat_observation_goal = torch.cat((observation, goal), 1).float().to(self.actor.device)
        concat_next_observation_goal = torch.cat((next_observation, goal), 1).float().to(self.actor.device)

        self.critic_optim.zero_grad()
        self.actor_optim.zero_grad()

        # critic_learn()
        target_actions = self.actor_target.forward(concat_observation_goal).detach()  # a1
        critic_next_value = self.critic_target.forward(concat_next_observation_goal, target_actions).view(-1)

        y_true = (reward + self.gamma * critic_next_value).view(self.batch_size, -1)
        y_pred = self.critic.forward(concat_observation_goal, action)  # critic_value
        loss_fn = nn.MSELoss()
        loss_critic = loss_fn(y_pred, y_true)
        self.critic_loss = loss_critic.item()
        loss_critic.backward()
        self.critic_optim.step()

        # actor_learn()
        actor_value = self.actor(concat_observation_goal)
        loss = - torch.mean(self.critic.forward(concat_observation_goal, actor_value))
        loss.backward()
        self.actor_loss = loss.item()
        self.actor_optim.step()

        self.decrement_epsilon()
        self.K += self.dec_epsilon

    @staticmethod
    def soft_update(net_target, net, tau):
        for target_param, param in zip(net_target.parameters(), net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def soft_update_all(self):
        self.soft_update(self.critic_target, self.critic, self.tau)
        self.soft_update(self.actor_target, self.actor, self.tau)

        # if self.counter_soft_update % self.soft_update_interval == 0:
        #     soft_update(self.critic_target, self.critic, self.tau)
        #     soft_update(self.actor_target, self.actor, self.tau)

        self.counter_soft_update += 1

        # return self.actor_los, self.critic_loss

    def save_model(self):
        self.actor.save_checkpoint()
        self.actor_target.save_checkpoint()
        self.critic.save_checkpoint()
        self.critic_target.save_checkpoint()

        if self.is_obs_norm:
            torch.save({'obs_rms_mean': self.obs_rms.mean,
                        'obs_rms_var': self.obs_rms.var,
                        'obs_rms_count': self.obs_rms.count,
                        'goal_rms_mean': self.goal_rms.mean,
                        'goal_rms_var': self.goal_rms.var,
                        'goal_rms_count': self.goal_rms.count},
                       '{}_rms_stat.pth'.format(self.checkpoint_file)
                       )

    def load_model(self):
        self.actor.load_checkpoint()
        self.actor_target.load_checkpoint()
        self.critic.load_checkpoint()
        self.critic_target.load_checkpoint()
        if self.is_obs_norm:
            rms = torch.load('{}_rms_stat.pth'.format(self.checkpoint_file))
            self.goal_rms.mean = rms['goal_rms_mean']
            self.goal_rms.var = rms['goal_rms_var']
            self.goal_rms.count = rms['goal_rms_count']
            self.obs_rms.mean = rms['obs_rms_mean']
            self.obs_rms.var = rms['obs_rms_var']
            self.obs_rms.count = rms['obs_rms_count']


"""for episode in range(4000):
    s0 = env.reset()
    episode_reward = 0
    agent.decrement_epsilon()
    for step in range(300):
        if episode > 3800:
            env.render()
        a0 = agent.choose_action(s0)
        action = agent.action_to_discrete(s0, a0)
        s1, r1, done, _ = env.step(action)
        # print(a0_actual)
        # print(step, done, s0, a0, r1)
        agent.put(s0, a0, r1, s1)
        episode_reward += r1
        s0 = s1
        agent.learn()"""
