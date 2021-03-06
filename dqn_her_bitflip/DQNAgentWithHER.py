import dqn_her_bitflip.DeepQNetwork as dqn
import dqn_her_bitflip.HERMemory as her
import numpy as np
import torch


class DQNAgentWithHER(object):
    def __init__(self, learning_rate, n_actions, input_dims, gamma,
                 epsilon, batch_size, memory_size, replace_network_count,
                 dec_epsilon, min_epsilon, checkpoint_dir='/tmp/ddqn/'):
        self.learning_rate = learning_rate
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.replace_network_count = replace_network_count
        self.dec_epsilon = dec_epsilon
        self.min_epsilon = min_epsilon
        self.checkpoint_dir = checkpoint_dir
        self.action_indices = [i for i in range(n_actions)]
        self.learn_steps_count = 0
        self.K = 1 / self.epsilon

        self.q_eval = dqn.DeepQNetwork(learning_rate=learning_rate, n_actions=n_actions,
                                       input_dims=2*input_dims, name='q_eval',
                                       checkpoint_dir=checkpoint_dir)

        self.q_next = dqn.DeepQNetwork(learning_rate=learning_rate, n_actions=n_actions,
                                       input_dims=2*input_dims, name='q_next',
                                       checkpoint_dir=checkpoint_dir)

        self.experience_replay_memory = her.HindsightExperienceReplayMemory(memory_size=memory_size,
                                                                            input_dims=input_dims,
                                                                            n_actions=n_actions)

        self.min_loss = 0

    def decrement_epsilon(self):
        """
        Decrements the epsilon after each step till it reaches minimum epsilon (0.1)
        epsilon = epsilon - decrement (default is 1e-5)
        """
        # self.epsilon = self.epsilon - self.dec_epsilon if self.epsilon > self.min_epsilon \
        self.epsilon = 1/self.K if self.epsilon > self.min_epsilon \
            else self.min_epsilon

    def store_experience(self, state, action, reward, next_state, done, goal):
        """
        Saves the experience to the hindsight experience replay memory
        """
        self.experience_replay_memory.add_experience(state=state, action=action,
                                                     reward=reward, next_state=next_state,
                                                     done=done, goal=goal)

    def get_sample_experience(self):
        """
        Gives a sample experience from the hindsight experience replay memory
        """
        state, action, reward, next_state, done, goal = self.experience_replay_memory.get_random_experience(self.batch_size)

        t_state = torch.tensor(state).to(self.q_eval.device)
        t_action = torch.tensor(action).to(self.q_eval.device)
        t_reward = torch.tensor(reward).to(self.q_eval.device)
        t_next_state = torch.tensor(next_state).to(self.q_eval.device)
        t_done = torch.tensor(done).to(self.q_eval.device)
        t_goal = torch.tensor(goal).to(self.q_eval.device)

        return t_state, t_action, t_reward, t_next_state, t_done, t_goal

    def replace_target_network(self):
        """
        Updates the parameters after replace_network_count steps
        """
        if self.learn_steps_count % self.replace_network_count == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def choose_action(self, observation, goal):
        """
        Chooses an action with epsilon-greedy method
        """
        if np.random.random() > self.epsilon:
            concat_state_goal = np.concatenate([observation, goal])
            state = torch.tensor([concat_state_goal], dtype=torch.float).to(self.q_eval.device)
            actions = self.q_eval.forward(state)

            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.n_actions)

        return action

    def learn(self):
        if self.experience_replay_memory.counter < self.batch_size:
            return

        self.q_eval.optimizer.zero_grad()
        self.q_next.optimizer.zero_grad()
        self.replace_target_network()

        # Sample minibatch of transitions
        state, action, reward, next_state, done, goal = self.get_sample_experience()
        # Gets the evenly spaced batches
        batches = np.arange(self.batch_size)

        # current state
        concat_state_goal = torch.cat((state, goal), 1)
        # next state
        concat_next_state_goal = torch.cat((next_state, goal), 1)

        # q_value
        q_pred = self.q_eval.forward(concat_state_goal)[batches, action]
        # q_value_next
        q_next = self.q_next.forward(concat_next_state_goal).max(dim=1)[0]

        # y = r,                        if done
        # y = r + gamma*q_value_next,   otherwise
        q_next[done] = 0.0
        q_target = reward + self.gamma * q_next

        # Computes loss between q_target and q_value_i
        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        self.min_loss = np.copy(loss.cpu().detach().numpy())

        # backpropagation
        loss.backward()

        self.q_eval.optimizer.step()
        self.decrement_epsilon()
        self.learn_steps_count += 1
        self.K += self.dec_epsilon

    def save_model(self):
        """
        Saves the values of q_eval and q_next at the checkpoint
        """
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_model(self):
        """
        Loads the values of q_eval and q_next at the checkpoint
        """
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()
