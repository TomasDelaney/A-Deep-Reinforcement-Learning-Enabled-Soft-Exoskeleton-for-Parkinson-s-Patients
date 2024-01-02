import numpy as np
import torch


class LAP(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            device,
            num_envs,
            max_size=1e6,
            batch_size=64,
            max_action=1,
            normalize_actions=True,
            prioritized=True
    ):

        max_size = int(max_size)
        self.max_size = max_size
        self.max_action = max_action
        self.ptr = 0
        self.count = 0
        self.size = 0

        self.device = device
        self.batch_size = batch_size
        self.num_envs = num_envs

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.state = np.zeros((num_envs, max_size, state_dim))
        self.action = np.zeros((num_envs, max_size, action_dim))
        self.next_state = np.zeros((num_envs, max_size, state_dim))
        self.reward = np.zeros((num_envs, max_size, 1))
        self.not_done = np.zeros((num_envs, max_size, 1))

        if prioritized:
            self.prioritized = True
            self.priority = torch.zeros((num_envs, max_size), device=device)
            self.max_priority = 1
            self.priority_indexes = []
        else:
            self.prioritized = False

        self.normalize_actions = max_action if normalize_actions else 1

    def add(self, state, action, next_state, reward, done, tremor_num):
        self.state[tremor_num, self.ptr] = state
        self.action[tremor_num, self.ptr] = action / self.normalize_actions
        self.next_state[tremor_num, self.ptr] = next_state
        self.reward[tremor_num, self.ptr] = reward
        self.not_done[tremor_num, self.ptr] = 1. - done

        if self.prioritized:
            self.priority[tremor_num, self.ptr] = self.max_priority

        if self.count % self.num_envs == 0:
            self.ptr = (self.ptr + 1) % self.max_size
            self.size = min(self.size + 1, self.max_size)

        self.count += 1

    def sample(self):

        if self.prioritized:
            # declare return values
            states = np.zeros((self.num_envs*self.batch_size, self.state_dim))
            actions = np.zeros((self.num_envs*self.batch_size, self.action_dim))
            next_state = np.zeros((self.num_envs*self.batch_size, self.state_dim))
            reward = np.zeros((self.num_envs*self.batch_size, 1))
            not_done = np.zeros((self.num_envs*self.batch_size, 1))

            for i in range(self.num_envs):
                csum = torch.cumsum(self.priority[i, :self.size], 0)
                val = torch.rand(size=(self.batch_size,), device=self.device) * csum[-1]
                self.ind = torch.searchsorted(csum, val).cpu().data.numpy()
                self.priority_indexes.append(self.ind)

                states[i*self.batch_size:(i+1)*self.batch_size] = self.state[i, self.ind]
                actions[i*self.batch_size:(i+1)*self.batch_size] = self.action[i, self.ind]
                next_state[i*self.batch_size:(i+1)*self.batch_size] = self.next_state[i, self.ind]
                reward[i*self.batch_size:(i+1)*self.batch_size] = self.reward[i, self.ind]
                not_done[i*self.batch_size:(i+1)*self.batch_size] = self.not_done[i, self.ind]

        else:
            self.ind = np.random.randint(0, self.size, size=self.batch_size)
            states = self.state[:, self.ind].reshape(-1, self.state_dim)
            actions = self.action[:, self.ind].reshape(-1, self.action_dim)
            next_state = self.next_state[:, self.ind].reshape(-1, self.state_dim)
            reward = self.reward[:, self.ind].reshape(-1, 1)
            not_done = self.not_done[:, self.ind].reshape(-1, 1)

            # reshuffle the elements
            index_array = np.arange(states.shape[0])
            np.random.shuffle(index_array)

            states = states[index_array]
            actions = actions[index_array]
            next_state = next_state[index_array]
            reward = reward[index_array]
            not_done = not_done[index_array]

        return (
            torch.tensor(states, dtype=torch.float, device=self.device),
            torch.tensor(actions, dtype=torch.float, device=self.device),
            torch.tensor(next_state, dtype=torch.float, device=self.device),
            torch.tensor(reward, dtype=torch.float, device=self.device),
            torch.tensor(not_done, dtype=torch.float, device=self.device)
        )

    def update_priority(self, priority):
        for i in range(self.num_envs):
            self.priority[i, self.priority_indexes[i]] = priority[i*self.batch_size:(i+1)*self.batch_size].reshape(-1).detach()
        self.max_priority = max(float(priority.max()), self.max_priority)
        self.priority_indexes.clear()

    def reset_max_priority(self):  # change the array indexing when setting them to 0.
        self.max_priority = float(self.priority[:, :].max())

    def reset_buffer(self):
        self.ptr = 0
        self.count = 0
        self.size = 0

        self.state[:, :, :] = 0
        self.action[:, :, :] = 0
        self.next_state[:, :, :] = 0
        self.reward[:, :, :] = 0
        self.not_done[:, :, :] = 0

        if self.prioritized:
            self.prioritized = True
            self.priority = torch.zeros((self.num_envs, self.max_size), device=self.device)
            self.max_priority = 1
            self.priority_indexes = []
        else:
            self.prioritized = False

    def load_D4RL(self, dataset):
        self.state = dataset['observations']
        self.action = dataset['actions']
        self.next_state = dataset['next_observations']
        self.reward = dataset['rewards'].reshape(-1, 1)
        self.not_done = 1. - dataset['terminals'].reshape(-1, 1)
        self.size = self.state.shape[0]

        if self.prioritized:
            self.priority = torch.ones(self.size).to(self.device)