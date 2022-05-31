from collections import deque, namedtuple
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence


Experience = namedtuple('Experience', field_names=['state', 'action', 'reward',
                               'done', 'new_state'])


class Episode:
    def __init__(self, observations, actions, rewards, dones):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.dones = dones

    def __len__(self):
        return len(self.observations)


class EpisodeReplayBuffer:
    def __init__(self, buffer_capacity, use_n_newest=0):
        self.buffer = deque(maxlen=buffer_capacity)
        self.use_n_newest = use_n_newest

    def __len__(self):
        return len(self.buffer)

    def append(self, episode):
        self.buffer.append(episode)

    def sample(self, sample_size):
        sample_indices = np.random.choice(self.__len__(),
                                          size=sample_size,
                                          replace=False)
        if self.use_n_newest > 0:
            replace_indices = []
            for i in range(1, self.use_n_newest + 1):
                index = self.__len__() - i - 1
                if index not in sample_indices:
                    sample_indices[-i] = index

        episodes = [self.buffer[i] for i in sample_indices]
        episode_lengths = [len(e) for e in episodes]
        seq_len = np.max(episode_lengths)
        fill_mask = torch.zeros((sample_size, seq_len - 1))
        for i in range(sample_size):
            fill_mask[i, :len(episodes[i])] = 1

        obs = [torch.tensor(e.observations, dtype=torch.float32) \
                   for e in episodes]
        a = [torch.tensor(e.actions, dtype=torch.int64) \
                 for e in episodes]
        r = [torch.tensor(e.rewards, dtype=torch.float32) \
                 for e in episodes]
        done = [torch.tensor(e.dones, dtype=torch.int64) \
                    for e in episodes]

        obs = pad_sequence(obs, batch_first=True)
        a = pad_sequence(a, batch_first=True)
        r = pad_sequence(r, batch_first=True)
        done = pad_sequence(done, batch_first=True)

        return obs, a, r, done, fill_mask


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])

        return (np.array(states), np.array(actions), np.array(rewards, dtype=np.float32),
                np.array(dones, dtype=np.bool), np.array(next_states))
