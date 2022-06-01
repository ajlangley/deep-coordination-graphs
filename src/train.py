from datetime import datetime as dt
import numpy as np
import torch

from src.replay_buffer import *

def train_q_learner(q_learner, env, n_episodes=10**6, batch_size=32, update_target_every=200,
                    buffer_capacity=10**6, eps_max=1, eps_min=0.05, eps_decay=0.99, eval_every=50,
                    n_eval=1, save_every=None, save_path=None, device=torch.device('cuda')):
    t0 = dt.now()
    eval_means = []
    replay_buffer = ReplayBuffer(buffer_capacity)
    q_learner.to(device)
    try:
        for episode in range(n_episodes):
            eps = max(eps_max * eps_decay ** episode, eps_min)
            obs = env.reset()
            done = False
            while done == False:
                # Experience generation
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
                a = q_learner.eps_greedy_action(obs_tensor, eps)
                obs_new, r, done, info = env.step(a)
                experience = Experience(obs, a, r, done, obs_new)
                replay_buffer.append(experience)
                obs = obs_new

                # Learning step
                if len(replay_buffer) >= batch_size:
                    q_learner.train(*replay_buffer.sample(batch_size))

            msg = f'Episode: [{episode + 1}]/[{n_episodes}], Time Elapsed: {dt.now() - t0}'
            # print(msg, end='\r')

            if episode % eval_every == 0:
                eval_means.append(q_learner.evaluate(n_eval))
                # print(f'\n\tEval. Reward: {eval_means[-1]}')
                print(f'[{episode}/{n_episodes}] Eval. reward: {eval_means[-1]}, Time Elapsed: {dt.now() - t0}')
            if episode % update_target_every == 0:
                q_learner.update_target_net()
            if save_every is not None and episode % save_every == 0:
                pass

    except KeyboardInterrupt:
        pass

    return np.array(eval_means)

from datetime import datetime as dt

def train_recurrent_q_learner(q_learner, env, n_episodes=10**6, buffer_capacity=5e2, batch_size=32,
                              eps_min=0.05, eps_max=1, eps_decay=0.99, update_target_every=200,
                              use_n_newest=1, eval_every=100, n_eval=15, device=torch.device('cuda')):
    t0 = dt.now()
    eval_means = []
    replay_buffer = EpisodeReplayBuffer(int(buffer_capacity), use_n_newest=use_n_newest)
    q_learner.to(device)

    try:
        for episode in range(n_episodes):
            eps = max(eps_min, eps_max * eps_decay ** episode)
            obs, a, r, done = q_learner.generate_eps_greedy_trajectory(eps)
            replay_buffer.append(Episode(obs, a, r, done))
            msg = f'Episode: [{episode + 1}]/[{n_episodes}], Time Elapsed: {dt.now() - t0}'
            print(msg, end='\r')
            if len(replay_buffer) >= batch_size:
                q_learner.train(*replay_buffer.sample(batch_size))
            if episode % eval_every == 0:
                eval_means.append(q_learner.evaluate(n_eval))
                print(f'\n\tEval. Reward: {eval_means[-1]}')
            if episode % update_target_every == 0:
                q_learner.update_target_net()
#             if save_every is not None and episode % save_every == 0:
#                 pass
    except KeyboardInterrupt:
        pass

    return np.array(eval_means)
