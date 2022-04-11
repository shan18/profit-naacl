import math
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from memory import RingBuffer, SequentialMemory
from random_process import OrnsteinUhlenbeckProcess
from configs_stock import N_DAYS
from util import soft_update, hard_update, to_tensor, to_numpy


criterion = nn.MSELoss()


class DDPG:

    def __init__(self, nb_states, nb_actions, args):
        if args.seed > 0:
            self.seed(args.seed)
        if args.model == 'profit':
            from model import Actor, Critic
        self.nb_states = nb_states
        self.nb_actions = nb_actions
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Create Actor and Critic Network
        self.actor = Actor().to(self.device)
        self.actor_target = Actor().to(self.device)
        self.actor_optim = Adam(self.actor.parameters(), lr=args.prate)

        self.critic = Critic().to(self.device)
        self.critic_target = Critic().to(self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.rate)

        hard_update(
            self.actor_target, self.actor
        )  # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)

        # Create replay buffer
        self.memory = SequentialMemory(
            limit=args.rmsize, window_length=args.window_length
        )
        self.random_process = OrnsteinUhlenbeckProcess(
            size=nb_actions, theta=args.ou_theta, mu=args.ou_mu, sigma=args.ou_sigma
        )

        # Hyper-parameters
        self.batch_size = args.bsize
        self.tau = args.tau
        self.discount = args.discount
        self.depsilon = 1.0 / args.epsilon

        self.epsilon = 1.0
        self.s_t = None  # Most recent state
        self.a_t = None  # Most recent action
        self.is_training = True
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Exponentially decaying Q value
        self.decay_q = args.decay_q
        if self.decay_q:
            self.q_ema_decay = 2 / (N_DAYS + 1)
            self.decay_q_buffer = RingBuffer(N_DAYS)

        # Temporal Discounting
        self.temporal_gamma = args.temporal_gamma
        if self.temporal_gamma:
            self.discount = lambda x: math.exp(-0.1 / (args.train_iter - args.warmup - x))

    def _calculate_exp_moving_average(self, next_q_values, timestamp):
        self.decay_q_buffer.append(next_q_values)
        q_ema = self.decay_q_buffer[0]
        for q_idx in range(1, len(self.decay_q_buffer)):
            q_ema = q_ema * self.q_ema_decay / timestamp + self.decay_q_buffer[q_idx] * (1 - self.q_ema_decay / timestamp)
        return q_ema

    def update_policy(self, timestamp):
        # Sample batch
        (
            state_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            terminal_batch,
        ) = self.memory.sample_and_split(self.batch_size)

        # Prepare for the target q batch
        with torch.no_grad():
            next_q_values = self.critic_target(
                to_tensor(next_state_batch),
                self.actor_target(to_tensor(next_state_batch)),
            )

        # Calculate EMA for Q
        if self.decay_q:
            next_q_values = self._calculate_exp_moving_average(next_q_values, timestamp)

        target_q_batch = (
            to_tensor(reward_batch)
            + (
                self.discount if not self.temporal_gamma else self.discount(timestamp)
            ) * to_tensor(terminal_batch.astype(np.float)) * next_q_values
        )

        # Critic update
        self.critic.zero_grad()

        q_batch = self.critic(to_tensor(state_batch), to_tensor(action_batch))

        value_loss = criterion(q_batch, target_q_batch)
        value_loss.backward()
        self.critic_optim.step()

        # Actor update
        self.actor.zero_grad()

        policy_loss = -self.critic(to_tensor(state_batch), self.actor(to_tensor(state_batch)))

        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()

        # Target update
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

    def train(self):
        self.actor.train()
        self.actor_target.train()
        self.critic.train()
        self.critic_target.train()

    def cuda(self):
        self.actor.to(self.device)
        self.actor_target.to(self.device)
        self.critic.to(self.device)
        self.critic_target.to(self.device)

    def observe(self, r_t, s_t1, done):
        if self.is_training:
            self.memory.append(self.s_t, self.a_t, r_t, done)
            self.s_t = s_t1

    def random_action(self):
        action = np.random.uniform(-1.0, 1.0, self.nb_actions)
        self.a_t = action
        return action

    def select_action(self, s_t, decay_epsilon=True):
        action = to_numpy(self.actor(to_tensor(np.array([s_t])))).squeeze(0)
        action += self.is_training * max(self.epsilon, 0) * self.random_process.sample()
        action = np.clip(action, -1.0, 1.0)

        if decay_epsilon:
            self.epsilon -= self.depsilon

        self.a_t = action
        return action

    def reset(self, obs):
        self.s_t = obs
        self.random_process.reset_states()

    def load_weights(self, output):
        if output is None:
            return

        self.actor.load_state_dict(torch.load(f'{output}/actor.pkl'))
        self.actor_target.load_state_dict(
            torch.load(f'{output}/actor_target.pkl')
        )
        self.critic.load_state_dict(torch.load(f'{output}/critic.pkl'))
        self.critic_target.load_state_dict(
            torch.load(f'{output}/critic_target.pkl')
        )

    def save_model(self, output):
        torch.save(self.actor.state_dict(), f'{output}/actor.pkl')
        torch.save(self.critic.state_dict(), f'{output}/critic.pkl')
        torch.save(self.actor_target.state_dict(), f'{output}/actor_target.pkl')
        torch.save(
            self.critic_target.state_dict(), f'{output}/critic_target.pkl'
        )

    def seed(self, s):
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(s)
