import numpy as np

from .base_agent import BaseAgent
from cs285.policies.MLP_policy import MLPPolicyPG
from cs285.infrastructure.replay_buffer import ReplayBuffer
from cs285.infrastructure.utils import normalize


class PGAgent(BaseAgent):
    def __init__(self, env, agent_params):
        super(PGAgent, self).__init__()

        # init vars
        self.env = env
        self.agent_params = agent_params
        self.gamma = self.agent_params['gamma']
        self.standardize_advantages = self.agent_params['standardize_advantages']
        self.nn_baseline = self.agent_params['nn_baseline']
        self.reward_to_go = self.agent_params['reward_to_go']
        self.gae_lambda = self.agent_params['gae_lambda']

        # actor/policy
        self.actor = MLPPolicyPG(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            discrete=self.agent_params['discrete'],
            learning_rate=self.agent_params['learning_rate'],
            nn_baseline=self.agent_params['nn_baseline']
        )

        # replay buffer
        self.replay_buffer = ReplayBuffer(1000000)

    def train(self, observations, actions, rewards_list, next_observations, terminals):
        """
            Training a PG agent refers to updating its actor using the given observations/actions
            and the calculated qvals/advantages that come from the seen rewards.

            Update the policy using the given batch of data, and return the train_log.
        """

        q_values = self.calculate_q_vals(rewards_list)
        advantages = self.estimate_advantage(
            observations, rewards_list, q_values, terminals)

        train_log = self.actor.update(
            observations, actions, advantages, q_values)

        return train_log

    def calculate_q_vals(self, rewards_list):
        """
            Monte Carlo estimation of the Q function.

            Return the estimated qvals based on the given rewards, using
            either the full trajectory-based estimator or the reward-to-go estimator.
        """
        q_values = []

        for traj_rew in rewards_list:
            q_val = self._discounted_cumsum(
                traj_rew) if self.reward_to_go else self._discounted_return(traj_rew)
            q_values.append(q_val)

        return np.concatenate(q_values)

    def estimate_advantage(self, obs, rews_list, q_values, terminals):
        """
            Computes advantages by (possibly) using GAE, or subtracting a baseline from the estimated Q values
        """

        # Estimate the advantage when nn_baseline is True,
        # by querying the neural network that you're using to learn the value function
        if self.nn_baseline:
            values_unnormalized = self.actor.run_baseline_prediction(obs)
            # ensure that the value predictions and q_values have the same dimensionality
            # to prevent silent broadcasting errors
            assert values_unnormalized.ndim == q_values.ndim

            # TODO: values were trained with standardized q_values
            q_mean, q_std = np.mean(q_values), np.std(q_values)
            values = normalize(values_unnormalized, q_mean, q_std)

            if self.gae_lambda is not None:
                # append a dummy T+1 value for simpler recursive calculation
                values = np.append(values, [0])

                # combine rews_list into a single array
                rews = np.concatenate(rews_list)

                # create empty numpy array to populate with GAE advantage
                # estimates, with dummy T+1 value for simpler recursive calculation
                batch_size = obs.shape[0]
                advantages = np.zeros(batch_size + 1)

                for i in reversed(range(batch_size)):
                    # recursively compute advantage estimates starting from timestep T.
                    if terminals[i] == 1:
                        advantages[i] = rews[i] - values[i]
                    else:
                        delta = rews[i] + \
                            self.gae_lambda * values[i+1] - values[i]
                        advantages[i] = delta + \
                            self.gae_lambda * self.gamma * advantages[i+1]

                # remove dummy advantage
                advantages = advantages[:-1]

            else:
                advantages = q_values - values

        # Else, just set the advantage to [Q]
        else:
            advantages = q_values.copy()

        # Normalize the resulting advantages
        if self.standardize_advantages:
            advantages = normalize(advantages, np.mean(
                advantages), np.std(advantages))

        return advantages

    #####################################################
    #####################################################

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_recent_data(batch_size, concat_rew=False)

    #####################################################
    ################## HELPER FUNCTIONS #################
    #####################################################

    def _discounted_return(self, rewards):
        """
            Input: list of rewards {r_0, r_1, ..., r_t', ... r_T} from a single rollout of length T
            Output: list where each index t contains sum_{t'=0}^T gamma^t' r_{t'}
        """

        dis_vec = np.array([self.gamma ** i for i in range(len(rewards))])
        cum_rew = np.array(rewards) @ dis_vec

        return np.repeat(cum_rew, len(rewards))

    def _discounted_cumsum(self, rewards):
        """
            Input: list of rewards {r_0, r_1, ..., r_t', ... r_T},
            Output: list where the entry in each index t' is sum_{t'=t}^T gamma^(t'-t) * r_{t'}
        """

        upp_tri = np.tril(np.tile(rewards[::-1], (len(rewards), 1)))
        dis_vec = np.array([self.gamma ** i for i in range(len(rewards))])

        return np.flip(upp_tri @ dis_vec)
