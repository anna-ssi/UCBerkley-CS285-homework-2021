import numpy as np
import torch

import cs285.infrastructure.pytorch_util as ptu


class ArgMaxPolicy(object):

    def __init__(self, critic):
        self.critic = critic

    def get_action(self, obs):
        if len(obs.shape) > 3:
            observation = obs
        else:
            observation = obs[None]

        observation = ptu.from_numpy(observation)
        
        with torch.no_grad():
            q_values = self.critic.q_net(observation)
            action = torch.argmax(q_values, -1).detach().cpu().numpy()

        return action.squeeze()
