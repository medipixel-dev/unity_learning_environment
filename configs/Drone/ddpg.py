"""Config for DDPG on LunarLanderContinuous-v2.

- Author: Kyunghwan Kim
- Contact: kh.kim@medipixel.io
"""
from rl_algorithms.common.helper_functions import identity
import torch.nn.functional as F

agent = dict(
    type="DDPGAgent",
    hyper_params=dict(
        gamma=0.99,
        tau=0.01,
        buffer_size=int(30000),
        batch_size=32,
        initial_random_action=int(200),
        multiple_update=1,  # multiple learning updates
        gradient_clip_ac=0.5,
        gradient_clip_cr=1.0,
    ),
    backbone=dict(actor=dict(), critic=dict(),),
    head=dict(
        actor=dict(
            type="MLP", configs=dict(hidden_sizes=[512], output_activation=F.tanh,),
        ),
        critic=dict(
            type="MLP",
            configs=dict(
                hidden_sizes=[256, 256], output_size=1, output_activation=identity,
            ),
        ),
    ),
    optim_cfg=dict(lr_actor=0.0001, lr_critic=0.001, weight_decay=1e-6),
    noise_cfg=dict(ou_noise_theta=0.15, ou_noise_sigma=0.2),
)
