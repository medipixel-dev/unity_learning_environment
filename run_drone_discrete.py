# -*- coding: utf-8 -*-
"""Train or test algorithms on LunarLanderContinuous-v2.

- Author: Curt Park
- Contact: curt.park@medipixel.io
"""

import argparse
import datetime
import logging

from rl_algorithms import build_agent
import rl_algorithms.common.helper_functions as common_utils
from rl_algorithms.utils import Config

from unity_wrappers import unity_env_generator

logger = logging.getLogger("mlagents.envs")
logger.disabled = True


def parse_args() -> argparse.Namespace:
    """Set argparse."""
    parser = argparse.ArgumentParser(description="Pytorch RL algorithms")
    parser.add_argument(
        "--seed", type=int, default=777, help="random seed for reproducibility"
    )
    parser.add_argument(
        "--cfg-path", type=str, default="./configs/Drone_discrete/iqn.py", help="config path",
    )
    parser.add_argument(
        "--test", dest="test", action="store_true", help="test mode (no training)"
    )
    parser.add_argument(
        "--load-from",
        type=str,
        default=None,
        help="load the saved model and optimizer at the beginning",
    )
    parser.add_argument(
        "--render-after",
        type=int,
        default=0,
        help="start rendering after the input number of episode",
    )
    parser.add_argument(
        "--log", dest="log", action="store_true", help="turn on logging"
    )
    parser.add_argument(
        "--save-period", type=int, default=100, help="save model period"
    )
    parser.add_argument(
        "--episode-num", type=int, default=20000, help="total episode num"
    )
    parser.add_argument(
        "--max-episode-steps", type=int, default=10000, help="max episode step"
    )
    parser.add_argument(
        "--interim-test-num",
        type=int,
        default=10,
        help="number of test during training",
    )
    parser.add_argument(
        "--worker-id", type=int, default=1, help="worker id of unity env"
    )
    parser.set_defaults(render=False)

    return parser.parse_args()


def main():
    """Main."""
    args = parse_args()

    # env initialization
    env_name = "Drone_discrete"
    train_mode = not args.test
    env = unity_env_generator(env_name, train_mode, args.worker_id)

    # set a random seed
    common_utils.set_random_seed(args.seed, env)

    # run
    NOWTIMES = datetime.datetime.now()
    curr_time = NOWTIMES.strftime("%y%m%d_%H%M%S")

    cfg = Config.fromfile(args.cfg_path)
    cfg.agent["log_cfg"] = dict(agent=cfg.agent.type, curr_time=curr_time)
    build_args = dict(args=args, env=env)
    agent = build_agent(cfg.agent, build_args)

    if not args.test:
        agent.train()
    else:
        agent.test()


if __name__ == "__main__":
    main()
