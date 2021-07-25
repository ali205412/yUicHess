
import argparse

from logging import getLogger,disable

from .lib.logger import setup_logger
from .config import Config

logger = getLogger(__name__)

CMD_LIST = ['self', 'opt', 'eval', 'uci']


def create_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument("--cmd", help="What can be done?", choices=CMD_LIST)
    parser.add_argument("--new", help="Start training from a fresh plate", action="store_true")
    parser.add_argument("--type", help="Use standard config", default="normal")
    parser.add_argument("--total-step", help="change TrainerConfig.start_total_steps value", type=int)
    return parser


def setup(config: Config, args):

    config.opts.new = args.new
    if args.total_step is not None:
        config.trainer.start_total_steps = args.total_step
    config.resource.create_directories()
    setup_logger(config.resource.main_log_path)


def start():

    parser = create_parser()
    args = parser.parse_args()
    config_type = args.type

    if args.cmd == 'uci':
        disable(999999)

    config = Config(config_type=config_type)
    setup(config, args)

    logger.info(f"config type: {config_type}")

    if args.cmd == 'self':
        from .worker import self_play
        return self_play.start(config)
    elif args.cmd == 'opt':
        from .worker import optimize
        return optimize.start(config)
    elif args.cmd == 'eval':
        from .worker import evaluate
        return evaluate.start(config)
    elif args.cmd == 'uci':
        from .play_game import uci
        return uci.start(config)
