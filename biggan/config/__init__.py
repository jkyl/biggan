import os
import yaml
import argparse


def _get_config(config_name: str):
    config_dir = os.path.abspath(os.path.dirname(__file__))
    config_file = os.path.join(config_dir, f"{config_name}.yaml")
    with open(config_file, "r") as f:
        return yaml.load(f, Loader=yaml.Loader)


def _get_config_parser(config_name: str):
    config = _get_config(config_name)
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    for argument, argument_config in config.items():
        parser.add_argument(
            "--" + argument,
            **argument_config,
            required="default" not in argument_config,
        )
    return parser


train = _get_config_parser("train")
