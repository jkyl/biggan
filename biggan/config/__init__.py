import os
import yaml
import argparse


def _get_config(config_name: str):
    config_dir = os.path.abspath(os.path.dirname(__file__))
    config_file = os.path.join(config_dir, f"{config_name}.yaml")
    with open(config_file, "r") as f:
        return yaml.load(f, Loader=yaml.Loader)


def _get_config_parser(config_name: str, **kwargs):
    config = _get_config(config_name)
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        **kwargs,
    )
    for argument, argument_config in config.items():
        parser.add_argument(
            "--" + argument,
            **argument_config,
            required="default" not in argument_config,
        )
    parser.defaults = argparse.Namespace(**{
        action.dest: parser.get_default(action.dest)
        for action in parser._actions
        if action.option_strings
    })
    return parser


model = _get_config_parser("model", add_help=False)
training = _get_config_parser("training", parents=[model])
# inference = _get_config_parser("inference", parents=[model])
