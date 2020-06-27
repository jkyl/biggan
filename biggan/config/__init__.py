import os
import yaml
import argparse


def _get_config(config_name: str):
    config_dir = os.path.abspath(os.path.dirname(__file__))
    config_file = os.path.join(config_dir, f"{config_name}.yaml")
    with open(config_file, "r") as f:
        return yaml.load(f, Loader=yaml.Loader)


def _get_config_parser(config_name: str, parent: argparse.ArgumentParser = None):
    config = _get_config(config_name)
    parents = [parent] if parent else []
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=parents,
        add_help=(parent is None),
        conflict_handler="resolve",
    )
    for argument, argument_config in config.items():
        if "help" not in argument_config and parent is not None:
            matches = [a for a in parent._actions if argument == a.option_strings[0][2:]]
            if matches:
                argument_config["help"] = matches[0].help

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


def __getattr__(name):
    if name == "default":
        return _get_config_parser("default")
    return _get_config_parser(name, parent=__getattr__("default"))
