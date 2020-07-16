import os
import yaml
import argparse


def _get_config(config_name: str):
    config_dir = os.path.abspath(os.path.dirname(__file__))
    config_file = os.path.join(config_dir, f"{config_name}.yaml")
    with open(config_file, "r") as f:
        return yaml.load(f, Loader=yaml.Loader)


def get_config_parser(config_name: str, parent: argparse.ArgumentParser = None):
    config = _get_config(config_name)
    parents = [parent] if parent else []
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=parents,
        add_help=(parent is None),
        conflict_handler="resolve",
    )
    for argument, argument_config in config.items():
        if parent is not None:
            matches = [a for a in parent._actions if argument == a.option_strings[0][2:]]
            if len(matches) == 1:
                match = matches[0]
                for attr in ("help", "choices", "action"):
                    if attr not in argument_config and hasattr(match, attr):
                        argument_config[attr] = getattr(match, attr)
        parser.add_argument(
            "--" + argument,
            required="default" not in argument_config,
            **argument_config,
        )
    parser.defaults = argparse.Namespace(**{
        action.dest: parser.get_default(action.dest)
        for action in parser._actions
        if action.option_strings
    })
    parser.choices = argparse.Namespace(**{
        action.dest: action.choices
        for action in parser._actions
        if action.choices is not None
    })
    for action in parser._actions:
        if action.type is None and action.dest in config:
            if action.default is not None:
                action.type = type(action.default)
            elif action.choices is not None:
                action.type = type(next(iter(action.choices)))
    return parser


base = get_config_parser("base")
