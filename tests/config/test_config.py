import os

from biggan import config


def test_get_base_config():
    assert len(vars(config.base)) > 0

def test_get_child_config():
    config_dir = os.path.dirname(config.__file__)
    child_config_file = os.path.join(config_dir, "_test_child.yaml")
    try:
        with open(child_config_file, "w") as f_obj:
            f_obj.write("foo:\n  default: bar")
        child_config = config.get_config_parser("_test_child", parent=config.base)
        assert child_config.defaults.foo == "bar"
        defaults = vars(child_config.defaults)
        defaults.pop("foo")
        assert vars(config.base.defaults) == defaults
    finally:
        os.remove(child_config_file)
