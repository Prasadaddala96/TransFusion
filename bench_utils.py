# bench_utils.py
import argparse
from typing import Any, Callable


_parser = argparse.ArgumentParser(add_help=True)


def DeclareArg(name: str, arg_type: Callable[[str], Any], default: Any, help_text: str = "") -> Any:
    """
    Simple CLI helper used across bench_train scripts.

    Example:
        seed = DeclareArg("seed", int, 42, "random seed")
        lr   = DeclareArg("learning_rate", float, 1e-4, "learning rate")

    It supports:
      --name 123
      --name=123
    """
    _parser.add_argument(f"--{name}", type=arg_type, default=default, help=help_text)

    
    args, _ = _parser.parse_known_args()
    return getattr(args, name)
