"""Run only with an explicit --execute-training flag on a prepared machine."""
from .cli import training_cli

if __name__ == "__main__":
    training_cli(public=True)
