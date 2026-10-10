"""Huawei-only training or public-checkpoint fine-tuning, never import-time training."""
from .cli import training_cli

if __name__ == "__main__":
    training_cli(public=False)
