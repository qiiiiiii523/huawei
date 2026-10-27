"""Frozen M7 backbone; experiment heads/losses remain independent choices."""
from .config import ModelConfig
from dataclasses import replace

M7_CHANNELS = (64, 128, 256, 256, 256, 256, 256)
M7_DILATIONS = (1, 2, 4, 8, 16, 32)


def m7_config(output_head="linear"):
    return ModelConfig(channels=M7_CHANNELS, time_dim=128, metadata_dim=128,
                       dilations=M7_DILATIONS, output_head=output_head,
                       head_channels=64, coarse_interval_seconds=0.5)


def model_preset(config):
    """Name the backbone without claiming a custom model is the frozen M7."""
    if config.record_position and replace(config, record_position=False) == m7_config(config.output_head):
        return "M7-Pos"
    return "M7" if config == m7_config(config.output_head) else "custom"
