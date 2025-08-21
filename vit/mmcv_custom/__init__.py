# Copyright (c) Shanghai AI Lab. All rights reserved.
from .checkpoint import load_checkpoint
from .my_checkpoint import my_load_checkpoint

__all__ = ["load_checkpoint", "my_load_checkpoint"]
