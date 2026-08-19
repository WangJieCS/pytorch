import torch

from ..lowering import add_layout_constraint, require_contiguous


add_layout_constraint(torch.ops.aten._adaptive_avg_pool2d.default, require_contiguous)
