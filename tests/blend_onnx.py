"""Build a tiny synthetic ONNX model standing in for the RIFE graph.

Output = (x[:, :3] + x[:, 3:]) / 2 — a plain blend, used by tests to exercise
the whole VapourSynth filter plumbing (FrameEval/ModifyFrame/backends) without
needing torch or the Practical-RIFE releases.
"""

from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def make_blend_onnx(path: str, name: str = "blend") -> str:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 6, "height", "width"])
    timestep = helper.make_tensor_value_info("timestep", TensorProto.FLOAT, [1, 1, 1, 1])
    y = helper.make_tensor_value_info("frame", TensorProto.FLOAT, [1, 3, "height", "width"])

    split = helper.make_node("Split", ["x"], ["a", "b"], axis=1, name="split_pair")
    add = helper.make_node("Add", ["a", "b"], ["ab"], name="add_pair")
    half = numpy_helper.from_array(np.float32(0.5), "half")
    scale = helper.make_node("Mul", ["ab", "half"], ["blend"], name="halve")
    zero = numpy_helper.from_array(np.float32(0.0), "zero")
    timestep_used = helper.make_node("Mul", ["timestep", "zero"], ["zero_t"], name="time_zero")
    merge = helper.make_node("Add", ["blend", "zero_t"], ["frame"], name="merge_time")

    graph = helper.make_graph(
        [split, add, scale, timestep_used, merge],
        name,
        [x, timestep],
        [y],
        [half, zero],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.save(model, path)
    return path
