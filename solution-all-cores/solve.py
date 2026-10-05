"""P pairs of square matrices, as one GL operation.

The scheme's ciphertext holds the whole batch, so the product of every pair is
one call: there is no loop to write, and no packing to choose — the engine's
tensor is the layout.
"""
import numpy as np

from fherma import Inputs, Outputs, Point, Tensor


def init(p: Point, cc):
    return None


def encoding(cc, inp: Inputs) -> list:
    shape = cc.engine.shape
    return [np.asarray(inp.a.data).reshape(shape), np.asarray(inp.b.data).reshape(shape)]


def run(state, cc, cts: list) -> list:
    return [cc.engine.matrix_multiply(cts[0], cts[1], cc.keys.matrix_multiplication)]


def decoding(p: Point, cc, pts: list) -> Outputs:
    values = np.asarray(pts[0]).real.reshape(-1)
    return Outputs(c=Tensor((p.P, p.N, p.N), values.tolist(), "f64"))


def free(state) -> None:
    pass
