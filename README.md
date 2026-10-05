# Batched matrix multiplication over GL — DESILO FHE

> Answers [`matrix-multiplication` / `batched@1.0.0`](https://www.fherma.io/kernels/matrix-multiplication/specifications/batched)
> on the FHERMA kernel catalogue.
>
> Uses the [DESILO FHE library](https://fhe.desilo.dev/), free for
> non-commercial use. The library is **not** vendored here: the build installs
> it from PyPI, so whoever runs this accepts its licence themselves.

P pairs of square matrices, multiplied pairwise — `c[i] = a[i] · b[i]` — in
one homomorphic operation.

```text
kernel matmul<P: u32, N: u32>(
    %a: secret<tensor<P x N x N x f64>>,
    %b: secret<tensor<P x N x N x f64>>,
) -> %c: secret<tensor<P x N x N x f64>>
```

## What the answer is

Three lines, and no loop:

```python
def run(state, cc, cts: list) -> list:
    return [cc.engine.matrix_multiply(cts[0], cts[1], cc.keys.matrix_multiplication)]
```

GL is a scheme whose ciphertext holds a batch of matrices and whose matrix
product is one operation rather than a circuit of rotations and masks. There
is nothing to pack and nothing to fold: the engine's tensor *is* the layout,
so `encoding` reshapes the case into it and `decoding` reshapes it back.

That is also why this answers the batched specification and not the
single-pair one beside it. The library offers three engine shapes —
`(256, 16, 16)`, `(256, 32, 32)`, `(256, 64, 64)` — and the batch of 256 is
the scheme's, not a setting. "Multiply one pair" is not a question this scheme
can be asked.

## Layout

```
solution/
  solve.py       the answer: four functions, this is the whole of it
  config.jsonc   engine, threads, which keys to generate
  fherma.toml    how it is built and started
  envelope.py    generated — context, keys, encryption. Holds the secret key
  main.py        generated — the measured loop
  fherma.py      generated — the types, from the signature
```

Only `solve.py` and `config.jsonc` are written by hand. The rest is emitted by
`fherma-lang` from the specification's signature and replaced at every
measurement, so an answer cannot drift from the contract it claims to meet.

## The library, and why it is not here

```toml
[build]
command = "pip install --no-cache-dir --target . desilofhe==1.17.0"
```

`--target .` puts the package beside `main.py`, the directory Python already
searches. On a measurement runner the build is the one stage with a network;
the measured container has none and needs none, because the package is already
there.

On an NVIDIA card the package is `desilofhe-cu130` and the image is
`fherma/desilo:1.17.0-cu130`; nothing else changes, and `"mode": "gpu"` in
`config.jsonc` is what moves the work to the device.

## Configuration

| Key | Here | Why |
|---|---|---|
| `scheme` | `gl` | the matrix scheme, not RNS-CKKS |
| `mode` | `cpu` | `gpu` with the CUDA image |
| `thread_count` | `1` | so a number means one core; `0` takes the machine |
| `keys` | `["matrix_multiplication"]` | the only key this answer uses; every other costs generation time and memory |
| `shape` | `null` | taken from the point, so the matrix size is stated in one place |

## Running it yourself

```sh
pip install --no-cache-dir --target solution desilofhe==1.17.0 numpy
python solution/main.py <point directory>
```

A point directory is what the specification's testing bundle writes with
`main.py make`. The answer is judged by the same bundle with
`main.py verify` — element-wise, to an absolute tolerance of 1e-2 against the
cleartext product.

## Citation

The DESILO FHE licence requires acknowledgement in any public project or
presentation that uses the library. If you publish numbers measured with this
solution, cite the library as DESILO specifies.
