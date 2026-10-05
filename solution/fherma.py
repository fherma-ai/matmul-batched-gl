"""GENERATED from matmul/secret-matrix-batches@1.0.0. Do not edit.

Types derived from the signature. Write the bodies in generate.py and oracle.py;
serialising is the runner's job, so nothing here opens a file.
"""
from dataclasses import dataclass, field
import hashlib
import os
import struct

SPEC = "matmul/secret-matrix-batches@1.0.0"

#: Bits on the wire, by element type.
WIDTHS = {"i8": 8, "i16": 16, "i32": 32, "i64": 64, "u8": 8, "u16": 16, "u32": 32, "u64": 64, "f32": 32, "f64": 64}


@dataclass(frozen=True)
class Point:
    """One field per resolved parameter: the free ones the point sets, then the
    derived ones its getters fill. `resolve` in main.py builds it."""
    P: int           # free · u32
    N: int           # free · u32


@dataclass
class Inputs:
    a: "Tensor"     # %a: tensor<P x N x N x f64>   (secret in the signature; cleartext here)
    b: "Tensor"     # %b: tensor<P x N x N x f64>   (secret in the signature; cleartext here)


@dataclass
class Outputs:
    c: "Tensor"     # %c: tensor<P x N x N x f64>   (secret in the signature; cleartext here)


@dataclass
class Verdict:
    passed: bool
    metrics: dict = field(default_factory=dict)


class Tensor:
    """A flat, row-major buffer. Shape is known at run time; the element type
    comes from the signature and fixes the width on the wire."""

    __slots__ = ("shape", "elem", "data")

    def __init__(self, shape, data, elem="i64"):
        self.shape, self.elem, self.data = tuple(shape), elem, list(data)
        if len(self.data) != self.count():
            raise ValueError(f"{len(self.data)} values for shape {self.shape}")

    def count(self):
        total = 1
        for dimension in self.shape:
            total *= dimension
        return total

    def nbytes(self):
        return self.count() * WIDTHS[self.elem] // 8

    def __repr__(self):
        return f"Tensor(shape={self.shape}, elem={self.elem!r}, …)"


class Stream:
    """The only source of randomness on offer.

    block(i) = SHA256(SPEC | seed | i); the stream is block(0) ‖ block(1) ‖ …
    It is the same in every language because it is written down, and it is
    seekable, so generating large inputs parallelises. Anything else — random,
    time, urandom, secrets — makes a generator irreproducible and is rejected.
    """

    def __init__(self, seed: int):
        self._prefix = f"{SPEC}|{seed}"
        self._buffer, self._counter = b"", 0

    def _fill(self, n: int) -> None:
        while len(self._buffer) < n:
            block = f"{self._prefix}|{self._counter}".encode()
            self._buffer += hashlib.sha256(block).digest()
            self._counter += 1

    def bytes(self, n: int) -> bytes:
        self._fill(n)
        head, self._buffer = self._buffer[:n], self._buffer[n:]
        return head

    def bits(self, k: int) -> int:
        return int.from_bytes(self.bytes((k + 7) // 8), "little") & ((1 << k) - 1)

    def below(self, n: int) -> int:
        """Uniform on [0, n). The extra 64 bits keep the bias under 2**-64."""
        return self.bits(n.bit_length() + 64) % n


def encode(tensor: Tensor) -> bytes:
    """A tensor as it goes on the wire: little-endian, row-major, no header.

    The shape travels beside it in case.json rather than inside the bytes, so a
    reader in another language needs no parser — only the width, which the
    signature already fixed.
    """
    if tensor.elem[0] == "f":
        kind = {32: "f", 64: "d"}.get(WIDTHS[tensor.elem])
        if kind is None:
            raise ValueError(f"{tensor.elem} has no wire form yet")
        return struct.pack(f"<{tensor.count()}{kind}", *tensor.data)

    width = WIDTHS[tensor.elem] // 8
    signed = tensor.elem[0] == "i"
    return b"".join(
        int(value).to_bytes(width, "little", signed=signed) for value in tensor.data
    )


def decode(raw: bytes, shape, elem="i64") -> Tensor:
    """The inverse of encode."""
    count = 1
    for dimension in shape:
        count *= dimension

    width = WIDTHS[elem] // 8
    if len(raw) != count * width:
        raise ValueError(f"{len(raw)} bytes for {tuple(shape)} of {elem}")

    if elem[0] == "f":
        kind = {32: "f", 64: "d"}.get(WIDTHS[elem])
        if kind is None:
            raise ValueError(f"{elem} has no wire form yet")
        values = list(struct.unpack(f"<{count}{kind}", raw))
    else:
        signed = elem[0] == "i"
        values = [
            int.from_bytes(raw[i * width : (i + 1) * width], "little", signed=signed)
            for i in range(count)
        ]
    return Tensor(shape, values, elem)


def pack(values, limbs, elem="i64") -> Tensor:
    """Integers wider than a word into `limbs` words each, low word first.

    A row per value, so a list of coefficients becomes a tensor<len x limbs x
    elem>. The word is as wide as the element the signature named — 32 bits for
    u32, which is what a limb array on the GPU expects — not always 64.
    """
    width = WIDTHS[elem]
    mask = (1 << width) - 1
    data = []
    for value in values:
        for index in range(limbs):
            data.append((value >> (width * index)) & mask)
    return Tensor((len(values), limbs), data, elem)


def pack_one(value, limbs, elem="i64") -> Tensor:
    """One wide integer as a flat tensor<limbs x elem> — a single value, not a row.

    This is what a derived modulus wants: q is one number, held as `limbs` words,
    with no leading dimension. `pack([q], limbs)` would give a tensor<1 x limbs>,
    which is the same bytes shaped wrong.
    """
    row = pack([value], limbs, elem)
    return Tensor((limbs,), row.data, elem)


def unpack(tensor: Tensor) -> list:
    """The inverse of pack, for a tensor whose last dimension is the limbs."""
    *head, limbs = tensor.shape
    width = WIDTHS[tensor.elem]
    count = 1
    for dimension in head:
        count *= dimension
    return [
        sum(tensor.data[i * limbs + j] << (width * j) for j in range(limbs))
        for i in range(count)
    ]


def unpack_one(tensor: Tensor) -> int:
    """The inverse of pack_one: a flat limb tensor back to one integer."""
    width = WIDTHS[tensor.elem]
    return sum(word << (width * j) for j, word in enumerate(tensor.data))


# ── derivation rules ──────────────────────────────────────────────────────────
# The standard library a getter reaches for. Deterministic by construction: the
# same arguments give the same value in the generator, the oracle and every
# runner, which is what stops the three disagreeing about the ring.
def _probably_prime(n: int) -> bool:
    """Miller–Rabin with a fixed base set, after trial division by small primes.

    The bases are fixed rather than random so the result is the same everywhere.
    Forty of them put the chance of a composite slipping through below any figure
    that matters here, and trial division rejects almost everything before them.
    """
    if n < 2:
        return False
    small = (
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67,
        71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149,
        151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199,
    )
    for p in small:
        if n == p:
            return True
        if n % p == 0:
            return False

    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in small[:40]:
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def ntt_prime(n: int, bits: int) -> int:
    """The largest prime below 2**bits that is 1 mod 2n.

    An NTT-friendly modulus: q ≡ 1 (mod 2n) is exactly the condition for a
    primitive 2n-th root of unity to exist mod q, so a negacyclic transform of
    length n works over it. Searched downward from just under 2**bits, over the
    residues that already satisfy the congruence, so only primality is left to test.
    """
    step = 2 * n
    # The largest q < 2**bits with q ≡ 1 (mod 2n): start there and walk down.
    q = ((1 << bits) - 1 - 1) // step * step + 1
    while q > 1:
        if _probably_prime(q):
            return q
        q -= step
    raise ValueError(f"no prime below 2**{bits} is 1 mod {step}")


def asset(name: str) -> str:
    """Path to a file in assets/, wherever this bundle happens to be mounted."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", name)
