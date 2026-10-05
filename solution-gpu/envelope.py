"""GENERATED for matmul/secret-matrix-batches@1.0.0. Do not edit — `--update` rewrites it.

The cryptographic envelope: engine, keys, encryption, decryption. This is the
only place a secret key exists. The measured call never enters this file.

DESILO keeps evaluation keys as objects rather than as state inside a context,
so what an answer is given is a small facade: `cc.engine` to compute with and
`cc.keys` for the public evaluation material. Holding those lets an answer
compute and does not let it decrypt, which is the same line OpenFHE draws by
keeping its evaluation keys inside the context.

Every parameter comes from config.jsonc; the scheme named there decides which
of them are read, so the config can switch schemes without re-scaffolding.
"""
import json
from pathlib import Path

import numpy as np
from desilofhe import Engine, GLEngine

# Keys an answer may ask for, and the engine method that makes each. Named
# rather than generated wholesale: every key is generation time and resident
# memory, and a point that needs none should pay for none.
CKKS_KEYS = {
    "relinearization": "create_relinearization_key",
    "rotation": "create_rotation_key",
    "conjugation": "create_conjugation_key",
    "bootstrap": "create_bootstrap_key",
    "small_bootstrap": "create_small_bootstrap_key",
}

GL_KEYS = {
    "matrix_multiplication": "create_matrix_multiplication_key",
    "rotation": "create_rotation_key",
    "transposition": "create_transposition_key",
    "conjugate_transposition": "create_conjugate_transposition_key",
    "hadamard_multiplication": "create_hadamard_multiplication_key",
    "conjugation": "create_conjugation_key",
}

# What the GL scheme can be shaped as. Checked here so a point outside it is
# refused before any key is generated, with the three answers in the message.
GL_SHAPES = ((256, 16, 16), (256, 32, 32), (256, 64, 64))


def load_config(path):
    """The config is JSONC: `//` to end of line is a comment."""
    text = Path(path).read_text()
    bare = "\n".join(line.split("//")[0] for line in text.splitlines())
    return json.loads(bare)


class Keys:
    """The public evaluation material, by the names the config asked for."""

    def __init__(self, made):
        self.__dict__.update(made)

    def __repr__(self):
        return "Keys(" + ", ".join(sorted(self.__dict__)) + ")"


class Context:
    """What an answer computes with: the engine, and the public keys."""

    def __init__(self, engine, keys, public_key=None, point=None):
        self.engine = engine
        self.keys = keys
        self.public_key = public_key
        self.point = point

    def __repr__(self):
        return f"Context({type(self.engine).__name__}, {self.keys!r})"


class Envelope:
    """Built once per point; keys are cached for every case at it."""

    def __init__(self, config_path="config.jsonc", point=None):
        cfg = load_config(config_path)
        self.cfg = cfg
        self.point = point
        scheme = cfg["scheme"]
        mode = cfg.get("mode") or "cpu"
        threads = cfg.get("thread_count")
        device = int(cfg.get("device_id") or 0)
        self.__level = cfg.get("encrypt_level")

        if scheme == "gl":
            shape = self.__gl_shape(cfg, point)
            engine = GLEngine(shape, mode, thread_count=int(threads or 0),
                              device_id=device)
            makers = GL_KEYS
        elif scheme == "ckks":
            engine = self.__ckks_engine(cfg, mode, threads, device)
            makers = CKKS_KEYS
        else:
            raise ValueError(f"config.jsonc: unknown scheme {scheme!r}")

        secret = engine.create_secret_key()
        wanted = list(cfg.get("keys") or [])
        unknown = [name for name in wanted if name not in makers]
        if unknown:
            raise ValueError(
                f"config.jsonc: {scheme} has no key named {unknown[0]!r}. "
                f"Keys: {', '.join(sorted(makers))}."
            )
        made = {name: getattr(engine, makers[name])(secret) for name in wanted}
        if scheme == "ckks":
            for delta in cfg.get("fixed_rotation_deltas") or []:
                made[f"fixed_rotation_{delta}"] = \
                    engine.create_fixed_rotation_key(secret, int(delta))

        public = engine.create_public_key(secret) if scheme == "ckks" else None
        self.scheme = scheme
        self.engine = engine
        self.cc = Context(engine, Keys(made), public_key=public, point=point)
        self.__secret = secret          # never handed out

    # ── sizing ──────────────────────────────────────────────────────────────

    @staticmethod
    def __gl_shape(cfg, point):
        """The engine's tensor: the config's if it names one, else the point's.

        A specification says the matrix size in its point; repeating it in the
        config would be a second place for it to be wrong.
        """
        shape = cfg.get("shape")
        if not shape:
            if point is None:
                raise ValueError(
                    'config.jsonc: "shape" is null and there is no point to '
                    "take it from — name the shape, or measure under a "
                    "specification whose point carries the matrix size."
                )
            rows = next((int(getattr(point, name)) for name in ("N", "n", "rows")
                         if getattr(point, name, None) is not None), None)
            if rows is None:
                raise ValueError(
                    "config.jsonc: the point names no matrix size (N), so the "
                    'GL engine cannot be shaped — set "shape" instead.'
                )
            batch = next((int(getattr(point, name)) for name in ("B", "b", "batch")
                          if getattr(point, name, None) is not None), 256)
            shape = (batch, rows, rows)
        shape = tuple(int(one) for one in shape)
        if shape not in GL_SHAPES:
            offered = ", ".join(str(one) for one in GL_SHAPES)
            raise ValueError(
                f"GL has no engine of shape {shape}. The library offers "
                f"{offered} — its batch of 256 is the scheme's, not a setting."
            )
        return shape

    @staticmethod
    def __ckks_engine(cfg, mode, threads, device):
        """One of the library's three ways to size an engine, by what is set."""
        common = {"device_id": device}
        if threads:
            common["thread_count"] = int(threads)
        if cfg.get("log_coeff_count") is not None:
            return Engine(
                int(cfg["log_coeff_count"]),
                int(cfg["special_prime_count"]),
                mode,
                log_slot_count=(None if cfg.get("log_slot_count") is None
                                else int(cfg["log_slot_count"])),
                **common,
            )
        if cfg.get("max_level") is not None:
            if cfg.get("slot_count") is not None:
                return Engine(slot_count=int(cfg["slot_count"]),
                              max_level=int(cfg["max_level"]), mode=mode, **common)
            return Engine(int(cfg["max_level"]), mode, **common)
        return Engine(
            mode,
            slot_count=(None if cfg.get("slot_count") is None
                        else int(cfg["slot_count"])),
            use_bootstrap=bool(cfg.get("use_bootstrap")),
            **common,
        )

    # ── the wire ────────────────────────────────────────────────────────────

    @property
    def public_key(self):
        """The CKKS public key. GL encrypts under the secret key, so there is
        none to hand out — and nothing an answer could do with one."""
        return self.cc.public_key

    def encrypt(self, packings: list) -> list:
        """Author packings in, ciphertexts out. The author never encrypts.

        A packing is whatever the library takes: an array of slot values under
        ckks, an array shaped like the engine's tensor under gl.
        """
        key = self.cc.public_key if self.scheme == "ckks" else self.__secret
        if self.__level is None:
            return [self.engine.encrypt(np.asarray(p), key) for p in packings]
        return [self.engine.encrypt(np.asarray(p), key, int(self.__level))
                for p in packings]

    def decrypt(self, cts: list) -> list:
        """Result ciphertexts in, decrypted arrays out."""
        return [self.engine.decrypt(ct, self.__secret) for ct in cts]

    def level_of(self, ct):
        """How much depth the result consumed — a fresh ciphertext has spent
        none, which is worth noticing about an answer that claims to have
        multiplied things."""
        try:
            return ct.level
        except Exception:
            return None
