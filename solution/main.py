"""GENERATED for matmul/secret-matrix-batches@1.0.0. Do not edit — `--update` rewrites it.

    ./solution <point directory>

The plaintext wire, with an encrypted middle: cases are read in cleartext,
answers are written in cleartext, and between them the envelope encrypts what
the signature marks secret. Only the call to run() is timed; encoding,
encryption, decryption and decoding are all outside the window.
"""
import json
import resource
import sys
import time
from pathlib import Path

from envelope import Envelope
from fherma import Inputs, Outputs, Point, Tensor, decode, encode
from solve import decoding, encoding, init, run

try:
    from solve import free
except ImportError:
    def free(state): ...


def _read(path: Path) -> bytes:
    return path.read_bytes()


def _write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)


def _read_inputs(p: Point, case: Path) -> Inputs:
    return Inputs(
        a=decode(_read(case / "a.bin"), (p.P, p.N, p.N), "f64"),
        b=decode(_read(case / "b.bin"), (p.P, p.N, p.N), "f64"),
    )


def _write_outputs(case: Path, out: Outputs) -> None:
    _write(case / "c.bin", encode(out.c))


def _peak_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024


def _report(out: Path, envelope_s: float, init_s: float, cases: list) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(
        {"envelope_s": envelope_s, "init_s": init_s, "cases": cases}, indent=2))


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    manifest = json.loads((root / "manifest.json").read_text())
    p = Point(**manifest["point"])

    began = time.perf_counter()
    envelope = Envelope(str(root / "config.jsonc") if (root / "config.jsonc").exists()
                        else "config.jsonc", point=p)
    envelope_s = time.perf_counter() - began

    began = time.perf_counter()
    state = init(p, envelope.cc)
    init_s = time.perf_counter() - began

    results = []
    for index in range(int(manifest["cases"])):
        case = root / "cases" / f"{index:06d}"

        # A case that fails is one row, not the end of the run: the cases
        # after it are still owed their answers.
        try:
            inp = _read_inputs(p, case)

            # Every stage is timed, and only one is the score. The others are
            # reported so the whole cost of an encrypted answer is visible —
            # what the layout costs, what the cryptography costs — without any
            # of it leaking into the measurement.
            mark = time.perf_counter()
            packings = encoding(envelope.cc, inp)
            encoding_s = time.perf_counter() - mark

            mark = time.perf_counter()
            cts = envelope.encrypt(packings)
            encrypt_s = time.perf_counter() - mark

            # ── the measured window: one call, objects already in memory ──
            began = time.perf_counter()
            out_cts = run(state, envelope.cc, cts)
            seconds = time.perf_counter() - began
            # ── window closed ──

            mark = time.perf_counter()
            pts = envelope.decrypt(out_cts)
            decrypt_s = time.perf_counter() - mark

            mark = time.perf_counter()
            answer = decoding(p, envelope.cc, pts)
            decoding_s = time.perf_counter() - mark

            _write_outputs(root / "out" / f"{index:06d}", answer)
        except Exception as failure:
            results.append({
                "i": index,
                "seconds": None,
                "status": "crashed",
                "note": f"{type(failure).__name__}: {failure}"[:200],
            })
            _report(root / "out", envelope_s, init_s, results)
            continue

        results.append({
            "i": index,
            "seconds": seconds,
            "status": "ok",
            "peak_mb": _peak_mb(),
            "encoding_s": encoding_s,
            "encrypt_s": encrypt_s,
            "decrypt_s": decrypt_s,
            "decoding_s": decoding_s,
            "ciphertexts_in": len(cts),
            "ciphertexts_out": len(out_cts),
            "result_level": envelope.level_of(out_cts[0]) if out_cts else None,
        })
        _report(root / "out", envelope_s, init_s, results)

    free(state)


if __name__ == "__main__":
    main()
