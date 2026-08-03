#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import math
import struct
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "dsv4-amx-full-model-gate.py"
SPEC = importlib.util.spec_from_file_location("dsv4_amx_full_model_gate", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
GATE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = GATE
SPEC.loader.exec_module(GATE)

COMMIT = "a" * 40
N_VOCAB = 8
NOMINAL_THERMAL = "\n".join(f"Note: {message}" for message in GATE.BASE.THERMAL_NOMINAL_MESSAGES)
BEFORE = {"swap_used_bytes": 0, "thermal_state": NOMINAL_THERMAL}
AFTER = {"swap_used_bytes": 0, "thermal_state": NOMINAL_THERMAL}


def audit(event: str, outcome: str, detail: str = "") -> str:
    return f"dsv4_amx_audit event={event} outcome={outcome} {detail}".rstrip()


def driver(event: str, outcome: str, mode: str, detail: str = "") -> str:
    return f"dsv4_amx_full_gate_driver event={event} outcome={outcome} mode={mode} {detail}".rstrip()


def fixture_tokens() -> tuple[tuple[int, ...], tuple[int, ...]]:
    prompt = tuple(index % N_VOCAB for index in range(GATE.PROMPT_TOKENS))
    teacher = tuple((index + 2) % N_VOCAB for index in range(GATE.CONTINUATION_TOKENS))
    return prompt, teacher


def write_capture(path: Path, mode: int, *, delta: float = 1.0e-5,
                  nonfinite: tuple[str, int, int] | None = None,
                  diverge: tuple[str, int] | None = None) -> None:
    prompt, teacher = fixture_tokens()
    with path.open("wb") as output:
        output.write(GATE.HEADER.pack(
            GATE.FILE_MAGIC,
            GATE.FORMAT_VERSION,
            GATE.ENDIAN_MARKER,
            mode,
            N_VOCAB,
            GATE.PROMPT_TOKENS,
            GATE.CONTINUATION_TOKENS,
            len(GATE.PATHS),
            GATE.VECTORS_PER_PATH,
            GATE.PROMPT_VERSION,
            COMMIT.encode("ascii"),
        ))
        output.write(struct.pack(f"<{len(prompt)}i", *prompt))
        output.write(struct.pack(f"<{len(teacher)}i", *teacher))
        for path_name in GATE.PATHS:
            output.write(GATE.PATH_MARKERS[path_name])
            prior_top1 = GATE.NO_INPUT_TOKEN
            for step in range(GATE.VECTORS_PER_PATH):
                logits = [-2.0 + 0.25 * token + 0.001 * step for token in range(N_VOCAB)]
                if mode == 1:
                    logits = [value + (delta if token % 2 else -delta) for token, value in enumerate(logits)]
                if nonfinite == (path_name, step, 0):
                    logits[0] = math.nan
                if diverge == (path_name, step):
                    logits[-2] = logits[-1] + 0.01
                top1 = max(range(N_VOCAB), key=logits.__getitem__)
                if step == 0:
                    input_token = GATE.NO_INPUT_TOKEN
                elif path_name == "teacher":
                    input_token = teacher[step - 1]
                else:
                    input_token = prior_top1
                output.write(GATE.RECORD_HEADER.pack(step, input_token, top1))
                output.write(struct.pack(f"<{N_VOCAB}f", *logits))
                prior_top1 = top1
        output.write(GATE.FOOTER_MARKER)


def binding(executable: int) -> str:
    return audit(
        "bindings", "complete",
        f"executable={executable} layers=43 callback_events=86 tensor_metadata=43 unique_buffers=1 "
        "unique_data=1 producerless=1 callback_edges=norm_completion visible_before_consumer=1 "
        "routed_join_exact=1 routed_joins=43 telemetry_consumers=0 direct_consumer_edges=43 "
        "external_output_bytes=33554432 gallocr_output_bytes=0",
    )


def driver_lines(mode: str, capture_bytes: int) -> list[str]:
    prompt, teacher = fixture_tokens()
    return [
        driver("start", "begin", mode,
               f"source_commit={COMMIT} build_commit={COMMIT[:9]} n_ctx=2304 logical_tokens=2112 "
               "context_alignment=256 context_padding=192 n_batch=2048 n_ubatch=2048 "
               "n_seq_max=1 n_rs_seq=0 flash_attn=1 n_gpu_layers=999 split_mode=layer main_gpu=0 load_mtp=0 "
               "n_outputs_max=1 no_perf=1 prompt_tokens=2048 steps=64 paths=2 validation=0 timing=0 benchmark=0"),
        driver("model_load", "pass", mode, "n_vocab=8"),
        driver("prompt", "pass", mode,
               f"prompt_version=1 tokens=2048 teacher_tokens=64 seed_tokens=20 filler_tokens=25 "
               f"prompt_hash=0x{GATE.token_hash(prompt):016x} teacher_hash=0x{GATE.token_hash(teacher):016x} padded=1"),
        driver("path", "begin", mode, "path=teacher prompt_tokens=2048 steps=64"),
        driver("context_contract", "pass", mode,
               "path=teacher n_ctx=2304 n_batch=2048 n_ubatch=2048 n_seq_max=1 "
               "logical_tokens=2112 context_padding=192"),
        driver("path", "pass", mode, "path=teacher prompt_tokens=2048 steps=64 vectors=65"),
        driver("context_teardown", "pass", mode, "path=teacher"),
        driver("path", "begin", mode, "path=greedy prompt_tokens=2048 steps=64"),
        driver("context_contract", "pass", mode,
               "path=greedy n_ctx=2304 n_batch=2048 n_ubatch=2048 n_seq_max=1 "
               "logical_tokens=2112 context_padding=192"),
        driver("path", "pass", mode, "path=greedy prompt_tokens=2048 steps=64 vectors=65"),
        driver("context_teardown", "pass", mode, "path=greedy"),
        driver("output", "pass", mode, f"bytes={capture_bytes} paths=2 vectors=130"),
        driver("model_teardown", "pass", mode),
        driver("backend_teardown", "pass", mode),
        driver("complete", "pass", mode,
               "prompt_tokens=2048 steps=64 paths=2 vectors=130 validation=0 benchmark=0"),
    ]


def candidate_audits() -> list[str]:
    lines = [audit("eligibility", "eligible", "scope=model lazy_pack=1 layers=43")]
    for path_index in range(2):
        lines.append(audit("context", "eligible", "lazy_pack=1 lazy_worker=1 validation=0 timing=0"))
        for tokens in (1, 16, 1, 1, 1, 1, 512, 1, 1):
            lines.append(audit("graph_mode", "fallback", f"graph_type=0 tokens={tokens} lora=0"))
        lines.extend([
            audit("graph_mode", "eligible", "mode=1 tokens=2048 lora=0 pack_ready=0"),
            audit("context_output", "allocated", f"bytes=33554432 base=0x{1000 + path_index:x}"),
            binding(0),
            audit("graph_mode", "fallback", "graph_type=0 tokens=1 lora=0"),
            audit("graph_mode", "eligible", "mode=1 tokens=2048 lora=0 pack_ready=0"),
            binding(0),
            audit("graph_mode", "eligible", "mode=1 tokens=2048 lora=0 pack_ready=0"),
        ])
        if path_index == 0:
            lines.extend([
                audit("pack", "begin", "layers=43 expected_bytes=4328521728"),
            ])
            for layer in range(43):
                for role_index, role in enumerate(("gate", "up", "down")):
                    lines.append(audit(
                        "pack_tensor", "pass",
                        f"layer={layer} role={role} tensor=0x{1 + 3 * layer + role_index:x} "
                        f"source=0x{0x100000000 + (3 * layer + role_index) * 0x1000000:x} "
                        "source_bytes=16777216 "
                        f"name=blk.{layer}.ffn_{role}_shexp.weight elements=8388608 exact_bf16_bits=1 "
                        "mapping=output_input_to_panel_k_col clamp_bits=0x41200000",
                    ))
            lines.append(audit(
                "pack_contract", "complete",
                "layers=43 matrices=129 unique_tensors=129 unique_sources=129 source_bytes=2164260864 "
                "values=1082130432 exact_bf16_expand=1 orientation=output_input_to_panel_k_col "
                "clamp_metadata=1 clamp_layers=43",
            ))
            lines.append(audit("pack", "complete", "layers=43 bytes=4328521728 ms=1.0"))
        lines.extend([
            audit("context", "allocated", "worker_scratch_bytes=67108864 oracle=0"),
            binding(1),
            audit("graph_reuse", "bound", "graph=0x1234 mode=1 prior_submissions=0"),
            audit("graph_reuse", "submit", "graph=0x1234 mode=1 submission=1 reused=0"),
        ])
        for layer in range(43):
            for edge in ("start", "end"):
                lines.append(audit("callback", "ask", f"submission=1 layer={layer} edge={edge} error=ok"))
                lines.append(audit("callback", "after", f"submission=1 layer={layer} edge={edge} error=ok"))
            lines.append(audit(
                "output_visibility", "ready",
                f"submission=1 layer={layer} shared_buffer=1 upload=0 worker_join=1 release_acquire=1",
            ))
        lines.extend([
            audit("callbacks", "complete",
                  "submission=1 expected_layer=43 expected_edge=start starts=0x7ffffffffff "
                  "ends=0x7ffffffffff completed_layers=43"),
            audit("graph", "complete", "submission=1 mode=1 layers=43"),
        ])
        for step in range(64):
            lines.append(audit("graph_mode", "fallback", "graph_type=0 tokens=1 lora=0"))
            if step in (0, 3):
                lines.append(audit("bindings", "fallback", "mode=disabled executable=1"))
    return lines


def process_log(mode: str, capture_bytes: int) -> str:
    lines = driver_lines(mode, capture_bytes)
    if mode == "control":
        lines.insert(1, audit("eligibility", "fallback", "scope=model reason=explicit_disable"))
    else:
        lines[1:1] = candidate_audits()
    lines.append("0 swaps")
    return "\n".join(lines) + "\n"


class FullModelGateTests(unittest.TestCase):
    def test_positive_captures_and_logs(self) -> None:
        self.assertEqual(GATE.LOGICAL_TOKENS, 2112)
        self.assertEqual(GATE.RUNTIME_N_CTX, 2304)
        self.assertEqual(GATE.RUNTIME_N_CTX % GATE.CONTEXT_ALIGNMENT, 0)
        self.assertEqual(GATE.CONTEXT_PADDING, 192)
        self.assertEqual(GATE.HEADER.size + 4 * GATE.LOGICAL_TOKENS, 8532)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            control = directory / "control.bin"
            candidate = directory / "candidate.bin"
            write_capture(control, 0, delta=0.0)
            write_capture(candidate, 1)
            comparison = GATE.compare_captures(control, candidate, COMMIT)
            self.assertEqual(comparison["outcome"], "telemetry")
            self.assertEqual(comparison["policy"], "f32_quality_first")
            self.assertFalse(comparison["metal_divergence_rejects"])
            self.assertEqual(comparison["logical_tokens"], 2112)
            self.assertEqual(comparison["runtime_n_ctx"], 2304)
            self.assertEqual(comparison["context_padding"], 192)
            self.assertEqual(comparison["total_vectors_per_process"], 130)
            self.assertEqual(len(comparison["control_greedy_tokens"]), 64)
            self.assertEqual(len(comparison["candidate_greedy_tokens"]), 64)
            self.assertLess(comparison["max_nmse"], GATE.LEGACY_LOGIT_NMSE_LIMIT)
            GATE.verify_process_log(process_log("control", control.stat().st_size), "control", COMMIT,
                                    comparison, control.stat().st_size, BEFORE, AFTER)
            candidate_report = GATE.verify_process_log(
                process_log("candidate", candidate.stat().st_size), "candidate", COMMIT,
                comparison, candidate.stat().st_size, BEFORE, AFTER)
            self.assertEqual(candidate_report["audit_counts"]["callbacks/complete"], 2)

    def test_adversarial_capture_fixtures(self) -> None:
        telemetry_cases = {
            "greedy divergence": {"diverge": ("greedy", 3)},
            "KL limit": {"delta": 2.0e-3},
            "metric limit": {"delta": 1.0},
        }
        for name, options in telemetry_cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                control = directory / "control.bin"
                candidate = directory / "candidate.bin"
                write_capture(control, 0, delta=0.0)
                write_capture(candidate, 1, **options)
                comparison = GATE.compare_captures(control, candidate, COMMIT)
                self.assertEqual(comparison["outcome"], "telemetry")

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            control = directory / "control.bin"
            candidate = directory / "candidate.bin"
            write_capture(control, 0, delta=0.0)
            write_capture(candidate, 1, nonfinite=("teacher", 7, 0))
            with self.assertRaises(GATE.GateFailure):
                GATE.compare_captures(control, candidate, COMMIT)

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            control = directory / "control.bin"
            candidate = directory / "candidate.bin"
            write_capture(control, 0, delta=0.0)
            write_capture(candidate, 1)
            candidate.write_bytes(candidate.read_bytes()[:-1])
            with self.assertRaises(GATE.GateFailure):
                GATE.compare_captures(control, candidate, COMMIT)

    def test_adversarial_audit_fixtures(self) -> None:
        capture = {
            "n_vocab": N_VOCAB,
            "prompt_token_fnv1a64": f"0x{GATE.token_hash(fixture_tokens()[0]):016x}",
            "teacher_token_fnv1a64": f"0x{GATE.token_hash(fixture_tokens()[1]):016x}",
        }
        valid = process_log("candidate", 1234)
        fallback = audit("graph_mode", "fallback", "graph_type=0 tokens=1 lora=0") + "\n"
        completion = audit("callbacks", "complete",
                           "submission=1 expected_layer=43 expected_edge=start starts=0x7ffffffffff "
                           "ends=0x7ffffffffff completed_layers=43") + "\n"
        mutations = {
            "unpadded start context": valid.replace("n_ctx=2304 logical_tokens=2112", "n_ctx=2112 logical_tokens=2112", 1),
            "runtime context mismatch": valid.replace(
                "path=teacher n_ctx=2304 n_batch=2048", "path=teacher n_ctx=2112 n_batch=2048", 1),
            "missing fallback": valid.replace(fallback, "", 1),
            "duplicate fallback": valid.replace(fallback, fallback + fallback, 1),
            "missing fallback binding": valid.replace(
                audit("bindings", "fallback", "mode=disabled executable=1") + "\n", "", 1),
            "missing completion": valid.replace(completion, "", 1),
            "poison": valid + audit("runtime", "poison", "output_poisoned=1 reason=test") + "\n",
            "validation enabled": valid.replace("validation=0 timing=0", "validation=1 timing=0", 1),
            "missing routed join proof": valid.replace(" routed_join_exact=1", "", 1),
            "wrong direct consumer count": valid.replace("direct_consumer_edges=43", "direct_consumer_edges=42", 1),
            "duplicate driver field": valid.replace("mode=candidate source_commit=", "mode=candidate mode=candidate source_commit=", 1),
            "malformed audit record": valid + "dsv4_amx_audit event=graph\n",
            "malformed full driver record": valid + "dsv4_amx_full_gate_driver event=complete outcome\n",
        }
        for name, log in mutations.items():
            with self.subTest(name=name), self.assertRaises(GATE.GateFailure):
                GATE.verify_process_log(log, "candidate", COMMIT, capture, 1234, BEFORE, AFTER)

        control = process_log("control", 1234)
        control = control.replace("0 swaps\n", audit("context_output", "allocated", "bytes=33554432 base=0x1") + "\n0 swaps\n")
        with self.assertRaises(GATE.GateFailure):
            GATE.verify_process_log(control, "control", COMMIT, capture, 1234, BEFORE, AFTER)

    def test_exact_mode_environment(self) -> None:
        parent = {
            "PATH": "/usr/bin:/bin",
            "LANG": "C",
            "GGML_METAL": "0",
            "LLAMA_DSV4_AMX_COEXEC_VALIDATE": "1",
            "METAL_DEBUG_ERROR_MODE": "1",
            "MTL_CAPTURE_ENABLED": "1",
            "DYLD_LIBRARY_PATH": "/tmp/injected",
            "ACCELERATE_NEW_LAPACK": "1",
            "OMP_NUM_THREADS": "99",
        }
        for mode, expected in GATE.MODE_ENVIRONMENTS.items():
            child = GATE.mode_child_environment(parent, mode)
            self.assertEqual(child["PATH"], parent["PATH"])
            self.assertEqual(child["LANG"], parent["LANG"])
            self.assertEqual(
                {key: value for key, value in child.items() if key.startswith(GATE.BASE.RUNTIME_ENV_PREFIXES)},
                expected,
            )
        with self.assertRaises(GATE.GateFailure):
            GATE.BASE.require_clean_runtime_environment(parent)

    def test_retained_oracle_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "PASS").write_text("production AMX F32 oracle gate passed\n", encoding="utf-8")
            verification = {
                "outcome": "pass",
                "expected_commit": COMMIT,
                "oracle_passes": 172,
                "metal_telemetry_records": 43,
                "completed_layers": 43,
                "callback_mask": "0x7ffffffffff",
                "system_swap_before_bytes": 0,
                "system_swap_after_bytes": 0,
                "resource_metrics": {"swaps": 0, "thermal_nominal": 1},
                "thermal_nominal": True,
                "benchmark": False,
                "model": {"set_sha256": GATE.ORACLE_MODEL_SET_SHA256},
            }
            (directory / "verification.json").write_text(json.dumps(verification), encoding="utf-8")
            (directory / "environment.json").write_text(
                json.dumps({"expected_commit": COMMIT}), encoding="utf-8")
            (directory / "model-set.json").write_text(
                json.dumps({"set_sha256": GATE.ORACLE_MODEL_SET_SHA256}), encoding="utf-8")
            (directory / "run.log").write_text("retained\n", encoding="utf-8")
            names = ["PASS", "environment.json", "model-set.json", "run.log", "verification.json"]
            manifest = "".join(f"{GATE._sha256(directory / name)}  {name}\n" for name in names)
            (directory / "artifact-sha256.txt").write_text(manifest, encoding="utf-8")
            result = GATE.verify_oracle_artifacts(directory, COMMIT)
            self.assertEqual(result["oracle_commit"], COMMIT)
            (directory / "run.log").write_text("corrupted\n", encoding="utf-8")
            with self.assertRaises(GATE.GateFailure):
                GATE.verify_oracle_artifacts(directory, COMMIT)


if __name__ == "__main__":
    unittest.main()
