#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "dsv4-amx-production-gate.py"
SPEC = importlib.util.spec_from_file_location("dsv4_amx_production_gate", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
GATE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = GATE
SPEC.loader.exec_module(GATE)

COMMIT = "a" * 40
NOMINAL_THERMAL = "\n".join(f"Note: {message}" for message in GATE.THERMAL_NOMINAL_MESSAGES)
BEFORE = {"swap_used_bytes": 0, "thermal_state": NOMINAL_THERMAL}
AFTER = {"swap_used_bytes": 0, "thermal_state": NOMINAL_THERMAL}


def audit(event: str, outcome: str, detail: str = "") -> str:
    return f"dsv4_amx_audit event={event} outcome={outcome} {detail}".rstrip()


def driver(event: str, outcome: str, detail: str = "") -> str:
    return f"dsv4_amx_gate_driver event={event} outcome={outcome} {detail}".rstrip()


def positive_fixture() -> str:
    lines = [
        driver("start", "begin", f"source_commit={COMMIT} build_commit={COMMIT[:9]} tokens=2048 submissions=1 benchmark=0"),
        audit("eligibility", "eligible", "scope=model lazy_pack=1 layers=43"),
        audit("context", "eligible", "lazy_pack=1 lazy_worker=1 validation=1 timing=0"),
    ]
    for tokens in (1, 16, 1, 1, 1, 1, 512, 1, 1):
        lines.append(audit("graph_mode", "fallback", f"graph_type=0 tokens={tokens} lora=0"))
    lines.extend([
        audit("graph_mode", "eligible", "mode=2 tokens=2048 lora=0 pack_ready=0"),
        audit("context_output", "allocated", "bytes=33554432 base=0x1234"),
        audit("bindings", "complete",
              "executable=0 layers=43 callback_events=86 tensor_metadata=43 external_output_bytes=33554432 "
              "gallocr_output_bytes=0 unique_buffers=1 unique_data=1 producerless=1 "
              "callback_edges=norm_completion visible_before_consumer=1 routed_join_exact=1 "
              "routed_joins=43 telemetry_consumers=43 direct_consumer_edges=86"),
        audit("graph_mode", "fallback", "graph_type=0 tokens=1 lora=0"),
        audit("graph_mode", "eligible", "mode=2 tokens=2048 lora=0 pack_ready=0"),
        audit("bindings", "complete",
              "executable=0 layers=43 callback_events=86 tensor_metadata=43 external_output_bytes=33554432 "
              "gallocr_output_bytes=0 unique_buffers=1 unique_data=1 producerless=1 "
              "callback_edges=norm_completion visible_before_consumer=1 routed_join_exact=1 "
              "routed_joins=43 telemetry_consumers=43 direct_consumer_edges=86"),
        driver("decode", "begin", "tokens=2048 submissions=1"),
        audit("graph_mode", "eligible", "mode=2 tokens=2048 lora=0 pack_ready=0"),
        audit("pack", "begin", "layers=43 expected_bytes=4328521728"),
    ])
    for layer in range(43):
        for role_index, role in enumerate(("gate", "up", "down")):
            lines.append(audit(
                "pack_tensor", "pass",
                f"layer={layer} role={role} tensor=0x{1 + 3 * layer + role_index:x} "
                f"source=0x{0x100000000 + (3 * layer + role_index) * 0x1000000:x} source_bytes=16777216 "
                f"name=blk.{layer}.ffn_{role}_shexp.weight elements=8388608 exact_bf16_bits=1 "
                "mapping=output_input_to_panel_k_col clamp_bits=0x41200000",
            ))
    lines.extend([
        audit("pack_contract", "complete",
              "layers=43 matrices=129 unique_tensors=129 unique_sources=129 source_bytes=2164260864 "
              "values=1082130432 exact_bf16_expand=1 orientation=output_input_to_panel_k_col "
              "clamp_metadata=1 clamp_layers=43"),
        audit("pack", "complete", "layers=43 bytes=4328521728 ms=1.0"),
        audit("oracle", "begin", "layers=43 rows=4 nmse_limit=1.000000000e-12 max_abs_limit=5.000000000e-04"),
    ])
    for layer in range(43):
        for stage, cols in (("gate", 2048), ("up", 2048), ("post_swiglu", 2048), ("down", 4096)):
            lines.append(audit(
                "oracle", "pass",
                f"layer={layer} stage={stage} finite=1 nonzero=1 reference_l2_valid=1 metrics_finite=1 "
                f"full_actual_elements={2048 * cols} compared={4 * cols} nmse=0.0 "
                "nmse_limit=1.000000000e-12 max_abs=0.0 max_abs_limit=5.000000000e-04 reference_l2=1.0",
            ))
    lines.extend([
        audit("oracle", "complete", "layers=43 stage_records=172"),
        audit("context", "allocated", "worker_scratch_bytes=67108864 oracle=1"),
        audit("bindings", "complete",
              "executable=1 layers=43 callback_events=86 tensor_metadata=43 external_output_bytes=33554432 "
              "gallocr_output_bytes=0 unique_buffers=1 unique_data=1 producerless=1 "
              "callback_edges=norm_completion visible_before_consumer=1 routed_join_exact=1 "
              "routed_joins=43 telemetry_consumers=43 direct_consumer_edges=86"),
        audit("graph_reuse", "bound", "graph=0x5678 mode=2 prior_submissions=0"),
        audit("graph_reuse", "submit", "graph=0x5678 mode=2 submission=1 reused=0"),
    ])
    for layer in range(43):
        for edge in ("start", "end"):
            lines.append(audit("callback", "ask", f"submission=1 layer={layer} edge={edge} error=ok"))
            lines.append(audit("callback", "after", f"submission=1 layer={layer} edge={edge} error=ok"))
        lines.append(audit(
            "metal_telemetry", "recorded",
            f"layer={layer} stage=down direct_finite=1 direct_nonzero=1 reference_finite=1 "
            "reference_l2_valid=1 metrics_finite=1 nmse=1.0e-2 max_abs=2.0 reference_l2=1.0 "
            "legacy_nmse_limit=1.000000000e-03 within_legacy_nmse=0 divergence_rejects=0",
        ))
        lines.append(audit(
            "output_visibility", "ready",
            f"submission=1 layer={layer} shared_buffer=1 upload=0 worker_join=1 release_acquire=1",
        ))
    lines.extend([
        audit("callbacks", "complete",
              "submission=1 expected_layer=43 expected_edge=start starts=0x7ffffffffff ends=0x7ffffffffff "
              "completed_layers=43"),
        audit("graph", "complete", "submission=1 mode=2 layers=43"),
        driver("decode", "pass", "tokens=2048 submissions=1"),
        driver("context_teardown", "pass"),
        driver("model_teardown", "pass"),
        driver("backend_teardown", "pass"),
        driver("complete", "pass", "tokens=2048 submissions=1 benchmark=0"),
        "0 swaps",
    ])
    return "\n".join(lines) + "\n"


class GateParserTests(unittest.TestCase):
    def test_positive_fixture(self) -> None:
        report = GATE.verify_gate(positive_fixture(), COMMIT, BEFORE, AFTER)
        self.assertEqual(report["outcome"], "pass")
        self.assertEqual(report["oracle_passes"], 172)
        self.assertEqual(report["metal_telemetry_records"], 43)
        self.assertEqual(report["metal_telemetry"]["max_nmse"], 1.0e-2)
        self.assertEqual(report["metal_telemetry"]["outside_legacy_nmse_layers"], 43)
        self.assertEqual(report["callback_mask"], "0x7ffffffffff")
        self.assertFalse(report["benchmark"])

    def test_adversarial_log_fixtures(self) -> None:
        valid = positive_fixture()
        oracle_line = next(line for line in valid.splitlines() if "event=oracle outcome=pass" in line)
        pack_tensor_line = next(line for line in valid.splitlines() if "event=pack_tensor outcome=pass" in line)
        allocation_line = next(line for line in valid.splitlines() if "event=context_output outcome=allocated" in line)
        fallback_line = next(line for line in valid.splitlines() if "event=graph_mode outcome=fallback" in line)
        telemetry_line = next(line for line in valid.splitlines() if "event=metal_telemetry outcome=recorded" in line)
        teardown_line = driver("context_teardown", "pass") + "\n"
        mutations = {
            "missing oracle stage": valid.replace(oracle_line + "\n", "", 1),
            "wrong oracle geometry": valid.replace("full_actual_elements=4194304", "full_actual_elements=4194303", 1),
            "wrong packed bytes": valid.replace("bytes=4328521728", "bytes=4328521727", 1),
            "missing packed tensor": valid.replace(pack_tensor_line + "\n", "", 1),
            "wrong packed tensor role": valid.replace("layer=0 role=gate", "layer=0 role=up", 1),
            "wrong packed tensor name": valid.replace("name=blk.0.ffn_gate_shexp.weight",
                                                       "name=blk.1.ffn_gate_shexp.weight", 1),
            "wrong packed tensor elements": valid.replace("elements=8388608", "elements=8388607", 1),
            "wrong packed tensor conversion": valid.replace("exact_bf16_bits=1", "exact_bf16_bits=0", 1),
            "wrong packed tensor mapping": valid.replace(
                "mapping=output_input_to_panel_k_col", "mapping=input_output_to_panel_k_col", 1),
            "duplicate packed tensor address": valid.replace("role=up tensor=0x2", "role=up tensor=0x1", 1),
            "duplicate packed source": valid.replace("source=0x101000000", "source=0x100000000", 1),
            "overlapping packed source": valid.replace("source=0x101000000", "source=0x100800000", 1),
            "wrong packed source size": valid.replace("source_bytes=16777216", "source_bytes=16777214", 1),
            "wrong clamp metadata": valid.replace("clamp_bits=0x41200000", "clamp_bits=0x41100000", 1),
            "missing routed join proof": valid.replace(" routed_join_exact=1", "", 1),
            "wrong routed join count": valid.replace("routed_joins=43", "routed_joins=42", 1),
            "duplicate shared allocation": valid.replace(allocation_line + "\n", allocation_line + "\n" + allocation_line + "\n", 1),
            "missing graph fallback": valid.replace(fallback_line + "\n", "", 1),
            "duplicate graph fallback": valid.replace(fallback_line + "\n", fallback_line + "\n" + fallback_line + "\n", 1),
            "wrong fallback token": valid.replace("graph_type=0 tokens=16 lora=0", "graph_type=0 tokens=2 lora=0", 1),
            "wrong fallback graph type": valid.replace(fallback_line, fallback_line.replace("graph_type=0", "graph_type=1"), 1),
            "wrong fallback LoRA state": valid.replace(fallback_line, fallback_line.replace("lora=0", "lora=1"), 1),
            "poison audit": valid + audit("runtime", "poison", "output_poisoned=1 reason=test") + "\n",
            "wrong callback mask": valid.replace("starts=0x7ffffffffff", "starts=0x3ffffffffff", 1),
            "nonzero swap": valid.replace("0 swaps", "1 swaps", 1),
            "masked nonzero swap": valid.replace("0 swaps", "1 swaps\n0 swaps", 1),
            "wrong source commit": valid.replace(f"source_commit={COMMIT}", f"source_commit={'b' * 40}", 1),
            "missing teardown": valid.replace(teardown_line, "", 1),
            "unexpected audit outcome": valid + audit("oracle", "maybe", "layer=0") + "\n",
            "loose oracle result": valid.replace("nmse=0.0 nmse_limit=1.000000000e-12",
                                                   "nmse=2.0e-12 nmse_limit=1.000000000e-12", 1),
            "invalid telemetry health": valid.replace(
                telemetry_line, telemetry_line.replace("direct_nonzero=1", "direct_nonzero=0"), 1),
            "duplicate audit field": valid.replace("scope=model lazy_pack=1", "scope=model scope=model lazy_pack=1", 1),
            "malformed audit record": valid + "dsv4_amx_audit event=oracle\n",
            "malformed driver record": valid + "dsv4_amx_gate_driver event=complete outcome\n",
            "ambiguous audit record": valid + f"{audit('oracle', 'pass')} {audit('oracle', 'pass')}\n",
        }
        for name, fixture in mutations.items():
            with self.subTest(name=name):
                with self.assertRaises(GATE.GateFailure):
                    GATE.verify_gate(fixture, COMMIT, BEFORE, AFTER)

    def test_adversarial_resource_fixture(self) -> None:
        with self.assertRaises(GATE.GateFailure):
            GATE.verify_gate(
                positive_fixture(), COMMIT, BEFORE, {"swap_used_bytes": 1024, "thermal_state": NOMINAL_THERMAL})
        with self.assertRaises(GATE.GateFailure):
            GATE.verify_gate(
                positive_fixture(), COMMIT, BEFORE,
                {"swap_used_bytes": 0, "thermal_state": "CPU_Speed_Limit = 80"})

    def test_canonical_model_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for index in range(1, 6):
                (directory / f"DeepSeek-V4-Flash-0731-UD-Q8_K_XL-{index:05d}-of-00005.gguf").write_bytes(b"x")
            first = directory / "DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf"
            shards = GATE.canonical_model_shards(first)
            self.assertEqual(len(shards), 5)
            artifact_dir = directory / "artifacts"
            artifact_dir.mkdir()
            fixture_digest = hashlib.sha256(b"x").hexdigest()
            with mock.patch.object(GATE, "CANONICAL_MODEL_SHA256", (fixture_digest,) * 5):
                model_set = GATE.hash_model_set(shards, artifact_dir)
            self.assertEqual(model_set["shard_count"], 5)
            self.assertEqual(model_set["total_bytes"], 5)
            self.assertTrue((artifact_dir / "model-sha256.txt").is_file())
            self.assertTrue((artifact_dir / "model-set.json").is_file())
            first.unlink()
            first.symlink_to(directory / "DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00002-of-00005.gguf")
            with self.assertRaises(GATE.GateFailure):
                GATE.canonical_model_shards(first)
            first.unlink()
            first.write_bytes(b"x")
            (directory / "DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00005-of-00005.gguf").unlink()
            with self.assertRaises(GATE.GateFailure):
                GATE.canonical_model_shards(first)

    def test_canonical_digest_contract(self) -> None:
        GATE.validate_canonical_model_digests(GATE.CANONICAL_MODEL_SHA256)
        corrupted = list(GATE.CANONICAL_MODEL_SHA256)
        corrupted[2] = "0" * 64
        with self.assertRaises(GATE.GateFailure):
            GATE.validate_canonical_model_digests(corrupted)

    def test_runtime_environment_contract(self) -> None:
        parent = {
            "PATH": "/usr/bin:/bin",
            "LANG": "C",
            "GGML_METAL": "1",
            "LLAMA_ARG_N_GPU_LAYERS": "0",
            "METAL_DEVICE_WRAPPER_TYPE": "1",
            "MTL_CAPTURE_ENABLED": "1",
            "DYLD_LIBRARY_PATH": "/tmp/injected",
            "ACCELERATE_NEW_LAPACK": "1",
            "OMP_NUM_THREADS": "99",
        }
        self.assertEqual(
            GATE.forbidden_runtime_environment(parent),
            sorted(key for key in parent if key not in ("PATH", "LANG")),
        )
        with self.assertRaises(GATE.GateFailure):
            GATE.require_clean_runtime_environment(parent)
        GATE.require_clean_runtime_environment({"PATH": parent["PATH"], "LANG": parent["LANG"]})
        child = GATE.gate_child_environment(parent)
        self.assertEqual(child["PATH"], parent["PATH"])
        self.assertEqual(child["LANG"], parent["LANG"])
        self.assertEqual(
            {key: value for key, value in child.items() if key.startswith(GATE.RUNTIME_ENV_PREFIXES)},
            GATE.GATE_ENVIRONMENT,
        )

    def test_model_child_timeout_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "timeout.log"
            with self.assertRaises(GATE.GateFailure):
                GATE._run_logged(
                    [sys.executable, "-c", "import time; time.sleep(60)"], ROOT, log,
                    required=False, timeout_seconds=0.05)
            self.assertIn("outcome=timeout", log.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
