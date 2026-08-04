import copy
import io
import json
import pathlib
import unittest

import live_probe


FIXTURES = pathlib.Path(__file__).resolve().parents[1] / "fixtures"


def target_snapshot():
    value = json.loads((FIXTURES / "state.json").read_text(encoding="utf-8"))
    value["availability"]["content"] = False
    for request in value["requests"]:
        request["content"] = {"prompt": "", "output": "", "retained": False}
    return value


class FakeResponse(io.BytesIO):
    def readline(self, size=-1):
        return super().readline(size)


class LiveProbeTests(unittest.TestCase):
    def test_snapshot_v2_wire_contract_and_redaction(self):
        snapshot = target_snapshot()
        self.assertIs(live_probe.validate_snapshot(snapshot, ["not-present"]), snapshot)
        exposed = copy.deepcopy(snapshot)
        exposed["requests"][0]["content"]["prompt"] = "secret prompt"
        with self.assertRaisesRegex(live_probe.ProbeError, "exposed"):
            live_probe.validate_snapshot(exposed, [])
        reflected = copy.deepcopy(snapshot)
        reflected["server"]["build"] = "api-secret"
        with self.assertRaisesRegex(live_probe.ProbeError, "credential"):
            live_probe.validate_snapshot(reflected, ["api-secret"])

    def test_fast_refill_invariants_and_state_labels(self):
        value = target_snapshot()["fast_refill"]
        self.assertIn("window_open", live_probe.refill_state_labels(value))
        self.assertIn("full_width", live_probe.refill_state_labels(value))
        invalid = copy.deepcopy(value)
        invalid["refill"]["fast_members_remaining"] = 0
        with self.assertRaisesRegex(live_probe.ProbeError, "member counts"):
            live_probe.validate_fast_refill(invalid)

    def test_event_v2_wire_contract_requires_refill_on_request_events(self):
        events = json.loads((FIXTURES / "events.json").read_text(encoding="utf-8"))
        for event in events:
            self.assertIs(live_probe.validate_event(event), event)
        request_event = next(event for event in events if event["type"] == "request.upsert")
        missing = copy.deepcopy(request_event)
        del missing["payload"]["fast_refill"]
        with self.assertRaisesRegex(live_probe.ProbeError, "authoritative fast_refill"):
            live_probe.validate_event(missing)

    def test_sse_capture_parses_id_and_json_with_bounds(self):
        event = json.loads((FIXTURES / "events.json").read_text(encoding="utf-8"))[0]
        wire = f"id: {event['id']}\ndata: {json.dumps(event)}\n\n".encode()
        capture = live_probe.SseCapture(FakeResponse(wire))
        capture.start()
        capture.thread.join(1)
        self.assertIsNotNone(capture.error)
        self.assertEqual(capture.events, [event])

    def test_loopback_url_rejects_credentials_and_queries(self):
        self.assertEqual(
            live_probe.normalize_loopback_url("http://127.0.0.1:18130/"),
            "http://127.0.0.1:18130",
        )
        for value in (
            "http://example.com:18130",
            "http://user:pass@127.0.0.1:18130",
            "http://127.0.0.1:18130?api_key=value",
        ):
            with self.assertRaises(live_probe.ProbeError):
                live_probe.normalize_loopback_url(value)


if __name__ == "__main__":
    unittest.main()
