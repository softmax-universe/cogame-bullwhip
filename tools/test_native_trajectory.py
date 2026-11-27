"""Run complete native socket games against a local Messages fixture; no paid calls."""

import contextlib
import http.server
import hashlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GAME, PLAYER = (str(Path(arg).resolve()) for arg in sys.argv[1:3])
for mode in ("accepted", "invalid", "sampled", "greedy-null", "greedy-tokens", "provider-error"):
    if os.environ.get("BULLWHIP_NATIVE_FIXTURE_MODE") and mode != os.environ["BULLWHIP_NATIVE_FIXTURE_MODE"]:
        continue
    calls = {}

    class Messages(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            assert self.path == "/v1/messages"
            request = json.loads(self.rfile.read(int(self.headers["content-length"])))
            assert request["temperature"] == (1 if mode == "sampled" else 0)
            assert request["model"] == "fixture/pinned"
            slot = int(self.headers["X-Coworld-Player-Slot"])
            text = (
                "invalid JSON"
                if mode == "invalid" and slot == 0
                else json.dumps({"order": 8, "say": "public forecast", "notes": "private-notebook-fixture"})
            )
            call_id = str(uuid.uuid4())
            body = {
                "id": "msg_" + call_id,
                "model": "fixture/served",
                "content": [{"type": "text", "text": text}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }
            if mode in {"sampled", "greedy-tokens"}:
                body["sampling_evidence"] = {
                    "policy_revision": "a" * 64,
                    "tokenizer_revision": "b" * 64,
                    "chat_template": "fixture-template",
                    "sampling": "full_softmax_temperature_one" if mode == "sampled" else "greedy",
                    "enable_thinking": False,
                    "max_new_tokens": request["max_tokens"],
                    "max_sequence_length": 4096,
                    "sampling_seed": 7,
                    "eos_token_ids": [4],
                    "prompt_token_ids": [1, 2],
                    "completion_token_ids": [3, 4],
                    "behavior_log_probs": [-0.5, -0.3] if mode == "sampled" else None,
                    "stop_reason": "eos",
                    "response": text,
                }
            if mode == "greedy-null":
                body["sampling_evidence"] = None
            failed = mode == "provider-error" and slot == 0
            if failed:
                body = {"error": {"message": "private-provider-error-fixture"}}
            calls[call_id] = (request, body)
            encoded = json.dumps(body).encode()
            self.send_response(500 if failed else 200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("X-Softmax-Llm-Call-Id", call_id)
            if mode in {"sampled", "greedy-tokens"}:
                self.send_header("X-Coworld-Checkpoint-Sha256", "a" * 64)
                self.send_header("X-Coworld-Tokenizer-Sha256", "b" * 64)
                self.send_header("X-Coworld-Chat-Template-Sha256", "c" * 64)
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, *_args):
            pass

    provider = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Messages)
    provider_owner = threading.Thread(target=provider.serve_forever)
    provider_owner.start()
    try:
        if len(sys.argv) == 4:
            target = Path(sys.argv[3]).resolve() / mode
            target.mkdir(parents=True, mode=0o700, exist_ok=False)
            output_context = contextlib.nullcontext(target)
        else:
            output_context = tempfile.TemporaryDirectory()
        with output_context as directory:
            output = Path(directory)
            with socket.socket() as reserve:
                reserve.bind(("127.0.0.1", 0))
                port = reserve.getsockname()[1]
            config = {
                "tokens": [f"t{s}" for s in range(4)],
                "players": [{"name": f"p{s}"} for s in range(4)],
                "seed": 7,
                "hostedModel": "fixture/pinned",
                "promptVariant": "mixed", "promptSeed": 4,
                "weeks": 4,
                "talk": True,
                "turnDelayMs": 0,
                "player_connect_timeout_seconds": 5,
                "llmTimeoutSeconds": 5,
            }
            (output / "config.json").write_text(json.dumps(config))
            env = {
                **os.environ,
                "COGAME_HOST": "127.0.0.1",
                "COGAME_PORT": str(port),
                "COGAME_CONFIG_URI": (output / "config.json").as_uri(),
                "COGAME_RESULTS_URI": (output / "results.json").as_uri(),
                "COGAME_SAVE_REPLAY_URI": (output / "replay.json").as_uri(),
                "COGAME_SAVE_TRAJECTORY_URI": (output / "trajectory.jsonl").as_uri(),
                "COWORLD_EPISODE_ID": str(uuid.uuid4()),
                "COWORLD_GAME_VERSION": "native-fixture",
                "COWORLD_SOURCE_REVISION": (
                    os.environ["COWORLD_SOURCE_REVISION"]
                    if "COWORLD_SOURCE_REVISION" in os.environ
                    else subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
                ),
                "COWORLD_LLM_ENDPOINT": f"http://127.0.0.1:{provider.server_port}",
                "COWORLD_LLM_MODEL": "fixture/requested",
                "COWORLD_LLM_TEMPERATURE": "1" if mode == "sampled" else "0",
                "PLAYER_PROMPT": "private-strategy-fixture",
                "PLAYER_SCRIPTED": "",
            }
            processes = []
            with (output / "game.log").open("w") as log:
                try:
                    processes.append(subprocess.Popen([GAME], cwd=ROOT, env=env, stdout=log, stderr=log))
                    for _ in range(100):
                        assert processes[0].poll() is None
                        with socket.socket() as probe:
                            if probe.connect_ex(("127.0.0.1", port)) == 0:
                                break
                        time.sleep(0.05)
                    else:
                        raise AssertionError("game did not start")
                    for seat in range(4):
                        player_env = {
                            **env,
                            "COWORLD_PLAYER_WS_URL": f"ws://127.0.0.1:{port}/player?slot={seat}&token=t{seat}",
                        }
                        processes.append(subprocess.Popen([PLAYER], cwd=ROOT, env=player_env, stdout=log, stderr=log))
                    failed_mode = mode in {"invalid", "provider-error"}
                    assert processes[0].wait(timeout=40) == (1 if failed_mode else 0), (output / "game.log").read_text()
                    for process in processes[1:]:
                        assert process.wait(timeout=40) == 0, (output / "game.log").read_text()
                    events = [json.loads(line) for line in (output / "trajectory.jsonl").read_text().splitlines()]
                    if failed_mode:
                        assert len(events[:-1]) == 4 and events[-1]["status"] == "failed"
                        assert events[-1]["participant_outcomes"] is None
                        assert not (output / "results.json").exists()
                        assert not (output / "replay.json").exists()
                    else:
                        assert len(events[:-1]) == 16 and events[-1]["status"] == "completed"
                        assert events[-1]["outcome"]["weeks"] == 4
                    recorded = set()
                    rejected = 0
                    for decision in events[:-1]:
                        assert decision["visibility"] == "private"
                        for attempt in decision["attempts"]:
                            assert attempt["inference_mode"] == "text_action"
                            call_id = attempt["platform_call_id"]
                            request, body = calls[call_id]
                            recorded.add(call_id)
                            assert attempt["request"] == request
                            assert attempt["decoder"] == {
                                "temperature": request["temperature"], "max_tokens": request["max_tokens"]
                            }
                            metadata = next(
                                item["prompt_template"] for item in decision["observation"]["prompt_attempts"]
                                if item["attempt_id"] == attempt["attempt_id"]
                            )
                            assert metadata["variant"] == "reordered"
                            assert metadata["system_sha256"] == hashlib.sha256(request["system"].encode()).hexdigest()
                            assert metadata["user_sha256"] == hashlib.sha256(request["messages"][0]["content"].encode()).hexdigest()
                            assert json.loads(attempt["raw_response"]) == body
                            assert attempt["prompt"] == [
                                {"role": "system", "content": request["system"]},
                                *request["messages"],
                            ]
                            if attempt["accepted"]:
                                assert attempt["model"] == "fixture/served"
                                assert attempt["parsed_action"] == decision["executed_action"]
                            else:
                                assert attempt["rejection_reason"]
                            if mode in {"sampled", "greedy-tokens"}:
                                assert attempt["prompt_token_ids"] == [1, 2]
                                assert attempt["sampled_token_ids"] == [3, 4]
                                assert attempt["behavior_logprobs"] == ([-0.5, -0.3] if mode == "sampled" else None)
                                assert attempt["model_identity"] == "a" * 64
                        if failed_mode:
                            assert decision["selected_attempt_id"] is None
                            assert decision["executed_action"] is None
                            if decision["seat"] == "0":
                                rejected += 1
                                assert decision["action_status"] == "rejected" and len(decision["attempts"]) == 2
                            else:
                                assert decision["action_status"] == "missing" and len(decision["attempts"]) == 1
                        else:
                            assert decision["action_status"] == "accepted"
                            assert decision["selected_attempt_id"] == decision["attempts"][-1]["attempt_id"]
                    assert recorded == set(calls)
                    assert rejected == (1 if failed_mode else 0)
                    public = (output / "game.log").read_text()
                    if not failed_mode:
                        public += (output / "replay.json").read_text()
                    for secret in [
                        "private-notebook-fixture",
                        "private-strategy-fixture",
                        "private-provider-error-fixture",
                        *calls,
                    ]:
                        assert secret not in public
                    assert (output / "trajectory.jsonl").stat().st_mode & 0o777 == 0o600
                    archive = output / "fixture_calls.json"
                    archive.write_text(
                        json.dumps(
                            {
                                "cohort": "local native HTTP fixture; no hosted platform archive",
                                "synthetic_model_identities": True,
                                "calls": calls,
                            }
                        )
                    )
                    archive.chmod(0o600)
                    print(mode, "16 decisions", len(recorded), "native fixture joins", flush=True)
                finally:
                    for process in processes:
                        if process.poll() is None:
                            process.terminate()
                            process.wait(timeout=5)
    finally:
        provider.shutdown()
        provider_owner.join(timeout=2)
        assert not provider_owner.is_alive()
        provider.server_close()

if not os.environ.get("BULLWHIP_NATIVE_FIXTURE_MODE"):
    manifest = json.loads((ROOT / "coworld_manifest_template.json").read_text())
    player_by_id = {player["id"]: player for player in manifest["player"]}
    model_player = player_by_id["bullwhip-player"]
    assert model_player.get("env", {}).get("COWORLD_LLM_ENABLED") == "true"
    assert model_player.get("env", {}).get("COWORLD_LLM_MODEL")
    cert_players = [seat["player_id"] for seat in manifest["certification"]["players"]]
    assert cert_players[0] == "bullwhip-player"
    assert set(cert_players) == set(player_by_id)

    for cert_mode in ("cert-lineup", "no-endpoint"):
        cert_calls = []

        class CertMessages(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                assert self.path == "/v1/messages"
                request = json.loads(self.rfile.read(int(self.headers["content-length"])))
                slot = int(self.headers["X-Coworld-Player-Slot"])
                cert_calls.append((slot, request["model"]))
                call_id = str(uuid.uuid4())
                # Fenced JSON with omitted optional say/notes (no null boilerplate).
                body = {
                    "id": "msg_" + call_id,
                    "model": request["model"],
                    "content": [{"type": "text", "text": "```json\n{\"order\": 7}\n```"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                }
                encoded = json.dumps(body).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.send_header("X-Softmax-Llm-Call-Id", call_id)
                self.end_headers()
                self.wfile.write(encoded)

            def log_message(self, *_args):
                pass

        cert_provider = http.server.ThreadingHTTPServer(("127.0.0.1", 0), CertMessages)
        cert_owner = threading.Thread(target=cert_provider.serve_forever)
        cert_owner.start()
        try:
            if len(sys.argv) == 4:
                target = Path(sys.argv[3]).resolve() / cert_mode
                target.mkdir(parents=True, mode=0o700, exist_ok=False)
                cert_ctx = contextlib.nullcontext(target)
            else:
                cert_ctx = tempfile.TemporaryDirectory()
            with cert_ctx as directory:
                output = Path(directory)
                with socket.socket() as reserve:
                    reserve.bind(("127.0.0.1", 0))
                    port = reserve.getsockname()[1]
                config = dict(manifest["certification"]["game_config"])
                config["tokens"] = [f"t{s}" for s in range(4)]
                config["weeks"] = 4
                config["player_connect_timeout_seconds"] = 5
                config["llmTimeoutSeconds"] = 5
                (output / "config.json").write_text(json.dumps(config))
                base_env = {
                    **os.environ,
                    "COGAME_HOST": "127.0.0.1",
                    "COGAME_PORT": str(port),
                    "COGAME_CONFIG_URI": (output / "config.json").as_uri(),
                    "COGAME_RESULTS_URI": (output / "results.json").as_uri(),
                    "COGAME_SAVE_REPLAY_URI": (output / "replay.json").as_uri(),
                    "COGAME_SAVE_TRAJECTORY_URI": (output / "trajectory.jsonl").as_uri(),
                    "COWORLD_EPISODE_ID": str(uuid.uuid4()),
                    "COWORLD_GAME_VERSION": "cert-lineup-fixture",
                    "COWORLD_SOURCE_REVISION": (
                        os.environ["COWORLD_SOURCE_REVISION"]
                        if "COWORLD_SOURCE_REVISION" in os.environ
                        else subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
                    ),
                    "COWORLD_LLM_ENDPOINT": (
                        "" if cert_mode == "no-endpoint" else f"http://127.0.0.1:{cert_provider.server_port}"
                    ),
                    "COWORLD_LLM_MODEL": model_player["env"]["COWORLD_LLM_MODEL"],
                    "COWORLD_LLM_TEMPERATURE": "0",
                }
                processes = []
                with (output / "game.log").open("w") as log:
                    try:
                        processes.append(subprocess.Popen([GAME], cwd=ROOT, env=base_env, stdout=log, stderr=log))
                        for _ in range(100):
                            assert processes[0].poll() is None
                            with socket.socket() as probe:
                                if probe.connect_ex(("127.0.0.1", port)) == 0:
                                    break
                            time.sleep(0.05)
                        else:
                            raise AssertionError("game did not start")
                        for seat, player_id in enumerate(cert_players):
                            role_env = player_by_id[player_id].get("env", {})
                            player_env = {
                                **base_env,
                                "PLAYER_SCRIPTED": "",
                                **role_env,
                                "COWORLD_PLAYER_WS_URL": f"ws://127.0.0.1:{port}/player?slot={seat}&token=t{seat}",
                            }
                            processes.append(subprocess.Popen([PLAYER], cwd=ROOT, env=player_env, stdout=log, stderr=log))
                        if cert_mode == "no-endpoint":
                            assert processes[0].wait(timeout=40) == 1, (output / "game.log").read_text()
                            for process in processes[1:]:
                                assert process.wait(timeout=40) == 0, (output / "game.log").read_text()
                            assert not (output / "results.json").exists()
                            assert not (output / "replay.json").exists()
                            events = [json.loads(line) for line in (output / "trajectory.jsonl").read_text().splitlines()]
                            assert events[-1]["status"] == "failed"
                            assert events[-1]["participant_outcomes"] is None
                            assert cert_calls == []
                        else:
                            for process in processes:
                                assert process.wait(timeout=40) == 0, (output / "game.log").read_text()
                            assert [slot for slot, _ in cert_calls] == [0, 0, 0, 0]
                            replay = json.loads((output / "replay.json").read_text())
                            slot0_orders = [
                                ev["order"]
                                for ev in replay["events"]
                                if ev["kind"] == "order" and ev["seat"] == 0
                            ]
                            assert slot0_orders == [7, 7, 7, 7]
                            results = json.loads((output / "results.json").read_text())
                            assert results["weeks"] == 4 and results["reason"] == "complete"
                            events = [json.loads(line) for line in (output / "trajectory.jsonl").read_text().splitlines()]
                            assert events[-1]["status"] == "completed"
                        print(cert_mode, len(cert_calls), "HTTP model calls", flush=True)
                    finally:
                        for process in processes:
                            if process.poll() is None:
                                process.terminate()
                                process.wait(timeout=5)
        finally:
            cert_provider.shutdown()
            cert_owner.join(timeout=2)
            assert not cert_owner.is_alive()
            cert_provider.server_close()
