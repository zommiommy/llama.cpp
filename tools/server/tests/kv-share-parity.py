#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Standalone kv-share cross-slot parity regression test.

This script intentionally does not use pytest: it is a GPU-gated local regression
for the forced-borrow server scenario that exercises both the f16 alias path and
the q8_0 copy path. Missing binary/model/GPU prints SKIP and exits 0; real
server/parity/path failures print FAIL and exit non-zero.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

HOST = "127.0.0.1"
PORT = int(os.environ.get("KV_SHARE_TEST_PORT", "8117"))

TEST_DIR = Path(__file__).resolve().parent
REPO_ROOT = TEST_DIR.parents[2]
SERVER_BIN = REPO_ROOT / "build" / "bin" / "llama-server"
MODEL = Path(os.environ.get("KV_SHARE_TEST_MODEL", "/home/zom/bench-emb/gguf/qwen3-0.6b-Q8_0.gguf"))
CUDA_FULL = Path(os.environ.get("KV_SHARE_TEST_CUDA_FULL", "/nix/store/0fkc0d78nb8q36jbj0dm86xw799hcnlp-cuda-full-12.8"))
CUDA_LIBRARY_DIRS = [Path("/run/opengl-driver/lib"), CUDA_FULL / "lib"]

SYS_PREFIX = (
    "You are a meticulous assistant. "
    + "The quick brown fox jumps over the lazy dog. " * 130
)
BORROWER_BODY: dict[str, object] = {
    "prompt": SYS_PREFIX + " Question B: name three colors.",
    "n_predict": 24,
    "temperature": 0,
    "seed": 0,
    "cache_prompt": True,
}
OWNER_BODY: dict[str, object] = {
    "prompt": SYS_PREFIX + " Question A: write a very long detailed story.",
    "n_predict": 400,
    "temperature": 0,
    "seed": 0,
}

REUSE_RE = re.compile(
    r"kv-share: reused\s+(?P<reused>\d+)/(?:\d+)\s+tokens.*?"
    r"aliased=(?P<aliased>\d+).*?copied=(?P<copied>\d+)",
)
FATAL_LOG_MARKERS = ("CUDA error", "illegal memory access", "GGML_ASSERT")


class TestFailure(Exception):
    """Raised for a real regression-test failure."""


@dataclass(frozen=True)
class ReuseStats:
    line: str
    reused: int
    aliased: int
    copied: int


@dataclass
class ServerRun:
    cache_type: str
    process: subprocess.Popen[str] | None = None
    output_lines: list[str] = field(default_factory=list)
    output_lock: threading.Lock = field(default_factory=threading.Lock)
    reader_thread: threading.Thread | None = None
    launched: bool = False

    def __enter__(self) -> "ServerRun":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.stop()

    def start(self) -> None:
        if is_port_open(PORT):
            raise TestFailure(f"port {PORT} is already in use before this test started")

        cmd = [
            str(SERVER_BIN),
            "-m",
            str(MODEL),
            "--kv-lazy",
            "--kv-share",
            "--no-context-shift",
            "--cache-type-k",
            self.cache_type,
            "--cache-type-v",
            self.cache_type,
            "--flash-attn",
            "on",
            "--cache-ram",
            "0",
            "-sps",
            "0",
            "-np",
            "2",
            "-c",
            "8192",
            "--host",
            HOST,
            "--port",
            str(PORT),
        ]
        env = os.environ.copy()
        ld_parts = [str(path) for path in CUDA_LIBRARY_DIRS if path.is_dir()]
        if env.get("LD_LIBRARY_PATH"):
            ld_parts.append(env["LD_LIBRARY_PATH"])
        if ld_parts:
            env["LD_LIBRARY_PATH"] = ":".join(ld_parts)

        self.process = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )
        self.launched = True
        self.reader_thread = threading.Thread(target=self._read_output, daemon=True)
        self.reader_thread.start()
        self._wait_ready(timeout_seconds=120.0)

    def stop(self) -> None:
        process = self.process
        if process is None:
            return

        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                process.kill()
                try:
                    process.wait(timeout=10.0)
                except subprocess.TimeoutExpired:
                    if self.launched:
                        subprocess.run(
                            ["pkill", "-f", f"llama-server.*{PORT}"],
                            check=False,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                        process.wait(timeout=10.0)
        self.process = None
        if self.reader_thread is not None:
            self.reader_thread.join(timeout=2.0)

    def assert_alive(self) -> None:
        process = self.process
        if process is None:
            raise TestFailure("server process handle is missing")
        if process.poll() is not None:
            raise TestFailure(
                f"server exited unexpectedly with code {process.returncode}\n{self.tail()}"
            )

    def output(self) -> str:
        with self.output_lock:
            return "".join(self.output_lines)

    def tail(self, max_lines: int = 80) -> str:
        with self.output_lock:
            return "".join(self.output_lines[-max_lines:])

    def wait_for_reuse_log(self, cache_type: str, timeout_seconds: float = 10.0) -> str:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            self.assert_alive()
            logs = self.output()
            assert_no_fatal_logs(logs, cache_type)
            if REUSE_RE.search(logs):
                return logs
            time.sleep(0.1)
        logs = self.output()
        assert_no_fatal_logs(logs, cache_type)
        raise TestFailure(
            f"[{cache_type}] missing 'kv-share: reused' log line after borrower returned\n"
            f"{self.tail()}"
        )

    def _read_output(self) -> None:
        process = self.process
        if process is None or process.stdout is None:
            return
        for line in process.stdout:
            with self.output_lock:
                self.output_lines.append(line)

    def _wait_ready(self, timeout_seconds: float) -> None:
        deadline = time.monotonic() + timeout_seconds
        last_error = "server did not answer /health"
        while time.monotonic() < deadline:
            process = self.process
            if process is not None and process.poll() is not None:
                raise TestFailure(
                    f"server exited during startup with code {process.returncode}\n{self.tail()}"
                )
            try:
                status, _body = request_json("GET", "/health", timeout_seconds=2.0)
                if status == 200:
                    return
                last_error = f"/health returned HTTP {status}"
            except OSError as exc:
                last_error = str(exc)
            except TestFailure as exc:
                last_error = str(exc)
            time.sleep(0.5)
        raise TestFailure(f"server did not become ready: {last_error}\n{self.tail()}")


def request_json(
    method: str,
    path: str,
    body: Mapping[str, object] | None = None,
    timeout_seconds: float = 120.0,
) -> tuple[int, object]:
    url = f"http://{HOST}:{PORT}{path}"
    encoded_body = None
    headers = {"Accept": "application/json"}
    if body is not None:
        encoded_body = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=encoded_body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = response.read().decode("utf-8")
            return response.status, json.loads(payload) if payload else None
    except urllib.error.HTTPError as exc:
        payload = exc.read().decode("utf-8", errors="replace")
        try:
            parsed: object = json.loads(payload)
        except json.JSONDecodeError:
            parsed = payload
        return exc.code, parsed
    except urllib.error.URLError as exc:
        raise TestFailure(f"request to {path} failed: {exc.reason}") from exc
    except OSError as exc:
        raise TestFailure(f"request to {path} failed: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise TestFailure(f"{path} returned invalid JSON: {exc}") from exc


def completion_content(body: Mapping[str, object], timeout_seconds: float = 180.0) -> str:
    status, payload = request_json("POST", "/completion", body, timeout_seconds)
    if status != 200:
        raise TestFailure(f"/completion returned HTTP {status}: {payload!r}")
    if not isinstance(payload, dict):
        raise TestFailure(f"/completion returned non-object JSON: {payload!r}")
    content = payload.get("content")
    if not isinstance(content, str):
        raise TestFailure(f"/completion response has no string content: {payload!r}")
    return content


def run_cold(cache_type: str) -> str:
    print(f"[{cache_type}] cold baseline")
    with ServerRun(cache_type) as server:
        content = completion_content(BORROWER_BODY)
        server.assert_alive()
        assert_no_fatal_logs(server.output(), cache_type)
        return content


def run_borrow(cache_type: str) -> tuple[str, str]:
    print(f"[{cache_type}] forced borrow")
    owner_errors: list[BaseException] = []

    def run_owner() -> None:
        try:
            completion_content(OWNER_BODY, timeout_seconds=300.0)
        except BaseException as exc:
            owner_errors.append(exc)

    with ServerRun(cache_type) as server:
        owner_thread = threading.Thread(target=run_owner, name=f"owner-{cache_type}")
        owner_thread.start()
        time.sleep(5.0)
        if owner_errors and not owner_thread.is_alive():
            raise TestFailure(f"owner request failed before borrower: {owner_errors[0]!r}")
        content = completion_content(BORROWER_BODY)
        server.assert_alive()
        logs = server.wait_for_reuse_log(cache_type)
    owner_thread.join(timeout=10.0)
    return content, logs


def extract_reuse_stats(logs: str, cache_type: str) -> ReuseStats:
    matches = list(REUSE_RE.finditer(logs))
    if not matches:
        raise TestFailure(f"[{cache_type}] missing 'kv-share: reused' log line")
    match = matches[-1]
    return ReuseStats(
        line=match.group(0),
        reused=int(match.group("reused")),
        aliased=int(match.group("aliased")),
        copied=int(match.group("copied")),
    )


def assert_no_fatal_logs(logs: str, cache_type: str) -> None:
    for marker in FATAL_LOG_MARKERS:
        if marker in logs:
            raise TestFailure(f"[{cache_type}] server log contains fatal marker {marker!r}")


def assert_cache_path(cache_type: str, stats: ReuseStats) -> None:
    if stats.reused <= 0:
        raise TestFailure(f"[{cache_type}] kv-share reused no tokens: {stats.line}")
    if cache_type == "f16":
        if stats.aliased <= 0:
            raise TestFailure(f"[f16] expected aliased > 0 in kv-share log: {stats.line}")
    elif cache_type == "q8_0":
        if stats.aliased != 0 or stats.copied <= 0:
            raise TestFailure(
                f"[q8_0] expected aliased=0 and copied > 0 in kv-share log: {stats.line}"
            )
    else:
        raise TestFailure(f"unexpected cache type {cache_type!r}")


def assert_byte_identical(cache_type: str, cold: str, borrow: str) -> None:
    cold_bytes = cold.encode("utf-8")
    borrow_bytes = borrow.encode("utf-8")
    if borrow_bytes != cold_bytes:
        raise TestFailure(
            f"[{cache_type}] borrower output differed from cold baseline\n"
            f"COLD  ({len(cold_bytes)} bytes): {cold!r}\n"
            f"BORROW({len(borrow_bytes)} bytes): {borrow!r}"
        )


def is_port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.25)
        return sock.connect_ex((HOST, port)) == 0


def has_cuda_gpu() -> bool:
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is not None:
        try:
            result = subprocess.run(
                [nvidia_smi, "-L"],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=5.0,
            )
            if result.returncode == 0 and "GPU" in result.stdout:
                return True
        except (OSError, subprocess.TimeoutExpired):
            pass

    gpu_dir = Path("/proc/driver/nvidia/gpus")
    if gpu_dir.is_dir() and any(gpu_dir.iterdir()):
        return True
    return Path("/dev/nvidia0").exists()


def preflight_skip_reason() -> str | None:
    if not SERVER_BIN.is_file():
        return f"server binary not found at {SERVER_BIN}"
    if not os.access(SERVER_BIN, os.X_OK):
        return f"server binary is not executable at {SERVER_BIN}"
    if not MODEL.is_file():
        return f"model not found at {MODEL}"
    if not has_cuda_gpu():
        return "CUDA GPU not detected"
    return None


def run_case(cache_type: str) -> None:
    cold = run_cold(cache_type)
    borrow, logs = run_borrow(cache_type)
    assert_byte_identical(cache_type, cold, borrow)
    stats = extract_reuse_stats(logs, cache_type)
    assert_cache_path(cache_type, stats)
    print(f"[{cache_type}] reused={stats.reused} aliased={stats.aliased} copied={stats.copied}")


def main() -> int:
    skip_reason = preflight_skip_reason()
    if skip_reason is not None:
        print(f"SKIP: {skip_reason}")
        return 0

    try:
        for cache_type in ("f16", "q8_0"):
            run_case(cache_type)
    except TestFailure as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("FAIL: interrupted", file=sys.stderr)
        return 130

    print(
        "PASS: kv-share borrower output matched cold baseline byte-for-byte "
        "for f16 alias path and q8_0 copy path"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
