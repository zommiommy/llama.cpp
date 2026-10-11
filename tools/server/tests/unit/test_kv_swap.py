import os
import tempfile
import threading
import pytest
from utils import *

server = ServerPreset.tinyllama2()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.n_slots = 2
    server.n_ctx = 4096
    server.n_predict = -1
    server.temperature = 0.0
    server.kv_swap_max_active = 1
    fd, server.log_path = tempfile.mkstemp(suffix='.log')
    os.close(fd)
    yield


def stream_completion(prompt: str, n_predict: int, priority: int, first_chunk: threading.Event | None = None) -> dict:
    content = ""
    final = {}
    for chunk in server.make_stream_request("POST", "/completion", data={
        "prompt": prompt,
        "n_predict": n_predict,
        "priority": priority,
        "ignore_eos": True,
        "stream": True,
    }):
        content += chunk["content"]
        if first_chunk is not None:
            first_chunk.set()
        if chunk["stop"]:
            final = chunk
    return {"content": content, "final": final}


# a request parked for a higher-priority one keeps its own n_predict when it resumes
def test_resumed_request_keeps_n_predict():
    global server
    server.start()

    first_chunk = threading.Event()
    results = {}

    def low():
        results["low"] = stream_completion("Once upon a time", 512, 0, first_chunk)

    t = threading.Thread(target=low)
    t.start()
    assert first_chunk.wait(timeout=30), "the low-priority request never started streaming"
    results["high"] = stream_completion("The little girl", 32, 5)
    t.join(timeout=120)
    assert not t.is_alive()

    with open(server.log_path) as f:
        log = f.read()
    assert "kv-swap: PARKED" in log
    assert "kv-swap: UNPARKED" in log

    assert results["high"]["final"]["tokens_predicted"] == 32
    # before the fix the resumed request ran on until its slot context was full
    assert results["low"]["final"]["tokens_predicted"] == 512
    assert results["low"]["final"]["stop_type"] == "limit"
