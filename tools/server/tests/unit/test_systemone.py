import os
import pytest
from utils import *
from test_vision_api import get_img_url

server = ServerPreset.tinylaya()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinylaya()


TEST_STATE = "I was charged twice for my order last week and nobody has replied."

TEST_QUESTIONS = {
    "route": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {"billing": "payments and refunds", "shipping": None, "technical": None},
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this?",
        "criteria": ["can wait", "this week", "today", "right now"],
    },
    "angry": {
        "type": "noul",
        "instructions": "Is the customer angry?",
    },
}


def get_prompt_metrics(server: ServerProcess) -> tuple[int, int]:
    """returns the number of prompt tokens (processed, cached) since the server started"""
    res = server.make_request("GET", "/metrics")
    assert res.status_code == 200
    values = {}
    for line in res.body.splitlines():
        if line.startswith("llamacpp:"):
            name, value = line.split(" ")
            values[name] = int(float(value))
    return values["llamacpp:prompt_tokens_total"], values["llamacpp:prompt_tokens_cached_total"]


@pytest.mark.parametrize("preset", ["tinylaya", "tinyopenjev"])
def test_systemone(preset: str):
    global server
    server = getattr(ServerPreset, preset)()
    server.start()
    res = server.make_request("POST", "/v1/systemone", data={
        "state": TEST_STATE,
        "questions": TEST_QUESTIONS,
    })
    assert res.status_code == 200
    assert res.body["usage"]["input_tokens"] > 0
    assert res.body["usage"]["output_tokens"] == 0

    answers = res.body["answers"]
    assert list(answers.keys()) == ["route", "urgency", "angry"]

    route = answers["route"]
    assert route["type"] == "choice"
    assert list(route["probabilities"].keys()) == ["billing", "shipping", "technical"]
    assert abs(sum(route["probabilities"].values()) - 1.0) < 1e-4
    assert route["choice"] == max(route["probabilities"], key=route["probabilities"].get)
    assert 0.0 <= route["confidence"] <= 1.0

    urgency = answers["urgency"]
    assert urgency["type"] == "score"
    assert urgency["legend"] == {"0": "can wait", "1": "this week", "2": "today", "3": "right now"}
    assert list(urgency["probabilities"].keys()) == ["0", "1", "2", "3"]
    assert abs(sum(urgency["probabilities"].values()) - 1.0) < 1e-4
    assert abs(urgency["score"] - sum(i * p for i, p in enumerate(urgency["probabilities"].values()))) < 1e-4
    assert 0.0 <= urgency["confidence"] <= 1.0

    angry = answers["angry"]
    assert angry["type"] == "noul"
    assert 0.0 <= angry["noul"] <= 1.0


def test_systemone_json_state():
    global server
    server.start()
    questions = {
        "refund": {
            "type": "noul",
            "instructions": "Is a refund requested?",
            "criteria": {"false": "no refund is asked", "true": "a refund is asked"},
        },
    }
    res_obj = server.make_request("POST", "/v1/systemone", data={
        "state": {"ticket": TEST_STATE, "plan": "pro"},
        "questions": questions,
    })
    assert res_obj.status_code == 200
    # an object is given to the model as JSON text
    res_str = server.make_request("POST", "/v1/systemone", data={
        "state": '{"ticket": "' + TEST_STATE + '", "plan": "pro"}',
        "questions": questions,
    })
    assert res_str.status_code == 200
    assert res_obj.body["usage"] == res_str.body["usage"]
    assert abs(res_obj.body["answers"]["refund"]["noul"] - res_str.body["answers"]["refund"]["noul"]) < 1e-4


@pytest.mark.parametrize("data", [
    {"questions": TEST_QUESTIONS},
    {"state": TEST_STATE},
    {"state": TEST_STATE, "questions": {}},
    {"state": TEST_STATE, "questions": {"q": {"type": "unknown", "instructions": "x"}}},
    {"state": TEST_STATE, "questions": {"q": {"type": "noul"}}},
    {"state": TEST_STATE, "questions": {"q": {"type": "choice", "instructions": "x"}}},
    {"state": TEST_STATE, "questions": {"q": {"type": "choice", "instructions": "x", "criteria": {}}}},
    {"state": TEST_STATE, "questions": {"q": {"type": "score", "instructions": "x", "criteria": ["only one"]}}},
])
def test_systemone_invalid_request(data: dict):
    global server
    server.start()
    res = server.make_request("POST", "/v1/systemone", data=data)
    assert res.status_code == 400
    assert "error" in res.body


def test_systemone_shared_prompt():
    global server
    server = ServerPreset.tinyopenjev()
    server.server_metrics = True
    server.start()
    res = server.make_request("POST", "/v1/systemone", data={
        "state": TEST_STATE,
        "questions": TEST_QUESTIONS,
    })
    assert res.status_code == 200

    # the first question evaluates the shared prefix, the 2 others start from it
    n_processed, n_cached = get_prompt_metrics(server)
    assert n_cached > 0
    assert n_cached % 2 == 0
    assert n_processed + n_cached == res.body["usage"]["input_tokens"]

    # with one slot the prompt cannot be shared, the answers must be the same
    server.stop()
    server = ServerPreset.tinyopenjev()
    server.n_slots = 1
    server.start()
    res_single = server.make_request("POST", "/v1/systemone", data={
        "state": TEST_STATE,
        "questions": TEST_QUESTIONS,
    })
    assert res_single.status_code == 200
    assert res_single.body["usage"] == res.body["usage"]
    for qid in ["route", "urgency"]:
        probs_shared = res.body["answers"][qid]["probabilities"]
        probs_single = res_single.body["answers"][qid]["probabilities"]
        for key in probs_shared:
            assert abs(probs_shared[key] - probs_single[key]) < 0.01
    assert abs(res.body["answers"]["angry"]["noul"] - res_single.body["answers"]["angry"]["noul"]) < 0.01


@pytest.mark.parametrize("cache_ram", [None, 100])
def test_systemone_prompt_cache(cache_ram, tmp_path):
    global server
    server.n_slots = 1
    server.cache_ram = cache_ram
    server.debug = True
    server.log_path = str(tmp_path / "server.log")
    server.start()

    # the third request comes back to the state of the first one
    for state in ["Ticket A: " + TEST_STATE, "Ticket B: " + TEST_STATE * 2, "Ticket A: " + TEST_STATE]:
        res = server.make_request("POST", "/v1/systemone", data={
            "state": state,
            "questions": TEST_QUESTIONS,
        })
        assert res.status_code == 200

    server.stop()

    with open(server.log_path) as f:
        log = f.read()
    # the prompt cache is opt-in for decision models, and restores their states once enabled
    assert ("__TEST_TAG_CACHE_IDLE_SLOTS_ENABLED__" in log) == (cache_ram is not None)
    assert ("found better prompt" in log) == (cache_ram is not None)


def test_systemone_prompt_cache_child_slots(tmp_path):
    global server
    server = ServerPreset.tinyopenjev()
    server.n_slots = len(TEST_QUESTIONS)
    server.cache_ram = 100
    server.debug = True
    server.log_path = str(tmp_path / "server.log")
    server.start()

    states = [
        "My parcel shows delivered but it never arrived at my door.",
        "The app crashes every time I open the settings page.",
        "Please close my account and delete all my personal data.",
    ]
    angry = {"angry": TEST_QUESTIONS["angry"]}

    def decide(state, questions):
        res = server.make_request("POST", "/v1/systemone", data={"state": state, "questions": questions})
        assert res.status_code == 200

    # one state per slot, then a request whose child tasks take every slot
    for state in states:
        decide(state, angry)
    decide(TEST_STATE, TEST_QUESTIONS)

    with open(server.log_path) as f:
        n_restored = f.read().count("found better prompt with f_keep = 1.000")

    # every overwritten state comes back whole from the prompt cache
    for state in states:
        decide(state, angry)

    server.stop()

    with open(server.log_path) as f:
        assert f.read().count("found better prompt with f_keep = 1.000") == n_restored + len(states)


def test_systemone_images():
    global server
    server = ServerPreset.tinyopenjev()
    server.start()
    image = get_img_url("IMG_BASE64_URI_0")

    res_text = server.make_request("POST", "/v1/systemone", data={
        "state": TEST_STATE,
        "questions": TEST_QUESTIONS,
    })
    assert res_text.status_code == 200

    res = server.make_request("POST", "/v1/systemone", data={
        "state": TEST_STATE,
        "questions": TEST_QUESTIONS,
        "images": [image],
    })
    assert res.status_code == 200
    assert list(res.body["answers"].keys()) == ["route", "urgency", "angry"]
    assert res.body["usage"]["input_tokens"] > res_text.body["usage"]["input_tokens"]

    # same image, given as a part of a chat message
    res_part = server.make_request("POST", "/v1/systemone", data={
        "state": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": image}},
            {"type": "text", "text": TEST_STATE},
        ]}],
        "questions": TEST_QUESTIONS,
    })
    assert res_part.status_code == 200
    assert res_part.body["usage"]["input_tokens"] > res_text.body["usage"]["input_tokens"]

    res = server.make_request("POST", "/v1/systemone", data={
        "state": TEST_STATE,
        "questions": TEST_QUESTIONS,
        "images": [image] * 9,
    })
    assert res.status_code == 400

    res = server.make_request("POST", "/v1/systemone", data={
        "state": TEST_STATE,
        "questions": TEST_QUESTIONS,
        "images": ["https://example.com/image.png"],
    })
    assert res.status_code == 400


def test_systemone_images_not_supported():
    global server
    server.start()
    res = server.make_request("POST", "/v1/systemone", data={
        "state": TEST_STATE,
        "questions": TEST_QUESTIONS,
        "images": [get_img_url("IMG_BASE64_URI_0")],
    })
    assert res.status_code == 501


# there is no tiny test model of the baselabs-decision type, set BASELABS_ELEPHANT_LOCAL_MODEL to a GGUF of baselabs/elephant-1
ELEPHANT_MODEL = os.environ.get("BASELABS_ELEPHANT_LOCAL_MODEL")

BLOCKS_STATE = {"email": "Hi, the login page shows error 500 since this morning. Our whole team is blocked. -- Ana, CTO"}
BLOCKS_QUESTIONS = {
    "severity": {"type": "score", "instructions": "How severe is the issue?", "criteria": ["minor", "moderate", "major", "critical"]},
    "affected": {"type": "score", "instructions": "How many people are affected?", "criteria": ["one", "a few", "a team", "everyone"]},
}
# probabilities of OneJev qev with the block mask (bfloat16), the runtime of the model repo
# a causal mask gives [0.011, 0.069, 0.255, 0.664] and [0.002, 0.009, 0.928, 0.061], a bidirectional one [0.002, 0.009, 0.093, 0.896] and [0.002, 0.005, 0.864, 0.129]
BLOCKS_REFERENCE = {
    "severity": [0.0065, 0.0349, 0.2044, 0.7542],
    "affected": [0.0027, 0.0084, 0.7446, 0.2444],
}


def elephant_server(n_slots: int, n_batch: int = 1024) -> ServerProcess:
    server = ServerProcess()
    server.model_hf_repo = None
    server.model_hf_file = None
    server.model_file = ELEPHANT_MODEL
    server.n_slots = n_slots
    server.n_ctx = 2048 * n_slots
    server.n_batch = n_batch
    server.n_ubatch = n_batch
    return server


@pytest.mark.skipif(ELEPHANT_MODEL is None, reason="BASELABS_ELEPHANT_LOCAL_MODEL is not set")
@pytest.mark.parametrize("n_slots", [1, 2])
def test_systemone_attention_blocks(n_slots: int):
    # with 2 slots the questions share the evaluation of the state block, with 1 slot each question evaluates it
    global server
    server = elephant_server(n_slots)
    server.start()
    res = server.make_request("POST", "/v1/systemone", data={"state": BLOCKS_STATE, "questions": BLOCKS_QUESTIONS})
    assert res.status_code == 200
    for qid, ref in BLOCKS_REFERENCE.items():
        probs = res.body["answers"][qid]["probabilities"]
        got = [probs[str(i)] for i in range(len(ref))]
        assert max(abs(a - b) for a, b in zip(got, ref)) < 0.05, f"{qid}: {got} instead of {ref}"


@pytest.mark.skipif(ELEPHANT_MODEL is None, reason="BASELABS_ELEPHANT_LOCAL_MODEL is not set")
def test_systemone_attention_block_too_large():
    # each block is evaluated in one ubatch, the prompt is not split inside a block
    global server
    server = elephant_server(1, n_batch=128)
    server.start()
    res = server.make_request("POST", "/v1/systemone", data={"state": "word " * 200, "questions": BLOCKS_QUESTIONS})
    assert res.status_code == 400
    assert "too large" in res.body["error"]["message"]
