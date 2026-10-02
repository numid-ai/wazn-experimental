"""The HTTP API on a tiny model, through FastAPI's test client and through
the library's own `Client`."""

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from wazn_experimental import Client, Instruction, Request, RequestError  # noqa: E402
from wazn_experimental.server.app import Limits, create_app  # noqa: E402

REQUEST = Request(
    "My shoes arrived in the wrong size.",
    [Instruction("Which team?", labels={"returns": "Exchanges", "billing": "Charges"},
                 name="team", true_label="returns")],
)


@pytest.fixture(scope="module")
def http(wazn):
    return TestClient(create_app(wazn, Limits(max_labels=40)))


@pytest.fixture(scope="module")
def client(http):
    c = Client("http://testserver")
    c._http = http  # TestClient is an httpx.Client
    return c


def test_predict_matches_in_process(wazn, client):
    remote = client.predict(REQUEST)
    local = wazn.predict(REQUEST)
    assert remote.answer.probabilities == pytest.approx(local.answer.probabilities, abs=1e-6)
    assert remote.answer.none_probability == pytest.approx(local.answer.none_probability, abs=1e-6)
    assert remote.answer.correct == local.answer.correct
    assert remote.prediction_seconds is not None


def test_usage_detail_is_opt_in(client):
    assert set(client.predict(REQUEST).to_dict()["usage"]) == {"input_tokens"}
    r = client.predict(REQUEST, usage_detail=True)
    assert r.usage_detail and r.usage.unshared_equivalent_tokens > 0


def test_label_limit_points_to_tournaments(client):
    big = Request("ctx", [Instruction("Pick", [f"l{i}" for i in range(41)], name="q")])
    with pytest.raises(RequestError, match="tournament"):
        client.predict(big)
    assert client.predict(big, group_size=10, top_k=2).answer.choice.startswith("l")


@pytest.mark.parametrize("body", [
    {"request": {"state": "s", "questions": {}}},
    {"request": {"state": "s", "questions": {"q": {"instructions": "Q?", "criteria": ["a"]}}}},
    {"request": {"state": "s", "questions": {"q": {"instructions": "Q?", "criteria": ["a", "b"],
                                                   "bogus": 1}}}},
    {"request": REQUEST.to_dict(), "options": {"group_size": 2, "top_k": 2}},
    {"request": {"state": "s", "questions": {"q": {  # the old question-level examples
        "instructions": "Q?", "criteria": ["a", "b"], "examples": [{"input": "x", "label": "a"}]}}}},
])
def test_bad_bodies_are_422(http, body):
    r = http.post("/predict", json=body)
    assert r.status_code == 422, r.text


def test_info_and_health(http):
    info = http.get("/info").json()
    assert info["model"] == "tiny"
    assert info["none_gate"] is True
    assert info["limits"]["max_labels"] == 40
    health = http.get("/health").json()
    assert health["status"] == "ok" and health["requests"] >= 0


@pytest.mark.parametrize("method, path", [
    ("get", "/docs"), ("get", "/redoc"), ("get", "/openapi.json"), ("post", "/predict_batch"),
    ("post", "/v1/predict"), ("get", "/v1/info"),
])
def test_only_the_documented_routes_exist(http, method, path):
    assert getattr(http, method)(path).status_code == 404
