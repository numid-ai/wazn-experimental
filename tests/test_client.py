"""The client against a mocked transport, and response parsing. No torch."""

import json

import httpx
import pytest

from wazn_experimental import Client, Instruction, Request, RequestError, Response, ServerError

REQUEST = Request(
    "My shoes arrived in the wrong size.",
    [Instruction("Which team?", labels=["returns", "billing"], name="team", true_label="returns")],
)

PAYLOAD = {
    "model": "wazn-test",
    "answers": {"team": {
        "type": "choice", "choice": "returns", "confidence": 0.8,
        "probabilities": {"returns": 0.8, "billing": 0.2},
        "none_probability": 0.1, "is_none": False,
        "true_label": "returns", "correct": True, "true_label_probability": 0.8,
    }},
    "usage": {"input_tokens": 38},
    "evaluation": {"n_labelled": 1, "n_correct": 1, "accuracy": 1.0, "incorrect": [],
                   "mean_true_label_probability": 0.8},
    "prediction_seconds": 0.12,
}


def mock_client(handler) -> Client:
    return Client("http://wazn.test", transport=httpx.MockTransport(handler))


def test_predict_sends_the_wire_format_and_parses_the_response():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["path"] = req.url.path
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json=PAYLOAD)

    response = mock_client(handler).predict(REQUEST, group_size=4, top_k=2, seed=0)
    assert seen["path"] == "/predict"
    assert seen["body"]["request"] == REQUEST.to_dict()
    assert seen["body"]["options"] == {"group_size": 4, "top_k": 2, "seed": 0, "usage_detail": False}
    assert response.answer.choice == "returns"
    assert response["team"].none_probability == 0.1
    assert response.answer.correct is True
    assert response.to_dict() == PAYLOAD  # parsing loses nothing


def test_requests_are_validated_before_sending():
    def handler(req):
        raise AssertionError("nothing should be sent")

    client = mock_client(handler)
    with pytest.raises(RequestError):
        client.predict({"state": "s", "questions": {}})
    with pytest.raises(RequestError):
        client.predict(REQUEST, group_size=2, top_k=2)


def test_server_validation_errors_become_request_errors():
    client = mock_client(lambda req: httpx.Response(422, json={"detail": "too many labels"}))
    with pytest.raises(RequestError, match="too many labels"):
        client.predict(REQUEST)


def test_server_failures_become_server_errors():
    client = mock_client(lambda req: httpx.Response(503, json={"detail": "out of memory"}))
    with pytest.raises(ServerError, match="out of memory") as e:
        client.predict(REQUEST)
    assert e.value.status == 503


def test_an_unreachable_server_says_how_to_start_one():
    def handler(req):
        raise httpx.ConnectError("refused")

    with pytest.raises(ServerError, match="wazn-experimental serve"):
        mock_client(handler).health()


def test_answer_helpers():
    r = Response.from_dict(PAYLOAD)
    assert r.answer.top(1) == [("returns", 0.8)]
    assert r.evaluation()["accuracy"] == 1.0
    team = PAYLOAD["answers"]["team"]
    multi = Response.from_dict({**PAYLOAD, "answers": {"team": team, "b": team}})
    with pytest.raises(ValueError, match="index it by name"):
        _ = multi.answer


def test_tournament_elimination_round():
    answer = Response.from_dict({**PAYLOAD, "answers": {"team": {
        **PAYLOAD["answers"]["team"],
        "choice": "billing", "probabilities": {"billing": 0.6, "other": 0.4},
        "rounds": [
            {"groups": [{"returns": 0.3, "billing": 0.7}, {"other": 0.9, "x": 0.1}]},
            {"groups": [{"billing": 0.6, "other": 0.4}]},
        ],
    }}}).answer
    assert answer.correct is False
    assert answer.true_label_probability == 0.0
    assert answer.eliminated_round == 1
