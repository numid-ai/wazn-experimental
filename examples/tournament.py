"""Choose among many labels by tournament, next to an ordinary question.

The intent question has 151 labels. A tournament scores them in groups of
`group_size`, advances each group's `top_k`, and repeats until one group is
left, so the model never compares more than `group_size` labels at a time.
The tone question has 3 labels and is answered as usual. Each instruction
picks its own mode; the context is still read once per round.

    python examples/tournament.py
"""

from pathlib import Path

from wazn_experimental import Client, Instruction, Request, Tournament

intents = Request.from_file(Path(__file__).parent / "requests" / "clinc_k151.json")
intent = intents.instructions[0]  # already carries {"group_size": 10, "top_k": 2}

request = Request(
    context=intents.context,
    instructions=[
        Instruction(intent.text, intent.labels, name="intent", true_label=intent.true_label,
                    tournament=Tournament(group_size=10, top_k=2, seed=0)),
        Instruction("What is the user's tone?", ["negative", "neutral", "positive"], name="tone"),
    ],
)

with Client() as client:
    response = client.predict(request)

answer = response["intent"]
print(f"intent: {answer.choice} ({answer.confidence:.2f}), "
      f"expected: {answer.true_label}, correct: {answer.correct}")
for r, rnd in enumerate(answer.rounds):
    print(f"  round {r}: {len(rnd['groups'])} group(s)")
if answer.eliminated_round is not None:
    print(f"  the expected label was knocked out in round {answer.eliminated_round}")
print(f"tone: {response['tone'].choice} (no tournament, rounds={response['tone'].rounds})")
