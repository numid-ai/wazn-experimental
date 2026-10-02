"""Choose among many labels by tournament.

With 151 labels in one request the model would compare all of them at once.
A tournament scores them in groups of `group_size`, advances each group's
`top_k`, and repeats until one group is left, so the model never compares
more than `group_size` labels at a time.

    python examples/tournament.py
"""

from pathlib import Path

from wazn_experimental import Client, Request

request = Request.from_file(Path(__file__).parent / "requests" / "clinc_k151.json")

with Client() as client:
    response = client.predict(request, group_size=10, top_k=2)

answer = response.answer
print(f"choice: {answer.choice} ({answer.confidence:.2f}), "
      f"expected: {answer.true_label}, correct: {answer.correct}")
for r, rnd in enumerate(answer.rounds):
    print(f"round {r}: {len(rnd['groups'])} group(s)")
if answer.eliminated_round is not None:
    print(f"the expected label was knocked out in round {answer.eliminated_round}")
