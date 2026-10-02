"""Use the model directly, without a server (e.g. in a notebook).

    pip install "wazn-experimental[server]"
    python examples/in_process.py                      # wazn-2b-v0.1
    python examples/in_process.py <hub id or checkpoint dir>
"""

import sys
from pathlib import Path

from wazn_experimental import Request, Wazn

model = Wazn.load(*sys.argv[1:2])
print(model.info())

for path in sorted((Path(__file__).parent / "requests").glob("*.json")):
    request = Request.from_file(path)
    options = {"group_size": 10, "top_k": 2} if request.num_labels > 30 else {}
    response = model.predict(request, **options)
    print(f"\n{path.name}  ({response.prediction_seconds}s)")
    for name, answer in response.answers.items():
        mark = "" if answer.correct is None else ("  ok" if answer.correct else "  WRONG")
        print(f"  {name}: {answer.choice} ({answer.confidence:.2f}){mark}")
