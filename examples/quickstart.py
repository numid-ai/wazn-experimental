"""Ask a running server a question.

Start a server first, in another terminal:

    wazn-experimental serve          # wazn-2b-v0.1; --model for another checkpoint

then:

    python examples/quickstart.py
"""

from wazn_experimental import Client, Instruction, Label, Request

request = Request(
    rules=(
        "Returns handles exchanges and refunds for delivered items. "
        "Shipping handles anything not yet delivered. Billing handles charges."
    ),
    context="My running shoes arrived in the wrong size. Can I swap them for a 10?",
    instructions=[
        Instruction(
            "Which team should handle this?",
            labels=[
                Label("returns", "Exchanges, refunds, wrong or damaged items"),
                Label("shipping", "Delivery status, delays, lost packages",
                      examples=["Tracking has said 'in transit' for two weeks"]),
                Label("billing", "Charges, invoices, payment problems",
                      examples=["I was charged twice for the same order",
                                "My invoice shows the wrong amount"]),
            ],
            name="department",
        ),
        Instruction(
            "What is the customer's tone?",
            labels=["negative", "neutral", "positive"],
            name="tone",
        ),
    ],
)

with Client("http://127.0.0.1:8000") as client:
    client.wait_until_ready()
    response = client.predict(request)

for name, answer in response.answers.items():
    print(f"{name}: {answer.choice} ({answer.confidence:.2f})")
    print(f"  top 3: {answer.top(3)}")
    if answer.none_probability is not None:
        print(f"  P(no label applies) = {answer.none_probability:.2f}")
for w in response.warnings:
    print("warning:", w)
