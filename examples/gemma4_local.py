"""OpenJev Choice/Noul against the local Gemma 4 server on port 1234."""

from enum import StrEnum

from openjevpro.client import OpenJevProClient


class TicketRoute(StrEnum):
    BILLING = "billing"
    TECH_SUPPORT = "tech_support"
    SECURITY = "security"
    ESCALATE = "human_review"


def main() -> None:
    client = OpenJevProClient(
        base_url="http://127.0.0.1:1234/v1",
        model="gemma-4-e4b-it",
        temperature_scaling=1.30,
        abstain_threshold=0.45,
    )

    decision = client.decide_choice(
        state={
            "ticket_text": "I noticed an billing issue."
        },
        candidates=TicketRoute,
        criteria="Classify the incoming support ticket into the correct handling department.",
    )
    print("choice", decision.value)
    print("confidence", f"{decision.confidence:.2%}")
    print("probabilities", {k: round(v, 4) for k, v in decision.probabilities.items()})
    print("abstained", decision.abstained)

    noul = client.decide_noul(
        state={"ticket_text": "I noticed an unauthorized login attempt from a known IP address."},
        assertion="This ticket is definitely a security incident.",
    )
    print("noul", noul.value, f"P(true)={noul.probability_true:.2%}")


if __name__ == "__main__":
    main()
