"""Drive a local STS2 instance with OpenJev Choice on Gemma 4."""

import sys

from openjevpro.client import OpenJevProClient

from sts2jev.http import Sts2Client
from sts2jev.loop import StopPlay, run_loop
from sts2jev.typesafe import TypeSafeJevClient, jev_key_from_argv

JEV_BASE_URL = "http://127.0.0.1:1234/v1"
JEV_MODEL = "gemma-4-e4b-it"
STS2_BASE_URL = "http://127.0.0.1:8080"


def main() -> None:
    game = Sts2Client(STS2_BASE_URL)
    jev_key = jev_key_from_argv(sys.argv[1:])
    if jev_key:
        print("jev remote https://api.typesafe.ai model=jev-latest")
        jev = TypeSafeJevClient(api_key=jev_key)
    else:
        jev = OpenJevProClient(
            base_url=JEV_BASE_URL,
            model=JEV_MODEL,
            temperature_scaling=1.30,
            abstain_threshold=0.45,
        )
    try:
        run_loop(game, jev)
    except StopPlay as stopped:
        print("stopped:", stopped.reason)


if __name__ == "__main__":
    main()
