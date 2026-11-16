import sys
from pathlib import Path

from calibrator import Calibrator

session = Path(sys.argv[1])
reference = float(sys.argv[2])
hand = Calibrator("hand")
for video in sorted(session.glob("hand_*")):
    hand.reset()
    f = hand.run(str(video))
    print(f"{video.name}: f = {f:.0f}, error {100 * (f - reference) / reference:+.1f}%")
