import time

import numpy as np

from tests.helpers import IDENTITY
from xr_commander.devices import HandSample, XrFrame, fresh_sample
from xr_commander.frames import Pose


class FakeSource:
    def __init__(self, frame=None):
        self.frame = frame

    def latest(self):
        return self.frame


def sample(squeezing=True):
    return HandSample(
        pose=Pose(np.zeros(3), IDENTITY), squeezing=squeezing, trigger=0.0
    )


def frame_at(age_s, hands):
    return XrFrame(received_monotonic_s=time.monotonic() - age_s, hands=hands)


def test_no_frame_yet_reads_as_no_sample():
    assert fresh_sample(FakeSource(), "left", 0.25) is None


def test_a_recent_frame_yields_its_hand():
    source = FakeSource(frame_at(0.0, {"left": sample()}))
    assert fresh_sample(source, "left", 0.25) is not None


def test_a_hand_missing_from_a_recent_frame_reads_as_no_sample():
    source = FakeSource(frame_at(0.0, {"left": sample()}))
    assert fresh_sample(source, "right", 0.25) is None


def test_a_stale_frame_is_discarded_rather_than_held():
    # A frozen headset is still reporting whatever button it last held. Treating
    # that as current would leave the robot engaged to an absent operator, so
    # age, not content, decides.
    source = FakeSource(frame_at(1.0, {"left": sample(squeezing=True)}))
    assert fresh_sample(source, "left", 0.25) is None
