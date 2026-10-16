"""The plain forms the pages take, as `forms` reads them."""

import pytest
from starlette.requests import ClientDisconnect

from tests.helpers import FORM
from xr_commander.forms import form_field, read_capped_body


def test_a_form_carries_its_task_decoded():
    assert form_field(FORM["content-type"], b"task=pick+up+the+red+block", "task") == (
        "pick up the red block"
    )
    charset = "application/x-www-form-urlencoded; charset=utf-8"
    assert form_field(charset, b"task=a%20b", "task") == "a b"


@pytest.mark.parametrize(
    ("content_type", "body", "reason"),
    [
        ("application/json", b'{"task": "x"}', "must be application"),
        (FORM["content-type"], b"other=x", "one task field"),
        (FORM["content-type"], b"task=a&task=b", "one task field"),
        (FORM["content-type"], b"task=\xff\xfe", "not valid UTF-8"),
        ("", b"task=x", "must be application"),
    ],
)
def test_a_malformed_submission_is_refused_structurally(content_type, body, reason):
    with pytest.raises(ValueError, match=reason):
        form_field(content_type, body, "task")


async def test_a_submission_cut_off_by_its_client_is_refused():
    class CutOff:
        async def stream(self):
            yield b"task=a"
            raise ClientDisconnect

    with pytest.raises(ValueError, match="before the form was complete"):
        await read_capped_body(CutOff())
