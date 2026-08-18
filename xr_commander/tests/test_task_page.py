import pytest
from fastapi import FastAPI

from tests.helpers import asgi_request
from xr_commander import task_page
from xr_commander.task_page import (
    MAX_TASK_CHARS,
    PATH,
    UNNAMED_TASK,
    TaskLabel,
    build_router,
    form_field,
    parse_task,
    render_page,
)

FORM = {"content-type": "application/x-www-form-urlencoded"}


def app_with(label: TaskLabel) -> FastAPI:
    app = FastAPI()
    app.include_router(build_router(label))
    return app


def post(app, body: bytes, headers=None):
    return asgi_request(app, "POST", PATH, body=body, headers=headers or FORM)


def labelled(text: str) -> TaskLabel:
    label = TaskLabel()
    label.set(text)
    return label


def test_a_session_starts_with_no_label():
    # Nothing seeds it: the operator's page is the only source.
    assert TaskLabel().current() is None


def test_an_unnamed_session_still_records_under_a_placeholder():
    # A lost demonstration cannot be recovered; a uniform placeholder is one
    # string to correct afterwards.
    assert TaskLabel().for_episode() == UNNAMED_TASK


def test_a_named_task_replaces_the_placeholder():
    assert labelled("fold the towel").for_episode() == "fold the towel"


def test_the_placeholder_does_not_read_as_a_demonstrated_task():
    # It is sorted through later as missing data, so it must not look like
    # something an operator chose to demonstrate.
    assert "unnamed" in UNNAMED_TASK


def test_a_label_is_normalized_onto_one_line():
    assert parse_task("  pick up\tthe  red block\n") == "pick up the red block"


@pytest.mark.parametrize(
    ("bad", "reason"),
    [
        ("", "blank"),
        ("   \n ", "blank"),
        ("x" * (MAX_TASK_CHARS + 1), "at most"),
        ("pick up the\x07block", "control characters"),
        (7, "must be text"),
    ],
)
def test_a_label_a_dataset_should_not_carry_is_refused(bad, reason):
    with pytest.raises(ValueError, match=reason):
        parse_task(bad)


def test_a_label_at_the_limit_is_accepted():
    longest = "x" * MAX_TASK_CHARS
    assert parse_task(longest) == longest


def test_setting_adopts_the_parsed_label():
    label = TaskLabel()
    assert label.set(" stack\tthe blocks ") == "stack the blocks"
    assert label.current() == "stack the blocks"


@pytest.mark.parametrize("bad", ["", "   ", "x" * (MAX_TASK_CHARS + 1), "a\x00b"])
def test_a_refused_label_leaves_the_previous_one_recording(bad):
    # A bad edit must not unset a label episodes are already recording under.
    label = labelled("fold the towel")
    with pytest.raises(ValueError):
        label.set(bad)
    assert label.current() == "fold the towel"


def test_a_form_carries_its_task_decoded():
    assert form_field(FORM["content-type"], b"task=pick+up+the+red+block") == (
        "pick up the red block"
    )
    charset = "application/x-www-form-urlencoded; charset=utf-8"
    assert form_field(charset, b"task=a%20b") == "a b"


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
        form_field(content_type, body)


def test_a_blank_field_is_the_parsers_refusal_not_the_forms():
    # Structure is fine, so form_field hands it on; what makes a label
    # acceptable is the parser's call alone.
    assert form_field(FORM["content-type"], b"task=") == ""
    with pytest.raises(ValueError, match="blank"):
        labelled("fold the towel").set("")


def test_the_page_shows_the_live_label_and_escapes_it():
    # Operator text, rendered into both the paragraph and the field's
    # attribute; neither may close its context.
    page = render_page('fold <b>"the"</b> towel')
    assert "&lt;b&gt;" in page
    assert "&quot;the&quot;" in page
    assert "<b>" not in page


def test_the_page_keeps_refused_text_in_the_field_to_be_corrected():
    page = render_page("fold the towel", pending="x" * 200, error="too long")
    assert "too long" in page
    assert "x" * 200 in page
    # The live label is still what the recorder would use.
    assert "fold the towel" in page


def test_the_page_names_what_an_unset_session_records_as():
    page = render_page(None)
    assert "No task set" in page
    assert UNNAMED_TASK in page
    assert 'value=""' in page


def test_get_serves_the_current_label():
    label = labelled("fold the towel")
    response = asgi_request(app_with(label), "GET", PATH)
    assert response.status == 200
    assert "fold the towel" in response.body


def test_the_first_post_of_a_session_sets_the_label_from_nothing():
    label = TaskLabel()
    app = app_with(label)
    assert "No task set" in asgi_request(app, "GET", PATH).body

    assert post(app, b"task=stack+the+blocks").status == 303
    assert label.current() == "stack the blocks"
    assert "stack the blocks" in asgi_request(app, "GET", PATH).body


def test_a_refused_first_edit_leaves_the_session_unset():
    label = TaskLabel()
    response = post(app_with(label), b"task=%20%20")
    assert response.status == 400
    assert label.current() is None
    assert "No task set" in response.body


def test_a_good_post_adopts_the_label_and_redirects_to_the_page():
    label = labelled("fold the towel")
    response = post(app_with(label), b"task=stack+the+blocks")
    # See Other, so reloading the page cannot resubmit the form.
    assert response.status == 303
    assert response.headers["location"] == PATH
    assert label.current() == "stack the blocks"


@pytest.mark.parametrize(
    ("body", "headers"),
    [(b"task=", FORM), (b'{"task": "x"}', {"content-type": "application/json"})],
)
def test_a_refused_post_answers_400_and_changes_nothing(body, headers):
    label = labelled("fold the towel")
    response = post(app_with(label), body, headers)
    assert response.status == 400
    assert label.current() == "fold the towel"
    assert "Refused" in response.body


@pytest.mark.parametrize(
    "body",
    [b"task=" + b"x" * 5000, [b"task=", b"x" * 3000, b"x" * 3000]],
    ids=["one delivery", "streamed"],
)
def test_an_oversized_submission_is_refused_before_it_is_all_read(body):
    label = labelled("fold the towel")
    response = post(app_with(label), body)
    assert response.status == 400
    assert "larger than" in response.body
    assert label.current() == "fold the towel"


def test_the_page_posts_back_to_the_path_it_is_served_from():
    assert f'action="{PATH}"' in render_page("fold the towel")
    assert f'maxlength="{MAX_TASK_CHARS}"' in render_page("fold the towel")


def test_the_module_serves_one_path():
    assert task_page.PATH == "/task"
