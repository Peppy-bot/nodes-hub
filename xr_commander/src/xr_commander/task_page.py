"""The label new episodes record under, and the page that sets it.

A headset has no keyboard reachable from inside a session, so the task is
typed on a page served from the origin the headset already trusts: by the
operator before entering VR, or by a second person on a laptop while the
session runs. The page is the only source of the label. A session starts with
nothing set, and whatever is recorded first goes down as `UNNAMED_TASK` rather
than being refused: a lost demonstration cannot be recovered, while a uniform
placeholder is one string to correct afterwards. Mounted only when a recorder
is bound; teleoperation without one serves no page.

The label is read when a goal fires, so a change lands on the next episode and
can never rename the episode being recorded.
"""

from __future__ import annotations

import html
import threading
import urllib.parse

from fastapi import APIRouter, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse

from xr_commander.bus import log

PATH = "/task"

# A task label is one instruction, not a paragraph: room for a sentence, and
# short enough to stay readable on the headset panel.
MAX_TASK_CHARS = 120

# What an episode records under before the operator names one. Deliberately
# not a plausible task: it should read as missing to whoever sorts the dataset
# out later, not as a description of what was demonstrated.
UNNAMED_TASK = "unnamed teleop task"

_FIELD = "task"
_FORM_MEDIA_TYPE = "application/x-www-form-urlencoded"
# One short form field; anything larger is not a submission from this page.
_MAX_BODY_BYTES = 4096
# The browser reissues the GET after a successful POST, so a reload cannot
# resubmit the form.
_SEE_OTHER = 303


def parse_task(value) -> str:
    """The dataset task label, whitespace normalized onto one line.

    Written into every recorded frame and read back as a policy's language
    instruction, so a blank, over-long, or control-character label is refused
    here rather than written into a dataset.
    """
    if not isinstance(value, str):
        raise ValueError(f"task must be text, got {value!r}")
    label = " ".join(value.split())
    if not label:
        raise ValueError("task must not be blank")
    if len(label) > MAX_TASK_CHARS:
        raise ValueError(
            f"task must be at most {MAX_TASK_CHARS} characters, got {len(label)}"
        )
    if not label.isprintable():
        raise ValueError("task must not contain control characters")
    return label


class TaskLabel:
    """What the operator has named the task, None until they name it.

    Shared between the web thread and the event loop: the server runs its own
    loop on its own thread, so unlike the panel's recorder state this is not
    event-loop-confined and takes a lock.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._label: str | None = None

    def current(self) -> str | None:
        """What the operator set, None while they have not. Callers that
        display it distinguish the two; callers that record use
        `for_episode`."""
        with self._lock:
            return self._label

    def for_episode(self) -> str:
        """The label to record under, standing in for an unnamed session."""
        return self.current() or UNNAMED_TASK

    def set(self, raw: str) -> str:
        """Adopt `raw` once parsed, returning what was adopted. A refusal
        raises ValueError and leaves the previous label in place, so a bad
        edit cannot unset a label that episodes are already recording under."""
        label = parse_task(raw)
        with self._lock:
            self._label = label
        return label


async def read_capped_body(request: Request) -> bytes:
    """The request body, refused past the cap without buffering the rest.

    Read in chunks rather than through `request.body()`, which would hold the
    whole submission in memory before any check could reject it.
    """
    chunks: list[bytes] = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > _MAX_BODY_BYTES:
            raise ValueError(f"form is larger than {_MAX_BODY_BYTES} bytes")
        chunks.append(chunk)
    return b"".join(chunks)


def form_field(content_type: str, body: bytes) -> str:
    """The raw task field a submitted form carries.

    Structure only: the body's size belongs to whoever read it, and what makes
    a label acceptable belongs to `parse_task`. Decoded here rather than
    through Starlette's form parser, which requires python-multipart even for
    urlencoded bodies.
    """
    media_type = content_type.split(";")[0].strip().lower()
    if media_type != _FORM_MEDIA_TYPE:
        raise ValueError(f"form must be {_FORM_MEDIA_TYPE}, got {media_type or 'none'}")
    try:
        fields = urllib.parse.parse_qs(body.decode("utf-8"), keep_blank_values=True)
    except UnicodeDecodeError:
        raise ValueError("form is not valid UTF-8") from None
    values = fields.get(_FIELD, [])
    if len(values) != 1:
        raise ValueError(f"form must carry one {_FIELD} field, got {len(values)}")
    return values[0]


_STYLE = """
:root { color-scheme: dark; }
body { background: #181818; color: #eee; font: 20px/1.5 system-ui, sans-serif;
       margin: 0 auto; padding: 2rem; max-width: 34rem; }
h1 { font-size: 1.4rem; margin: 0 0 1.5rem; }
.current { background: #222; border-left: 4px solid #6cd97e; padding: .75rem 1rem;
           margin: 0 0 1.5rem; word-break: break-word; }
.unset { border-left-color: #d9c46c; }
.error { background: #2a1c1c; border-left: 4px solid #e06c6c; padding: .75rem 1rem;
         margin: 0 0 1.5rem; }
input { width: 100%; box-sizing: border-box; font: inherit; padding: .75rem;
        background: #222; color: #eee; border: 1px solid #555; border-radius: 4px; }
button { font: inherit; margin-top: 1rem; padding: .75rem 1.5rem; border: 0;
         border-radius: 4px; background: #3a6ea5; color: #fff; }
.hint { color: #999; font-size: .85rem; margin-top: 2rem; }
"""


def render_page(
    current: str | None, *, pending: str | None = None, error: str = ""
) -> str:
    """The page as served: the live label, a field to replace it, and the
    rejection to fix when there is one. `pending` keeps refused text in the
    field so it can be corrected rather than retyped in a headset."""
    field = (current or "") if pending is None else pending
    problem = f'<p class="error">Refused: {html.escape(error)}</p>' if error else ""
    if current is None:
        state = (
            '<p class="current unset">No task set: episodes record as '
            f"<strong>{html.escape(UNNAMED_TASK)}</strong> until one is.</p>"
        )
    else:
        state = (
            '<p class="current">Recording as: '
            f"<strong>{html.escape(current)}</strong></p>"
        )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>xr_commander task</title>
<style>{_STYLE}</style>
</head>
<body>
<h1>Episode task</h1>
{state}
{problem}
<form method="post" action="{PATH}">
<label for="{_FIELD}">What the operator is demonstrating</label>
<input id="{_FIELD}" name="{_FIELD}" value="{html.escape(field, quote=True)}"
       maxlength="{MAX_TASK_CHARS}" autocomplete="off" autofocus>
<button type="submit">Set task</button>
</form>
<p class="hint">Applies to the next episode. An episode already recording keeps
the label it started with.</p>
</body>
</html>
"""


def _refused(label: TaskLabel, pending: str | None, error: Exception) -> HTMLResponse:
    log(f"task refused: {error}")
    return HTMLResponse(
        render_page(label.current(), pending=pending, error=str(error)),
        status_code=400,
    )


def build_router(label: TaskLabel) -> APIRouter:
    """The page's two routes over one label."""
    router = APIRouter()

    @router.get(PATH, response_class=HTMLResponse)
    async def show_task() -> HTMLResponse:
        return HTMLResponse(render_page(label.current()))

    @router.post(PATH)
    async def set_task(request: Request) -> Response:
        try:
            submitted = form_field(
                request.headers.get("content-type", ""),
                await read_capped_body(request),
            )
        except ValueError as e:
            return _refused(label, None, e)
        try:
            adopted = label.set(submitted)
        except ValueError as e:
            return _refused(label, submitted, e)
        log(f"episode task set to {adopted!r}")
        return RedirectResponse(PATH, status_code=_SEE_OTHER)

    return router
