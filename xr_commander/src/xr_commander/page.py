"""The page the headset opens: teleop_xr's dashboard, trimmed to what this
node honors.

teleop_xr's page carries a VR mode, speed sliders, a precision switch, robot
model settings and network tuning for its own IK stack and robot model. This
node reads the controller poses and buttons off the session and nothing else,
so those controls change nothing the robot does. The page served at `/` is
teleop_xr's own `index.html` with one script appended that hides them as they
render, keyed on the ids, titles and labels the page bundle gives them; the
passthrough button, the connection state and the camera feeds stay.
`tests/test_page.py` holds those keys against the installed bundle.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

PATH = "/"

# A control goes with the block that labels it: the element's parent.
HIDDEN_CONTROL_IDS = ("speed-slider", "turn-speed-slider", "precision-mode")
# A card goes whole, found from its title: its controls render only once it
# is expanded.
HIDDEN_CARD_TITLES = ("Robot Settings", "Advanced Settings")
# The class teleop_xr gives a card's root element.
CARD_CLASS = "bg-card"
HIDDEN_BUTTON_LABELS = ("VR Mode",)
# The subtitle of the card whose controls go.
HIDDEN_LINES = ("Adjust movement sensitivity and precision",)

_SCRIPT = """
(() => {
  const hidden = HIDDEN;
  const hide = (element) => element?.style.setProperty("display", "none", "important");
  const trim = () => {
    for (const id of hidden.controlIds) hide(document.getElementById(id)?.parentElement);
    for (const button of document.querySelectorAll("button"))
      if (hidden.buttonLabels.includes(button.textContent.trim())) hide(button);
    for (const div of document.querySelectorAll("div")) {
      if (div.children.length) continue;
      const text = div.textContent.trim();
      if (hidden.cardTitles.includes(text)) hide(div.closest("." + hidden.cardClass));
      if (hidden.lines.includes(text)) hide(div);
    }
  };
  // The dashboard re-renders as the session changes (the mode buttons come
  // back when it ends); trim whatever it adds.
  new MutationObserver(trim).observe(document.documentElement, { childList: true, subtree: true });
  trim();
})();
"""

TRIM_SCRIPT = _SCRIPT.replace(
    "HIDDEN",
    json.dumps(
        {
            "controlIds": HIDDEN_CONTROL_IDS,
            "cardTitles": HIDDEN_CARD_TITLES,
            "cardClass": CARD_CLASS,
            "buttonLabels": HIDDEN_BUTTON_LABELS,
            "lines": HIDDEN_LINES,
        }
    ),
    1,
)

_BODY_END = "</body>"


def trimmed(dashboard: str) -> str:
    """`dashboard` with the trim script closing its body.

    Refuses a page that closes no body, or more than one: the script runs
    inside the document it trims.
    """
    closings = dashboard.count(_BODY_END)
    if closings != 1:
        raise ValueError(f"the dashboard must close one body, found {closings}")
    return dashboard.replace(_BODY_END, f"<script>{TRIM_SCRIPT}</script>{_BODY_END}")


def build_router(dashboard: Path) -> APIRouter:
    """The page route, over `dashboard` read once: the page is static."""
    page = trimmed(dashboard.read_text(encoding="utf-8"))
    router = APIRouter()

    @router.get(PATH, response_class=HTMLResponse)
    async def show_dashboard() -> HTMLResponse:
        return HTMLResponse(page)

    return router
