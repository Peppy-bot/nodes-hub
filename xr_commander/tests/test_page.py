from html.parser import HTMLParser

import pytest
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from tests.helpers import asgi_request
from xr_commander.page import (
    CARD_CLASS,
    HIDDEN_BUTTON_LABELS,
    HIDDEN_CARD_TITLES,
    HIDDEN_CONTROL_IDS,
    HIDDEN_LINES,
    PATH,
    TRIM_SCRIPT,
    build_router,
    trimmed,
)
from xr_commander.xr import DASHBOARD, include_router_first

# What the trimmed page keeps: the mode the node is driven in, the state of
# the page's link to it, and the cameras it relays.
KEPT = ("Passthrough", "Connection", "Camera Feeds", "Teleoperation Control")

VOID_TAGS = {"meta", "link", "input", "img", "br", "hr"}


class Element:
    """One element of the parsed dashboard: enough of the DOM for the trim
    script's lookups, so a test can apply its rules to the shipped page."""

    def __init__(self, tag, attrs, parent):
        self.tag = tag
        self.attrs = dict(attrs)
        self.parent = parent
        self.nodes: list[Element | str] = []

    def text(self) -> str:
        return "".join(
            node if isinstance(node, str) else node.text() for node in self.nodes
        ).strip()

    def is_leaf(self) -> bool:
        return not any(isinstance(node, Element) for node in self.nodes)

    def descendants(self):
        for node in self.nodes:
            if isinstance(node, Element):
                yield node
                yield from node.descendants()

    def ancestors(self):
        node = self.parent
        while node is not None:
            yield node
            node = node.parent

    def closest(self, class_name: str):
        return next(
            (
                node
                for node in (self, *self.ancestors())
                if class_name in node.attrs.get("class", "").split()
            ),
            None,
        )


class Dom(HTMLParser):
    def __init__(self):
        super().__init__()
        self.root = Element("document", [], None)
        self._open = self.root

    def handle_starttag(self, tag, attrs):
        element = Element(tag, attrs, self._open)
        self._open.nodes.append(element)
        if tag not in VOID_TAGS:
            self._open = element

    def handle_startendtag(self, tag, attrs):
        self._open.nodes.append(Element(tag, attrs, self._open))

    def handle_endtag(self, tag):
        self._open = self._open.parent

    def handle_data(self, data):
        self._open.nodes.append(data)


def dashboard() -> Element:
    dom = Dom()
    dom.feed(DASHBOARD.read_text(encoding="utf-8"))
    return dom.root


def by_id(root: Element, element_id: str) -> list[Element]:
    return [e for e in root.descendants() if e.attrs.get("id") == element_id]


def leaves(root: Element, text: str) -> list[Element]:
    return [e for e in root.descendants() if e.is_leaf() and e.text() == text]


def buttons(root: Element, label: str) -> list[Element]:
    return [e for e in root.descendants() if e.tag == "button" and e.text() == label]


def hidden_by_rule(root: Element) -> dict[str, list[Element | None]]:
    """What each key of the trim script hides, by the script's rules."""
    return {
        **{i: [e.parent for e in by_id(root, i)] for i in HIDDEN_CONTROL_IDS},
        **{
            t: [e.closest(CARD_CLASS) for e in leaves(root, t)]
            for t in HIDDEN_CARD_TITLES
        },
        **{b: buttons(root, b) for b in HIDDEN_BUTTON_LABELS},
        **{line: leaves(root, line) for line in HIDDEN_LINES},
    }


def test_the_dashboard_is_teleop_xrs_prebuilt_page():
    assert DASHBOARD.name == "index.html"
    assert "<title>TeleopXR</title>" in DASHBOARD.read_text(encoding="utf-8")


def test_every_key_hides_exactly_one_element_of_the_installed_page():
    # The keys are teleop_xr's; a release that renames one fails here.
    for key, hidden in hidden_by_rule(dashboard()).items():
        assert len(hidden) == 1, key
        assert hidden[0] is not None, key


@pytest.mark.parametrize("control_id", HIDDEN_CONTROL_IDS)
def test_a_control_goes_with_the_label_that_names_it(control_id):
    (control,) = by_id(dashboard(), control_id)
    labels = [e for e in control.parent.descendants() if e.tag == "label"]
    assert [label.attrs.get("for") for label in labels] == [control_id]


@pytest.mark.parametrize("text", KEPT)
def test_what_the_node_honors_is_outside_everything_hidden(text):
    root = dashboard()
    hidden = [e for found in hidden_by_rule(root).values() for e in found]
    shown = leaves(root, text) + buttons(root, text)
    assert shown, text
    for element in shown:
        assert element not in hidden
        assert not any(ancestor in hidden for ancestor in element.ancestors())


def test_the_script_carries_every_key():
    for key in (
        *HIDDEN_CONTROL_IDS,
        *HIDDEN_CARD_TITLES,
        *HIDDEN_BUTTON_LABELS,
        *HIDDEN_LINES,
        CARD_CLASS,
    ):
        assert f'"{key}"' in TRIM_SCRIPT


def test_the_script_closes_the_body_it_trims():
    page = trimmed("<html><body><main>x</main></body></html>")
    assert page.endswith(f"<script>{TRIM_SCRIPT}</script></body></html>")
    assert page.startswith("<html><body><main>x</main>")


@pytest.mark.parametrize(
    "dashboard_html", ["<html>no body</html>", "<body></body><body></body>"]
)
def test_a_page_that_is_not_one_document_is_refused(dashboard_html):
    with pytest.raises(ValueError, match="one body"):
        trimmed(dashboard_html)


def test_the_route_outranks_the_page_teleop_xr_serves():
    # teleop_xr answers '/' with its untrimmed index; the node's route must
    # come first.
    app = FastAPI()

    @app.get(PATH)
    async def untrimmed() -> HTMLResponse:
        return HTMLResponse("<body>untrimmed</body>")

    include_router_first(app, build_router(DASHBOARD))
    served = asgi_request(app, "GET", PATH)
    assert served.status == 200
    assert "<title>TeleopXR</title>" in served.body
    assert TRIM_SCRIPT in served.body
    assert "untrimmed" not in served.body
