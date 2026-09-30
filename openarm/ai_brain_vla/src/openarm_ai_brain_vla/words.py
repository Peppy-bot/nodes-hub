"""The one rule for whether a description names a label, shared by the
core's best match and the enrolment gallery: the description is the label,
or every word of it is a whole word of the label, stopwords aside. So "the
coffee can please" names "coffee can" and "red coffee can" but not
"thermos", and "can" names "coffee can" but not "candle".
"""

from __future__ import annotations

from typing import Sequence

# Words a description carries that name nothing.
STOPWORDS = frozenset({"a", "an", "the", "this", "that", "my", "some", "please", "me", "of", "it"})


def normalised(text: str) -> str:
    """`text` as names are compared: lower case, underscores as spaces,
    one space between words."""
    return " ".join(text.lower().replace("_", " ").split())


def content_words(text: str) -> frozenset[str]:
    """The words of `text` that name something."""
    return frozenset(normalised(text).split()) - STOPWORDS


def named_by(description: str, labels: Sequence[str]) -> list[int]:
    """The indices of the labels `description` names: the labels that are
    the description, else those that hold every content word of it as a
    whole word. Empty for an empty description or one of stopwords alone."""
    wanted = normalised(description)
    if not wanted:
        return []
    exact = [i for i, label in enumerate(labels) if normalised(label) == wanted]
    if exact:
        return exact
    words = content_words(description)
    if not words:
        return []
    return [i for i, label in enumerate(labels) if words <= frozenset(normalised(label).split())]
