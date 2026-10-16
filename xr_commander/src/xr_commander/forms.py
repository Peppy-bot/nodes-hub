"""The plain forms the node's pages take: one urlencoded field per post."""

from __future__ import annotations

import urllib.parse

from fastapi import Request
from starlette.requests import ClientDisconnect

_FORM_MEDIA_TYPE = "application/x-www-form-urlencoded"
# Longer than any field a page takes; a bigger body is refused unread.
_MAX_BODY_BYTES = 4096
# A form posts and the browser is sent back to the page: a reload then
# repeats no post.
SEE_OTHER = 303
# The status of a refused submission.
BAD_REQUEST = 400
# The status of a choice taken: the page's script reads the status and no
# body.
NO_CONTENT = 204


async def read_capped_body(request: Request) -> bytes:
    """The request body, read in chunks and refused at the first chunk past
    the cap, so a bigger submission is never held whole."""
    chunks: list[bytes] = []
    size = 0
    try:
        async for chunk in request.stream():
            size += len(chunk)
            if size > _MAX_BODY_BYTES:
                raise ValueError(f"form is larger than {_MAX_BODY_BYTES} bytes")
            chunks.append(chunk)
    except ClientDisconnect:
        raise ValueError("request ended before the form was complete") from None
    return b"".join(chunks)


def form_field(content_type: str, body: bytes, field: str) -> str:
    """The raw `field` a submitted form carries.

    Structure only: the body's size belongs to whoever read it, and what makes
    the value acceptable belongs to the page. The urlencoded body is decoded
    with the standard library's parser.
    """
    media_type = content_type.split(";")[0].strip().lower()
    if media_type != _FORM_MEDIA_TYPE:
        raise ValueError(f"form must be {_FORM_MEDIA_TYPE}, got {media_type or 'none'}")
    try:
        fields = urllib.parse.parse_qs(body.decode("utf-8"), keep_blank_values=True)
    except UnicodeDecodeError:
        raise ValueError("form is not valid UTF-8") from None
    values = fields.get(field, [])
    if len(values) != 1:
        raise ValueError(f"form must have exactly one {field} field, got {len(values)}")
    return values[0]
