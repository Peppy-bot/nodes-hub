import asyncio
from types import SimpleNamespace

from peppygen.emitted_topics.headset import page as page_topic

from tests.helpers import boot
from xr_commander.page import publish_page

URLS = ["https://198.51.100.7:4443", "https://100.64.0.7:4443"]


async def test_the_page_goes_out_with_its_urls():
    async with boot() as h:
        await publish_page(h.node_runner, page_topic, URLS)
        published = await asyncio.wait_for(h.emitted.headset_page.next(), 10.0)
        assert published.urls == URLS


async def test_a_failed_publish_is_logged_and_the_session_drives_on(capsys):
    class Publisher:
        async def publish(self, _message):
            raise RuntimeError("wire down")

    async def declare_publisher(_node_runner):
        return Publisher()

    topic = SimpleNamespace(
        declare_publisher=declare_publisher, build_message=lambda urls: urls
    )
    await publish_page(None, topic, URLS)
    assert "failed to publish the page's URLs: RuntimeError('wire down')" in (
        capsys.readouterr().out
    )
