"""Boots the node in-process under the generated test harness, with the
parameters its manifest requires, and checks that it comes up and shuts
down cleanly."""

from conftest import PARAMS
from peppygen.fixtures import harness
from peppygen.parameters import Parameters

from openarm_ai_brain_vla.__main__ import setup


async def test_node_boots_and_shuts_down_cleanly():
    params = Parameters.from_dict(dict(PARAMS))
    async with harness.start(setup, parameters=params) as h:
        assert h.instance_id


def test_the_nodes_own_info_lines_reach_stderr(capsys):
    import logging

    from openarm_ai_brain_vla.__main__ import configure_logging

    configure_logging()
    configure_logging()  # idempotent: one handler, however many times it is called
    logger = logging.getLogger("openarm_ai_brain_vla.perception.sam3_siglip")
    logger.info("sam3_siglip: %d items", 120)
    logging.getLogger("transformers").info("stays quiet")
    err = capsys.readouterr().err
    assert err.count("[brain] sam3_siglip: 120 items") == 1 and "stays quiet" not in err
