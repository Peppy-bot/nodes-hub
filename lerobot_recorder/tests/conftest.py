"""Process environment the suites depend on. The fake peppygen that used to
live here is gone: `peppy node sync` generates the real bindings plus the
`mock`/`fixtures` test packages under .peppy/libs/peppygen, and every suite
now imports those real modules (the harness suite boots the node over a real
ephemeral router). What remains are the two env side effects that must be in
place before pyarrow / huggingface_hub first import anywhere in the run."""

import os
import tempfile

# pyarrow's bundled jemalloc corrupts parquet reads after in-process av/torch
# use (the integration tests' dataset reload); the system allocator is stable.
# Must be set before pyarrow first loads, hence here, and unconditionally so
# ambient shell state cannot reintroduce the corruption.
os.environ["ARROW_DEFAULT_MEMORY_POOL"] = "system"
# The dataset reloads in the integration tests cache under HF_HOME; pointing
# it at a per-run temp dir keeps the suite from growing ~/.cache/huggingface.
# Must be set before huggingface_hub first loads, hence here.
os.environ["HF_HOME"] = tempfile.mkdtemp(prefix="lerobot_recorder_tests_hf_")
