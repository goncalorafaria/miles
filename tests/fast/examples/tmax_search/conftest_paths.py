"""Import paths of the search port's tests: the example and the pinned Prime-RL source modules it runs."""

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[4]
PRIME_RL_SOURCE = Path(
    "/weka/oe-adapt-default/shashankg/miles_search/prime-rl-sources/"
    "a5fcb924645d9bd8f326efd5087c8de6447c83fe17c6d07c36e99ee736d30a37/source"
)
PRIME_RL_CONFIG = "configs/experiments/search-agent-bc-rl-v1-1-fp32-lr5e-7/rl.toml"

needs_prime_rl_source = pytest.mark.skipif(not PRIME_RL_SOURCE.is_dir(), reason="needs the pinned Prime-RL source")

for path in (
    REPO / "examples" / "tmax_search",
    REPO / "examples" / "tmax",
    PRIME_RL_SOURCE / "deps/literegistry/literegistry_tool_client/src",
    PRIME_RL_SOURCE / "environments/search_agent",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
