These runs are INVALID for old/new comparisons and failure attribution.

The first harness incorrectly decoded synchronously for the old client (the actual
base commit already uses asyncio.to_thread) and performed cleanup through an
unpooled aiohttp client with different timeout/error handling. In particular, its
ConnectionTimeoutError failures may originate in this artificial cleanup path.
Do not use these timings or failure counts as evidence of a production fix.
The replacement harness calls the exact base-commit OpenAIEndpointTracer and HTTP
helpers, with the production connection limit, versus the working implementation.
