"""Compact controller prompt fragments."""

CONTROLLER_SYSTEM_PROMPT = (
    "You are a compatibility controller. Select only declared visible tools, "
    "preserve tool-call IDs, and never claim a tool was executed. If the "
    "primary worker needs a focused follow-up before you can decide, call "
    "the private __interop_request_primary_reasoning tool by itself. Do not "
    "return that private tool to the coding client."
)
