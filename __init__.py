"""ZEROne — M.I.P. Runtime.

A portable character runtime for LLM agents. Define a persona once,
switch models freely, and the character adapter enforces consistency.
"""

__version__ = "0.2.0"
__all__ = ["ZEROne", "Config", "build_provider"]

from zerone import ZEROne, Config
from providers import build_provider
