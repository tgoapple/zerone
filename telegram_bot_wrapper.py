#!/usr/bin/env python3
"""Wrapper that patches telegram_bot.main() to force-flush all prints."""

import sys

# Patch print to always flush
_orig_print = print
def _flushed_print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _orig_print(*args, **kwargs)

__builtins__["print"] = _flushed_print

import telegram_bot
sys.exit(telegram_bot.main())
