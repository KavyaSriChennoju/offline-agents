"""Tiny terminal helpers. Colours switch off automatically if your terminal hates them."""
import os
import sys

_ON = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
if os.name == "nt":
    os.system("")  # enables ANSI colours in modern Windows terminals


def _c(code):
    return (lambda s: f"\033[{code}m{s}\033[0m") if _ON else (lambda s: str(s))


dim, bold = _c("2"), _c("1")
red, green, yellow, blue, magenta, cyan = (_c(str(n)) for n in (31, 32, 33, 34, 35, 36))


def ok(msg):
    print(f"  {green('ok')}    {msg}")


def bad(msg):
    print(f"  {red('nope')}  {msg}")


def meh(msg):
    print(f"  {yellow('hmm')}   {msg}")


def say_stream(piece: str):
    """Print a streamed token without a newline."""
    sys.stdout.write(piece)
    sys.stdout.flush()


def banner(title: str):
    line = "-" * (len(title) + 4)
    print(f"\n{cyan(line)}\n{cyan('| ')}{bold(title)}{cyan(' |')}\n{cyan(line)}")
