import contextlib

from rich.console import Console

console = Console(highlight=False, soft_wrap=True)


def info(message: str, **kwargs):
    console.print(f"[cyan]INFO[/] {message}", **kwargs)


def error(message: str, **kwargs):
    console.print(f"[red]ERROR[/] {message}", **kwargs)


@contextlib.contextmanager
def status(message: str):
    status = console.status(message)
    try:
        status.start()
        yield
    finally:
        status.stop()
