import time


class Timer:
    """Context manager for timing code blocks and printing elapsed seconds."""

    def __init__(self, prefix: str = 'Cost') -> None:
        """Initialize the Timer context manager.

        Args:
            prefix (str): Prefix used when printing elapsed time.
        """
        self.prefix = prefix
        self.start_time: float | None = None

    def tic(self) -> None:
        """Start or reset the timer."""
        self.start_time = time.time()

    def toc(self) -> None:
        """Stop the timer and print the elapsed time in seconds."""
        print('{}: {}'.format(self.prefix, time.time() - self.start_time))

    def __enter__(self):
        """Enter the runtime context and start timing."""
        self.tic()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the runtime context, printing elapsed time."""
        self.toc()


def get_cur_time() -> str:
    """Current local time formatted as YYYY-MM-DD-HHMMSS."""
    return time.strftime('%Y-%m-%d-%H%M%S', time.localtime(time.time()))
