"""Console output that is also saved to run_log.txt."""

import sys


def format_duration(seconds):
    """Format a duration in seconds as e.g. '1h 03m 12s', '3m 05s', or '4.2s'."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    return f"{minutes}m {secs:02d}s"


class RunLog:
    """Collects console-style lines and writes them to a run_log.txt at the end."""

    def __init__(self):
        self.lines = []

    def emit(self, message=""):
        print(message)
        self.lines.append(message)

    def fail(self, message):
        """Record an error message in the log, then exit with it."""
        self.lines.append(message)
        sys.exit(message)

    def write(self, path):
        with open(path, "w") as handle:
            handle.write("\n".join(self.lines) + "\n")
