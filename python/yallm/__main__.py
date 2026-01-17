import os
import sys

from yallm import find_yallm_bin


def _run() -> None:
    yallm = os.fsdecode(find_yallm_bin())

    if sys.platform == "win32":
        import subprocess

        # Avoid emitting a traceback on interrupt
        try:
            completed_process = subprocess.run([yallm, *sys.argv[1:]])
        except KeyboardInterrupt:
            sys.exit(2)

        sys.exit(completed_process.returncode)
    else:
        os.execvpe(yallm, [yallm, *sys.argv[1:]])


if __name__ == "__main__":
    _run()
