"""Keep explicit shared-lock waits outside a child command timeout."""
import fcntl
import os
from pathlib import Path


def acquire_command_lock(command,lock_path):
    """Separate shared-lock queue time from the command's execution timeout."""
    if len(command)<3 or Path(command[0]).name!='flock' or command[1]!=lock_path:
        return command,None
    descriptor=os.open(lock_path,os.O_CREAT|os.O_RDWR,0o600)
    try:
        fcntl.flock(descriptor,fcntl.LOCK_EX)
    except BaseException:
        os.close(descriptor);raise
    # flock on an inherited descriptor uses the same open-file description.
    # Opening the pathname again while the parent holds it would deadlock.
    # Descriptor mode takes no command operands; pass argv through a shell
    # without interpolating any command text or reopening the lock file.
    return ['bash','-c','flock -x "$1" || exit "$?"; shift; exec "$@"',
            'tilemega-lock',str(descriptor),*command[2:]],descriptor
