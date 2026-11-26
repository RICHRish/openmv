"""
boot.py - frozen into the firmware, so it runs before any script on flash or SD.

To stop the app from starting (for development): create an empty file called
SKIP_APP on the camera drive (internal flash or SD card), then reset the camera.
"""
import os
import sys
import time
import machine

STARTUP_DELAY_MS = 3000     # window to connect the IDE and press Stop
SKIP_FILES = ("/flash/SKIP_APP", "/sdcard/SKIP_APP")
CRASH_LOG = "horizon_crash.txt"


def _exists(path):
    try:
        os.stat(path)
        return True
    except OSError:
        return False


def _start():
    for path in SKIP_FILES:
        if _exists(path):
            print("boot: found", path, "- horizon app not started")
            return
    time.sleep_ms(STARTUP_DELAY_MS)
    # Search the firmware first, so old copies on flash/SD cannot replace it.
    sys.path.insert(0, ".frozen")
    try:
        import horizon_app  # runs forever
    except Exception as e:  # IDE Stop is a KeyboardInterrupt, so it is not caught
        sys.print_exception(e)
        try:
            if not _exists(CRASH_LOG):  # keep the first crash, limit flash wear
                with open(CRASH_LOG, "w") as f:
                    sys.print_exception(e, f)
        except OSError:
            pass
        time.sleep_ms(5000)
        machine.reset()


_start()
