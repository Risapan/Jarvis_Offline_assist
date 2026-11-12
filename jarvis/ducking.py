"""Lower other media while Jarvis talks, then put it back.

macOS has no per-app volume API, so this works on the players that expose their own
volume to AppleScript: Spotify and Apple Music. They are only touched if already running
(a plain `tell application` would launch them).
"""

from __future__ import annotations

import atexit
import subprocess
import threading

PLAYERS = ("Spotify", "Music")


def _osa(script: str) -> str:
    try:
        r = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=3)
        return r.stdout.strip()
    except Exception:
        return ""


def _get_volume(app: str) -> int | None:
    out = _osa(f'if application "{app}" is running then tell application "{app}" to return sound volume')
    return int(out) if out.isdigit() else None


def _set_volume(app: str, volume: int) -> None:
    _osa(f'if application "{app}" is running then tell application "{app}" to set sound volume to {volume}')


class Ducker:
    def __init__(self, level: float = 0.2, restore_delay: float = 0.8):
        self.level = level  # fraction of the original volume while Jarvis speaks
        self.restore_delay = restore_delay  # wait this long after speech before restoring
        self._saved: dict[str, int] = {}
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None
        atexit.register(self.restore)

    def duck(self) -> None:
        """Lower running players (idempotent; cancels a pending restore)."""
        with self._lock:
            if self._timer:
                self._timer.cancel()
                self._timer = None
            if self._saved:
                return  # already ducked
            for app in PLAYERS:
                vol = _get_volume(app)
                if vol:
                    self._saved[app] = vol
                    _set_volume(app, max(1, round(vol * self.level)))

    def release(self) -> None:
        """Restore volumes shortly, unless Jarvis starts talking again first."""
        with self._lock:
            if not self._saved or self._timer:
                return
            self._timer = threading.Timer(self.restore_delay, self.restore)
            self._timer.daemon = True
            self._timer.start()

    def restore(self) -> None:
        with self._lock:
            self._timer = None
            saved, self._saved = self._saved, {}
        for app, vol in saved.items():
            _set_volume(app, vol)
