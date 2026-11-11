"""Mac control tools the LLM can call.

Each public function below is passed to Ollama as a tool; the type hints and
docstrings become the tool schema. Every tool returns a short string that the
LLM turns into a spoken reply.

Raw shell / AppleScript run only after a spoken "yes" (see `set_confirm`),
because a misheard command could otherwise do real damage.
"""

from __future__ import annotations

import difflib
import subprocess
import webbrowser
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

APP_DIRS = [
    Path("/Applications"),
    Path("/Applications/Utilities"),
    Path("/System/Applications"),
    Path("/System/Applications/Utilities"),
    Path.home() / "Applications",
]
ALIASES = {
    "vs code": "Visual Studio Code", "vscode": "Visual Studio Code", "code": "Visual Studio Code",
    "chrome": "Google Chrome", "brave": "Brave Browser", "intellij": "IntelliJ IDEA",
    "zoom": "zoom.us", "settings": "System Settings", "system preferences": "System Settings",
    "teams": "Microsoft Teams", "word": "Microsoft Word", "outlook": "Microsoft Outlook",
    "davinci": "DaVinci Resolve", "obs studio": "OBS", "finder": "Finder",
}
KEY_CODES = {
    "return": 36, "enter": 36, "tab": 48, "space": 49, "delete": 51, "backspace": 51, "escape": 53, "esc": 53,
    "left": 123, "right": 124, "down": 125, "up": 126, "home": 115, "end": 119, "pageup": 116, "pagedown": 121,
    **{f"f{i}": c for i, c in zip(range(1, 13), [122, 120, 99, 118, 96, 97, 98, 100, 101, 109, 103, 111])},
}
MODIFIERS = {"cmd": "command", "command": "command", "shift": "shift", "alt": "option", "option": "option",
             "opt": "option", "ctrl": "control", "control": "control"}

_confirm: Callable[[str], bool] = lambda question: False
confirm_risky = True


def set_confirm(fn: Callable[[str], bool], enabled: bool = True) -> None:
    global _confirm, confirm_risky
    _confirm, confirm_risky = fn, enabled


# ---------------------------------------------------------------- helpers
@lru_cache(maxsize=1)
def installed_apps() -> dict[str, str]:
    """Lowercased app name -> real app name."""
    apps = {"finder": "Finder"}
    for d in APP_DIRS:
        if d.is_dir():
            for p in d.glob("*.app"):
                apps[p.stem.lower()] = p.stem
    return apps


def resolve_app(name: str) -> str | None:
    n = name.lower().strip().removesuffix(".app").removeprefix("the ").strip()
    apps = installed_apps()
    if n in ALIASES:
        return ALIASES[n]
    if n in apps:
        return apps[n]
    starts = [real for low, real in apps.items() if low.startswith(n)]
    if starts:
        return min(starts, key=len)
    contains = [real for low, real in apps.items() if n in low]
    if contains:
        return min(contains, key=len)
    close = difflib.get_close_matches(n, list(apps), n=1, cutoff=0.6)
    return apps[close[0]] if close else None


def _osa(script: str, timeout: float = 15) -> tuple[bool, str]:
    r = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=timeout)
    return r.returncode == 0, (r.stdout or r.stderr).strip()


def _q(s: str) -> str:
    """Quote a Python string as an AppleScript string literal."""
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _clip(text: str, limit: int = 1500) -> str:
    return text if len(text) <= limit else text[:limit] + " ...(truncated)"


# ---------------------------------------------------------------- tools
def open_app(name: str) -> str:
    """Open (or bring to the front) a Mac application.

    Args:
        name: The application name, e.g. "Safari", "Xcode", "VS Code", "Spotify".
    """
    app = resolve_app(name)
    if not app:
        return f"No installed app matches '{name}'."
    r = subprocess.run(["open", "-a", app], capture_output=True, text=True)
    return f"Opened {app}." if r.returncode == 0 else f"Failed to open {app}: {r.stderr.strip()}"


def quit_app(name: str) -> str:
    """Quit a running Mac application gracefully (it may ask to save documents).

    Args:
        name: The application name.
    """
    app = resolve_app(name) or name
    ok, out = _osa(f"tell application {_q(app)} to quit")
    return f"Quit {app}." if ok else f"Could not quit {app}: {out}"


def list_running_apps() -> str:
    """List the applications currently running, and which one is in front."""
    ok, out = _osa(
        'tell application "System Events" to set a to name of every application process whose background only is false\n'
        'tell application "System Events" to set f to name of first application process whose frontmost is true\n'
        'set AppleScript\'s text item delimiters to ", "\n'
        'return (a as text) & "|" & f'
    )
    if not ok:
        return f"Could not list apps: {out}"
    apps, _, front = out.rpartition("|")
    return f"Running: {apps}. Frontmost: {front}."


def open_url(url: str) -> str:
    """Open a web page in the default browser. Needs internet for remote sites.

    Args:
        url: Full URL or a domain like "github.com".
    """
    if "://" not in url:
        url = "https://" + url
    webbrowser.open(url)
    return f"Opened {url}."


def open_path(path: str) -> str:
    """Open a file or folder with its default app (folders open in Finder).

    Args:
        path: File or folder path. "~" is allowed, e.g. "~/Downloads".
    """
    p = Path(path).expanduser()
    if not p.exists():
        return f"{p} does not exist."
    subprocess.run(["open", str(p)])
    return f"Opened {p}."


def type_text(text: str) -> str:
    """Type text into the frontmost app as if from the keyboard.

    Args:
        text: The text to type.
    """
    ok, out = _osa(f'tell application "System Events" to keystroke {_q(text)}')
    return "Typed it." if ok else f"Typing failed (grant Accessibility permission?): {out}"


def press_keys(keys: str) -> str:
    """Press a keyboard shortcut or key in the frontmost app.

    Args:
        keys: Keys joined by "+", e.g. "cmd+t", "cmd+shift+n", "return", "escape", "cmd+tab".
    """
    parts = [k.strip().lower() for k in keys.split("+") if k.strip()]
    mods = [MODIFIERS[k] + " down" for k in parts if k in MODIFIERS]
    main = [k for k in parts if k not in MODIFIERS]
    if len(main) != 1:
        return f"Unrecognised key combination '{keys}'."
    key = main[0]
    using = f" using {{{', '.join(mods)}}}" if mods else ""
    action = f"key code {KEY_CODES[key]}" if key in KEY_CODES else f"keystroke {_q(key)}"
    ok, out = _osa(f'tell application "System Events" to {action}{using}')
    return f"Pressed {keys}." if ok else f"Key press failed (grant Accessibility permission?): {out}"


def set_volume(percent: int) -> str:
    """Set the system output volume.

    Args:
        percent: 0 to 100. Use 0 to mute.
    """
    percent = max(0, min(100, int(percent)))
    ok, out = _osa(f"set volume output volume {percent}")
    return f"Volume set to {percent} percent." if ok else out


def media_control(action: str) -> str:
    """Control music playback in Spotify (if running) or Apple Music.

    Args:
        action: One of "play", "pause", "playpause", "next", "previous", "current" (what's playing).
    """
    _, running = _osa('tell application "System Events" to (name of processes) contains "Spotify"')
    player = "Spotify" if running == "true" else "Music"
    commands = {"play": "play", "pause": "pause", "playpause": "playpause",
                "next": "next track", "previous": "previous track"}
    if action == "current":
        ok, out = _osa(f'tell application "{player}" to return (name of current track) & " by " & (artist of current track)')
        return f"Now playing: {out}." if ok else "Nothing is playing."
    if action not in commands:
        return f"Unknown media action '{action}'."
    ok, out = _osa(f'tell application "{player}" to {commands[action]}')
    return f"{player}: {action} done." if ok else f"{player} error: {out}"


def run_applescript(script: str) -> str:
    """Run arbitrary AppleScript to control any scriptable app (Finder, Mail, Safari tabs,
    Notes, Reminders, window positions, etc.). The user is asked to confirm first.

    Args:
        script: The AppleScript source code.
    """
    if confirm_risky and not _confirm("Shall I run this AppleScript?\n" + script):
        return "The user declined. Nothing was run."
    ok, out = _osa(script, timeout=60)
    return _clip(out) if ok else f"AppleScript error: {_clip(out)}"


def run_shell(command: str) -> str:
    """Run a zsh shell command on the Mac and return its output. The user is asked to confirm first.

    Args:
        command: The shell command, e.g. "ls ~/Desktop" or "pmset -g batt".
    """
    if confirm_risky and not _confirm("Shall I run this command?\n" + command):
        return "The user declined. Nothing was run."
    try:
        r = subprocess.run(["zsh", "-lc", command], capture_output=True, text=True, timeout=60, cwd=Path.home())
    except subprocess.TimeoutExpired:
        return "The command timed out after 60 seconds."
    out = (r.stdout + r.stderr).strip() or "(no output)"
    return f"Exit code {r.returncode}. {_clip(out)}"


TOOLS = [open_app, quit_app, list_running_apps, open_url, open_path, type_text, press_keys,
         set_volume, media_control, run_applescript, run_shell]
BY_NAME = {f.__name__: f for f in TOOLS}


def call(name: str, args: dict) -> str:
    fn = BY_NAME.get(name)
    if not fn:
        return f"Unknown tool {name}."
    try:
        return fn(**(args or {}))
    except Exception as e:
        return f"{name} failed: {e}"
