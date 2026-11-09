"""Where the research core keeps its database and snapshots.

Never inside the repository and never inside a synced folder (OneDrive,
Dropbox, iCloud Drive): see ``UnsafeStorageLocationError``. The default is the
per-user local data directory; ``TAQ_DATA_DIR`` moves it.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from tradingagents_quant.errors import UnsafeStorageLocationError

APP_DIR_NAME = "tradingagents-quant"
DB_FILENAME = "taq.sqlite3"

# Variables a sync client sets to the root of the folder it syncs.
_SYNC_ROOT_VARIABLES = ("OneDrive", "OneDriveConsumer", "OneDriveCommercial")
# Folder names a sync client gives its root, for machines where the variable is
# missing: "OneDrive", "OneDrive - <organisation>", "Dropbox", "iCloudDrive".
_SYNC_FOLDER_NAMES = ("onedrive", "dropbox", "icloud drive", "iclouddrive")


def default_data_home(environ: Mapping[str, str] | None = None) -> Path:
    """The per-user local data directory, which no sync client manages."""
    environ = os.environ if environ is None else environ
    if os.name == "nt":
        base = environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    else:
        base = environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / APP_DIR_NAME


def data_home(environ: Mapping[str, str] | None = None) -> Path:
    """The configured data directory, resolved and checked; not created."""
    environ = os.environ if environ is None else environ
    raw = environ.get("TAQ_DATA_DIR", "").strip()
    path = Path(raw).expanduser() if raw else default_data_home(environ)
    # resolve() follows symlinks and Windows junctions, so a link that points
    # into a synced folder is judged by where the data really lands.
    path = path.resolve()
    assert_not_synced(path, environ)
    return path


def db_path(home: Path) -> Path:
    return home / DB_FILENAME


def snapshot_dir(home: Path) -> Path:
    return home / "snapshots"


def _within(child: Path, root: Path) -> bool:
    child_s = os.path.normcase(str(child))
    root_s = os.path.normcase(str(root)).rstrip("\\/")
    return child_s == root_s or child_s.startswith(root_s + os.sep)


def sync_root_of(path: Path, environ: Mapping[str, str] | None = None) -> str | None:
    """The synced folder ``path`` lies in, or None."""
    environ = os.environ if environ is None else environ
    for name in _SYNC_ROOT_VARIABLES:
        raw = environ.get(name, "").strip()
        if raw and _within(path, Path(raw).expanduser().resolve()):
            return raw
    for part in path.parts:
        lowered = part.lower()
        if any(lowered == n or lowered.startswith(n + " - ") for n in _SYNC_FOLDER_NAMES):
            return part
    return None


def assert_not_synced(path: Path, environ: Mapping[str, str] | None = None) -> None:
    root = sync_root_of(path, environ)
    if root is not None:
        raise UnsafeStorageLocationError(
            f"data directory {path} lies inside the synced folder {root!r}. "
            "SQLite and a sync client together lock or corrupt the database. "
            "Set TAQ_DATA_DIR to a local folder outside it, e.g. C:\\Dev\\data\\taq."
        )
