"""Data lives in a local, unsynced directory outside the repository."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tradingagents_quant import paths
from tradingagents_quant.errors import UnsafeStorageLocationError

REPO = Path(__file__).resolve().parents[2]


def _local(tmp_path: Path) -> dict[str, str]:
    return {"LOCALAPPDATA": str(tmp_path / "local"), "XDG_DATA_HOME": str(tmp_path / "xdg")}


def test_default_is_the_per_user_local_data_directory(tmp_path):
    env = _local(tmp_path)
    home = paths.data_home(env)
    assert home.name == paths.APP_DIR_NAME
    expected = env["LOCALAPPDATA"] if os.name == "nt" else env["XDG_DATA_HOME"]
    assert home.parent == Path(expected).resolve()


def test_the_real_default_is_outside_the_repository():
    environ = {k: v for k, v in os.environ.items() if k != "TAQ_DATA_DIR"}
    home = paths.data_home(environ)
    assert not paths._within(home, REPO.resolve())


def test_data_dir_variable_moves_it(tmp_path):
    env = {**_local(tmp_path), "TAQ_DATA_DIR": str(tmp_path / "elsewhere")}
    assert paths.data_home(env) == (tmp_path / "elsewhere").resolve()
    assert paths.db_path(paths.data_home(env)).name == paths.DB_FILENAME


def test_a_folder_inside_the_onedrive_root_is_refused(tmp_path):
    root = tmp_path / "SyncRoot"
    env = {"OneDrive": str(root), "TAQ_DATA_DIR": str(root / "data" / "taq")}
    with pytest.raises(UnsafeStorageLocationError, match="synced folder"):
        paths.data_home(env)


@pytest.mark.parametrize("folder", ["OneDrive", "OneDrive - Firma AG", "Dropbox", "iCloudDrive"])
def test_a_folder_named_like_a_sync_root_is_refused(tmp_path, folder):
    env = {"TAQ_DATA_DIR": str(tmp_path / folder / "taq")}
    with pytest.raises(UnsafeStorageLocationError):
        paths.data_home(env)


@pytest.mark.parametrize("folder", ["OneDriveBackup", "my-dropbox-notes", "data"])
def test_similar_names_are_not_mistaken_for_a_sync_root(tmp_path, folder):
    env = {"TAQ_DATA_DIR": str(tmp_path / folder / "taq")}
    assert paths.data_home(env).name == "taq"


def test_a_link_into_a_synced_folder_is_judged_by_its_target(tmp_path):
    root = tmp_path / "SyncRoot"
    (root / "data").mkdir(parents=True)
    link = tmp_path / "innocent"
    try:
        link.symlink_to(root / "data", target_is_directory=True)
    except OSError:
        if os.name != "nt":
            raise
        # Without the symlink privilege, Windows still allows a junction, which
        # is the link this project itself uses (C:\Dev\tradingagents).
        import _winapi

        _winapi.CreateJunction(str(root / "data"), str(link))
    with pytest.raises(UnsafeStorageLocationError):
        paths.data_home({"OneDrive": str(root), "TAQ_DATA_DIR": str(link)})
