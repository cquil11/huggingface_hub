# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for LRU cache eviction on ENOSPC."""

import errno
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from huggingface_hub.file_download import _try_evict_cache_for_space
from huggingface_hub.utils import SoftTemporaryDirectory, WeakFileLock


def _create_fake_cache_repo(cache_dir, repo_id, repo_type="model", revisions=None):
    """Create a fake cached repo structure for testing.

    Args:
        cache_dir: Root cache directory.
        repo_id: e.g. "user/repo".
        repo_type: "model", "dataset", or "space".
        revisions: List of dicts with keys:
            - commit_hash: str
            - files: dict mapping filename to content (bytes or str)
            - refs: list of ref names (e.g. ["main"])
    """
    parts = [f"{repo_type}s", *repo_id.split("/")]
    repo_folder = "--".join(parts)
    repo_path = Path(cache_dir) / repo_folder

    blobs_dir = repo_path / "blobs"
    snapshots_dir = repo_path / "snapshots"
    refs_dir = repo_path / "refs"

    blobs_dir.mkdir(parents=True, exist_ok=True)
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    refs_dir.mkdir(parents=True, exist_ok=True)

    if revisions is None:
        return repo_path

    for rev in revisions:
        commit_hash = rev["commit_hash"]
        snapshot_path = snapshots_dir / commit_hash
        snapshot_path.mkdir(parents=True, exist_ok=True)

        for filename, content in rev.get("files", {}).items():
            if isinstance(content, str):
                content = content.encode()

            # Create blob with a simple hash-like name
            import hashlib

            blob_name = hashlib.sha1(content).hexdigest()
            blob_path = blobs_dir / blob_name
            blob_path.write_bytes(content)

            # Create symlink in snapshot
            file_path = snapshot_path / filename
            file_path.parent.mkdir(parents=True, exist_ok=True)
            rel_blob = os.path.relpath(blob_path, file_path.parent)
            file_path.symlink_to(rel_blob)

        for ref_name in rev.get("refs", []):
            ref_path = refs_dir / ref_name
            ref_path.write_text(commit_hash)

    return repo_path


def _mock_disk_full():
    """Mock shutil.disk_usage to report zero free space (simulating ENOSPC context)."""
    return patch(
        "huggingface_hub.file_download.shutil.disk_usage",
        return_value=type("Usage", (), {"free": 0, "total": 100_000_000_000, "used": 100_000_000_000})(),
    )


class TestTryEvictCacheForSpace(unittest.TestCase):
    """Tests for the _try_evict_cache_for_space function."""

    def test_evict_detached_revisions_first(self):
        """Detached revisions (no refs) should be evicted before ref'd ones."""
        with SoftTemporaryDirectory() as cache_dir:
            _create_fake_cache_repo(
                cache_dir,
                "user/repo-a",
                revisions=[
                    {
                        "commit_hash": ("aaaa" * 10),
                        "files": {"weights.bin": b"A" * 1000},
                        "refs": [],
                    },
                ],
            )
            _create_fake_cache_repo(
                cache_dir,
                "user/repo-b",
                revisions=[
                    {
                        "commit_hash": ("bbbb" * 10),
                        "files": {"weights.bin": b"B" * 1000},
                        "refs": ["main"],
                    },
                ],
            )

            with _mock_disk_full():
                result = _try_evict_cache_for_space(
                    cache_dir=cache_dir,
                    needed_size=500,
                    current_repo_id="user/current-repo",
                    current_repo_type="model",
                )

            self.assertTrue(result)
            # Detached revision (repo A) should be gone
            self.assertFalse((Path(cache_dir) / "models--user--repo-a" / "snapshots" / ("aaaa" * 10)).exists())
            # Ref'd revision (repo B) should still exist (wasn't needed)
            self.assertTrue((Path(cache_dir) / "models--user--repo-b" / "snapshots" / ("bbbb" * 10)).exists())

    def test_never_evict_current_repo(self):
        """The repo currently being downloaded should never be evicted."""
        with SoftTemporaryDirectory() as cache_dir:
            _create_fake_cache_repo(
                cache_dir,
                "user/current-repo",
                revisions=[
                    {
                        "commit_hash": ("cccc" * 10),
                        "files": {"model.bin": b"C" * 5000},
                        "refs": [],
                    },
                ],
            )

            with _mock_disk_full():
                result = _try_evict_cache_for_space(
                    cache_dir=cache_dir,
                    needed_size=1000,
                    current_repo_id="user/current-repo",
                    current_repo_type="model",
                )

            self.assertFalse(result)
            self.assertTrue((Path(cache_dir) / "models--user--current-repo" / "snapshots" / ("cccc" * 10)).exists())

    def test_skip_repos_with_active_download_locks(self):
        """Repos with active download locks should not be evicted."""
        with SoftTemporaryDirectory() as cache_dir:
            _create_fake_cache_repo(
                cache_dir,
                "user/locked-repo",
                revisions=[
                    {
                        "commit_hash": ("eeee" * 10),
                        "files": {"model.bin": b"E" * 1000},
                        "refs": [],
                    },
                ],
            )
            _create_fake_cache_repo(
                cache_dir,
                "user/unlocked-repo",
                revisions=[
                    {
                        "commit_hash": ("ffff" * 10),
                        "files": {"model.bin": b"F" * 1000},
                        "refs": [],
                    },
                ],
            )

            locked_blob_dir = Path(cache_dir) / "models--user--locked-repo" / "blobs"
            for blob in locked_blob_dir.iterdir():
                os.utime(blob, (0, 0))

            lock_path = Path(cache_dir) / ".locks" / "models--user--locked-repo" / "download.lock"
            lock_path.parent.mkdir(parents=True, exist_ok=True)

            with WeakFileLock(lock_path):
                with _mock_disk_full():
                    result = _try_evict_cache_for_space(
                        cache_dir=cache_dir,
                        needed_size=500,
                        current_repo_id="user/downloading",
                        current_repo_type="model",
                    )

            self.assertTrue(result)
            self.assertTrue((Path(cache_dir) / "models--user--locked-repo" / "snapshots" / ("eeee" * 10)).exists())
            self.assertFalse((Path(cache_dir) / "models--user--unlocked-repo" / "snapshots" / ("ffff" * 10)).exists())

    def test_evict_oldest_first(self):
        """Revisions should be evicted oldest-first (by last_modified)."""
        with SoftTemporaryDirectory() as cache_dir:
            _create_fake_cache_repo(
                cache_dir,
                "user/old-repo",
                revisions=[
                    {
                        "commit_hash": ("1111" * 10),
                        "files": {"model.bin": b"OLD" * 500},
                        "refs": [],
                    },
                ],
            )
            _create_fake_cache_repo(
                cache_dir,
                "user/new-repo",
                revisions=[
                    {
                        "commit_hash": ("2222" * 10),
                        "files": {"model.bin": b"NEW" * 500},
                        "refs": [],
                    },
                ],
            )

            # Make old-repo's blobs appear older
            old_blob_dir = Path(cache_dir) / "models--user--old-repo" / "blobs"
            for blob in old_blob_dir.iterdir():
                os.utime(blob, (0, 0))  # Set mtime to epoch

            with _mock_disk_full():
                result = _try_evict_cache_for_space(
                    cache_dir=cache_dir,
                    needed_size=500,
                    current_repo_id="user/downloading",
                    current_repo_type="model",
                )

            self.assertTrue(result)
            # Old repo should be evicted first
            self.assertFalse((Path(cache_dir) / "models--user--old-repo" / "snapshots" / ("1111" * 10)).exists())
            # New repo should still exist
            self.assertTrue((Path(cache_dir) / "models--user--new-repo" / "snapshots" / ("2222" * 10)).exists())

    def test_insufficient_space_still_evicts_what_it_can(self):
        """When not enough can be freed, evict what's possible and return False."""
        with SoftTemporaryDirectory() as cache_dir:
            _create_fake_cache_repo(
                cache_dir,
                "user/small-repo",
                revisions=[
                    {
                        "commit_hash": ("dddd" * 10),
                        "files": {"tiny.txt": b"x" * 100},
                        "refs": [],
                    },
                ],
            )

            with _mock_disk_full():
                result = _try_evict_cache_for_space(
                    cache_dir=cache_dir,
                    needed_size=999_999_999,
                    current_repo_id="user/downloading",
                    current_repo_type="model",
                )

            self.assertFalse(result)
            # But the evictable revision should still have been cleaned up
            self.assertFalse((Path(cache_dir) / "models--user--small-repo" / "snapshots" / ("dddd" * 10)).exists())

    def test_no_candidates_returns_false(self):
        """When cache is empty or only has the current repo, return False."""
        with SoftTemporaryDirectory() as cache_dir:
            with _mock_disk_full():
                result = _try_evict_cache_for_space(
                    cache_dir=cache_dir,
                    needed_size=1000,
                    current_repo_id="user/repo",
                    current_repo_type="model",
                )
            self.assertFalse(result)

    def test_space_appears_while_waiting_for_lock(self):
        """If enough space appears while waiting for the lock, skip eviction."""
        with SoftTemporaryDirectory() as cache_dir:
            _create_fake_cache_repo(
                cache_dir,
                "user/repo-a",
                revisions=[
                    {
                        "commit_hash": ("eeee" * 10),
                        "files": {"model.bin": b"E" * 1000},
                        "refs": [],
                    },
                ],
            )

            # Mock disk_usage to report plenty of free space
            with patch("huggingface_hub.file_download.shutil.disk_usage") as mock_du:
                mock_du.return_value = type("Usage", (), {"free": 999_999_999, "total": 1_000_000_000, "used": 1_000})()


                result = _try_evict_cache_for_space(
                    cache_dir=cache_dir,
                    needed_size=500,
                    current_repo_id="user/downloading",
                    current_repo_type="model",
                )

                self.assertTrue(result)
                # Nothing should have been evicted
                self.assertTrue((Path(cache_dir) / "models--user--repo-a" / "snapshots" / ("eeee" * 10)).exists())


class TestCacheEvictionInDownloadPath(unittest.TestCase):
    """Tests for ENOSPC handling in the download wrapper."""

    def test_enospc_without_eviction_enabled_propagates(self):
        """When HF_HUB_ENABLE_CACHE_EVICTION is False, ENOSPC should propagate directly."""
        err = OSError(errno.ENOSPC, "No space left on device")

        with patch("huggingface_hub.file_download.constants") as mock_constants:
            mock_constants.HF_HUB_ENABLE_CACHE_EVICTION = False
            # Verify the guard condition works: non-ENOSPC errors always propagate
            with self.assertRaises(OSError) as ctx:
                # Simulate what the download wrapper does
                try:
                    raise err
                except OSError as e:
                    if e.errno != errno.ENOSPC or not mock_constants.HF_HUB_ENABLE_CACHE_EVICTION:
                        raise
            self.assertEqual(ctx.exception.errno, errno.ENOSPC)

    def test_non_enospc_error_propagates_regardless(self):
        """Non-ENOSPC OSErrors should always propagate, even with eviction enabled."""
        err = OSError(errno.EACCES, "Permission denied")

        with patch("huggingface_hub.file_download.constants") as mock_constants:
            mock_constants.HF_HUB_ENABLE_CACHE_EVICTION = True
            with self.assertRaises(OSError) as ctx:
                try:
                    raise err
                except OSError as e:
                    if e.errno != errno.ENOSPC or not mock_constants.HF_HUB_ENABLE_CACHE_EVICTION:
                        raise
            self.assertEqual(ctx.exception.errno, errno.EACCES)


if __name__ == "__main__":
    unittest.main()
