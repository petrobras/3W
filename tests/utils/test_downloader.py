import json
import pytest
import warnings
import requests  # type: ignore
import time

from ThreeWToolkit.utils.downloader import FIGSHARE_BASE_URL, FIGSHARE_VERSION_IDS


class TestFigshareURLS:
    def test_known_versions(self):
        assert len(FIGSHARE_VERSION_IDS) == 4
        assert "1.0.0" in FIGSHARE_VERSION_IDS
        assert "1.1.0" in FIGSHARE_VERSION_IDS
        assert "1.1.1" in FIGSHARE_VERSION_IDS
        assert "2.0.0" in FIGSHARE_VERSION_IDS

    @pytest.fixture(scope="class")
    def all_metadata(self):
        """
        Fetch metadata for all versions. This fixture runs once per test class
        and all test methods depend on it. If this fails, all dependent tests are skipped.
        """
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }

        metadata_dict = {}
        for version in FIGSHARE_VERSION_IDS.keys():
            url_metadata = (
                FIGSHARE_BASE_URL
                + "/articles/"
                + FIGSHARE_VERSION_IDS[version]
                + "/files"
            )

            try:
                known_files = requests.get(url_metadata, headers=headers, timeout=30)
                known_files.raise_for_status()
                metadata = json.loads(known_files.text)
                metadata_dict[version] = metadata
            except requests.exceptions.HTTPError as http_err:
                warnings.warn(
                    f"Failed to fetch metadata for version {version}: {http_err}"
                )
                pytest.skip(f"Skipping tests for version {version} due to HTTP error.")
            except Exception as e:
                pytest.fail(f"Failed to fetch metadata for version {version}: {e}")
            time.sleep(1)  # Figshare prefers 1 fetch per second at most.

        return metadata_dict

    def test_v_1_0_0(self, all_metadata):
        metadata = all_metadata["1.0.0"]
        assert len(metadata) == 1
        meta = metadata[0]

        assert meta["name"] == "3w_dataset_1.0.0.zip"
        assert meta["supplied_md5"] == "189eb151fa07cf27f82976575586df1a"
        assert "download_url" in meta

    def test_v_1_1_0(self, all_metadata):
        metadata = all_metadata["1.1.0"]
        assert len(metadata) == 1
        meta = metadata[0]

        assert meta["name"] == "3w_dataset_1.1.0.zip"
        assert meta["supplied_md5"] == "e74641bb6b8a0466bb1f28c57a9163f9"
        assert "download_url" in meta

    def test_v_1_1_1(self, all_metadata):
        metadata = all_metadata["1.1.1"]
        assert len(metadata) == 1
        meta = metadata[0]

        assert meta["name"] == "3w_dataset_1.1.1.zip"
        assert meta["supplied_md5"] == "ae105a8abff4ac33058fea3ebb991fc2"
        assert "download_url" in meta

    def test_v_2_0_0(self, all_metadata):
        metadata = all_metadata["2.0.0"]
        assert len(metadata) == 1
        meta = metadata[0]

        assert meta["name"] == "3w_dataset_2.0.0.zip"
        assert meta["supplied_md5"] == "2c23b87b60c5d19ed9cf9559efa6ffa7"
        assert "download_url" in meta
