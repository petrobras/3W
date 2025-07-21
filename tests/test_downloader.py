import json
import requests

from ThreeWToolkit.utils.downloader import FIGSHARE_BASE_URL, FIGSHARE_VERSION_IDS


class TestFigshareURLS:
    def test_known_versions(self):
        assert len(FIGSHARE_VERSION_IDS) == 4
        assert "1.0.0" in FIGSHARE_VERSION_IDS
        assert "1.1.0" in FIGSHARE_VERSION_IDS
        assert "1.1.1" in FIGSHARE_VERSION_IDS
        assert "2.0.0" in FIGSHARE_VERSION_IDS

    def get_metadata(self, version: str):
        known_files = requests.get(
            FIGSHARE_BASE_URL + "/articles/" + FIGSHARE_VERSION_IDS[version] + "/files"
        )
        metadata = json.loads(known_files.text)
        return metadata

    def test_v_1_0_0(self):
        metadata = self.get_metadata("1.0.0")
        assert len(metadata) == 1
        meta = metadata[0]

        assert meta["name"] == "3w_dataset_1.0.0.zip"
        assert meta["supplied_md5"] == "189eb151fa07cf27f82976575586df1a"

    def test_v_1_1_0(self):
        metadata = self.get_metadata("1.1.0")
        assert len(metadata) == 1
        meta = metadata[0]

        assert meta["name"] == "3w_dataset_1.1.0.zip"
        assert meta["supplied_md5"] == "e74641bb6b8a0466bb1f28c57a9163f9"

    def test_v_1_1_1(self):
        metadata = self.get_metadata("1.1.1")
        assert len(metadata) == 1
        meta = metadata[0]

        assert meta["name"] == "3w_dataset_1.1.1.zip"
        assert meta["supplied_md5"] == "ae105a8abff4ac33058fea3ebb991fc2"

    def test_v_2_0_0(self):
        metadata = self.get_metadata("2.0.0")
        assert len(metadata) == 1
        meta = metadata[0]

        assert meta["name"] == "3w_dataset_2.0.0.zip"
        assert meta["supplied_md5"] == "2c23b87b60c5d19ed9cf9559efa6ffa7"
