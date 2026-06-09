import configparser

from ThreeWToolkit.utils.data_utils import (
    load_config_in_dataset_ini,
    get_config_dataset_ini,
    UNUSED_TAGS,
)


class TestUnusedTags:
    def test_unused_tags_list(self):
        """Test that UNUSED_TAGS constant contains expected values."""
        assert isinstance(UNUSED_TAGS, list)
        assert "P-JUS-BS" in UNUSED_TAGS
        assert "P-MON-SDV-P" in UNUSED_TAGS
        assert "PT-P" in UNUSED_TAGS
        assert "QBS" in UNUSED_TAGS
        assert "P-MON-CKGL" in UNUSED_TAGS
        assert "state" in UNUSED_TAGS


class TestLoadConfigInDatasetIni:
    def test_load_config_success(self):
        """Test successful loading of dataset.ini configuration file."""
        config = load_config_in_dataset_ini()

        assert isinstance(config, dict)
        assert len(config) > 0
        assert "PARQUET_FILE_PROPERTIES" in config
        assert "EVENTS" in config

    def test_config_returns_section_proxy(self):
        """Test that configuration sections are SectionProxy objects."""
        config = load_config_in_dataset_ini()

        assert isinstance(config["PARQUET_FILE_PROPERTIES"], configparser.SectionProxy)
        assert isinstance(config["EVENTS"], configparser.SectionProxy)


class TestGetConfigDatasetIni:
    def test_get_config_returns_expected_keys(self):
        """Test that get_config_dataset_ini returns all expected keys."""
        config = get_config_dataset_ini()

        assert isinstance(config, dict)
        assert "COLUMNS_DESCRIPTIONS" in config
        assert "TRANSIENT_OFFSET" in config
        assert "COLUMNS_DATA_FILES" in config
        assert "LABELS_DESCRIPTIONS" in config
        assert "TRANSIENT_LABELS_DESCRIPTIONS" in config

    def test_columns_descriptions_is_dict(self):
        """Test that COLUMNS_DESCRIPTIONS is a dictionary."""
        config = get_config_dataset_ini()
        columns_desc = config["COLUMNS_DESCRIPTIONS"]

        assert isinstance(columns_desc, dict)
        assert len(columns_desc) > 0

    def test_transient_offset_is_int(self):
        """Test that TRANSIENT_OFFSET is an integer."""
        config = get_config_dataset_ini()
        transient_offset = config["TRANSIENT_OFFSET"]

        assert isinstance(transient_offset, int)
        assert transient_offset > 0

    def test_columns_data_files_is_list(self):
        """Test that COLUMNS_DATA_FILES is a list."""
        config = get_config_dataset_ini()
        columns_files = config["COLUMNS_DATA_FILES"]

        assert isinstance(columns_files, list)
        assert len(columns_files) > 0
        assert all(isinstance(col, str) for col in columns_files)

    def test_labels_descriptions_is_dict_with_int_keys(self):
        """Test that LABELS_DESCRIPTIONS is a dict with integer keys."""
        config = get_config_dataset_ini()
        labels_desc = config["LABELS_DESCRIPTIONS"]

        assert isinstance(labels_desc, dict)
        assert len(labels_desc) > 0
        assert all(isinstance(k, int) for k in labels_desc.keys())
        assert all(isinstance(v, str) for v in labels_desc.values())

    def test_transient_labels_descriptions_format(self):
        """Test that TRANSIENT_LABELS_DESCRIPTIONS has correct format."""
        config = get_config_dataset_ini()
        transient_desc = config["TRANSIENT_LABELS_DESCRIPTIONS"]

        assert isinstance(transient_desc, dict)
        # All values should start with "Transient: "
        for value in transient_desc.values():
            assert value.startswith("Transient: ")

    def test_transient_label_offset_applied(self):
        """Test that transient labels have offset applied."""
        config = get_config_dataset_ini()
        labels_desc = config["LABELS_DESCRIPTIONS"]
        transient_desc = config["TRANSIENT_LABELS_DESCRIPTIONS"]
        offset = config["TRANSIENT_OFFSET"]

        # If there are transient labels, check offset is applied
        if len(transient_desc) > 0:
            # Transient label keys should be original label + offset
            for transient_label in transient_desc.keys():
                original_label = transient_label - offset
                # The original label should exist in labels_descriptions
                assert original_label in labels_desc

    def test_columns_consistency(self):
        """Test that COLUMNS_DATA_FILES matches COLUMNS_DESCRIPTIONS keys."""
        config = get_config_dataset_ini()
        columns_files = config["COLUMNS_DATA_FILES"]
        columns_desc = config["COLUMNS_DESCRIPTIONS"]

        # The list should contain the same columns as the dict keys
        assert set(columns_files) == set(columns_desc.keys())
