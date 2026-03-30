import configparser
from pathlib import Path
from ..constants import DATASET_INI_2_0_0

"""Signals in dataset with few samples (or no samples at all)."""
UNUSED_TAGS = [
    "P-JUS-BS",  # zero instances
    "P-MON-SDV-P",  # zero instances
    "PT-P",  # zero instances
    "QBS",  # zero instances
    "P-MON-CKGL",  # only two events have this tag non-NA
    "state",
]


def load_config_in_dataset_ini() -> dict[str, configparser.SectionProxy]:
    """Loads all configurations present in the 3W Dataset's main configuration file.

    Raises:
        Exception: Error if the configuration file is not found.
        Exception: Error if the configuration file cannot be loaded.

    Returns:
        dict: Dict with all configurations present in the 3W Dataset's main configuration file. This dict is formated with the basic configuration language used by the configparser module.
    """
    # Check if the configuration file exists in the expected path
    if not DATASET_INI_2_0_0.exists():
        raise Exception(
            f"the 3w Dataset's main configuration file was not found "
            f"in {Path(DATASET_INI_2_0_0)}"
        )

    # Load the configuration file
    dataset_ini = configparser.ConfigParser()
    setattr(dataset_ini, "optionxform", str)
    try:
        dataset_ini.read(Path(DATASET_INI_2_0_0))
    except Exception as e:
        raise Exception(
            f"the 3w Dataset's main configuration file "
            f"({Path(DATASET_INI_2_0_0)}) could not be loaded. {e}"
        )

    return dict(dataset_ini)


def get_config_dataset_ini() -> dict[
    str, dict[str, str] | int | list[str] | dict[int, str]
]:
    """
    Load and process dataset configuration from the ``dataset.ini`` file.

    This function reads the configuration entries related to dataset columns,
    event labels, transient event offset, and other metadata defined in the
    ``dataset.ini`` file. It builds and returns structured dictionaries
    containing column descriptions, label mappings, and transient label
    mappings used throughout the dataset processing pipeline.

    Returns:
        dict: A dictionary with the following keys:
            - COLUMNS_DESCRIPTIONS (dict): Maps each column name to its textual
            description.
            - TRANSIENT_OFFSET (int): Offset applied to transient event labels.
            - COLUMNS_DATA_FILES (list): List of column names found in the
            dataset configuration.
            - LABELS_DESCRIPTIONS (dict): Maps each event label (int) to its
            description (str).
            - TRANSIENT_LABELS_DESCRIPTIONS (dict): Maps each transient event
            label (label + offset) to its description.

    Notes:
        The function expects the following sections and keys to exist inside
        ``dataset.ini`` of data version 2.0.0.
    """
    dt_ini = load_config_in_dataset_ini()

    # Ensure required sections exist (mypy-safe: get() may return None)
    parquet_section = dt_ini.get("PARQUET_FILE_PROPERTIES")
    if parquet_section is None:
        raise KeyError(
            "Missing required section 'PARQUET_FILE_PROPERTIES' in dataset.ini."
        )

    events_section = dt_ini.get("EVENTS")
    if events_section is None:
        raise KeyError("Missing required section 'EVENTS' in dataset.ini.")

    COLUMNS_DESCRIPTIONS = dict(parquet_section)
    TRANSIENT_OFFSET = int(events_section["TRANSIENT_OFFSET"])
    COLUMNS_DATA_FILES = list(COLUMNS_DESCRIPTIONS.keys())
    LABELS_DESCRIPTIONS: dict[int, str] = {}
    TRANSIENT_LABELS_DESCRIPTIONS: dict[int, str] = {}

    names_value = events_section.get("NAMES")
    if not names_value:
        raise KeyError("The 'EVENTS' section is missing the 'NAMES' key or it is empty")

    for name in (n.strip() for n in names_value.split(",")):
        # Accessing by index guarantees SectionProxy type (KeyError if absent)
        if name not in dt_ini:
            raise KeyError(f"Missing section for event name '{name}' in dataset.ini")
        n_ = dt_ini[name]

        # Validate presence of mandatory keys
        if n_ is None:
            raise KeyError(f"Section '{name}' missing in dataset.ini")

        if "LABEL" not in n_ or "DESCRIPTION" not in n_:
            raise KeyError(f"Section '{name}' must define 'LABEL' and 'DESCRIPTION'")

        label = int(n_["LABEL"])
        description = n_["DESCRIPTION"]
        LABELS_DESCRIPTIONS[label] = description

        # getboolean exists on SectionProxy; default False if key missing
        try:
            is_transient = n_.getboolean("TRANSIENT")
        except (ValueError, configparser.NoOptionError):
            is_transient = False

        if is_transient:
            TRANSIENT_LABELS_DESCRIPTIONS[label + TRANSIENT_OFFSET] = (
                f"Transient: {description}"
            )

    return {
        "COLUMNS_DESCRIPTIONS": COLUMNS_DESCRIPTIONS,
        "TRANSIENT_OFFSET": TRANSIENT_OFFSET,
        "COLUMNS_DATA_FILES": COLUMNS_DATA_FILES,
        "LABELS_DESCRIPTIONS": LABELS_DESCRIPTIONS,
        "TRANSIENT_LABELS_DESCRIPTIONS": TRANSIENT_LABELS_DESCRIPTIONS,
    }
