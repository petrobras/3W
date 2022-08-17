"""This 3W toolkits' sub-module groups objects used by the other 
sub-modules. 

Any resource that is not used by another sub-module must be maintained 
in the miscellaneous sub-module.
"""

import configparser
import os
from os.path import exists
from pathlib import Path

# Important paths
#
PATH_3W_PROJECT = Path(__file__).parents[1]
PATH_TOOLKIT = os.path.join(PATH_3W_PROJECT, "src", "toolkit")
PATH_DATASET = os.path.join(PATH_3W_PROJECT, "dataset")
PATH_FOLDS = os.path.join(PATH_DATASET, "folds")
PATH_DATASET_INI = os.path.join(PATH_DATASET, "dataset.ini")

# Methods
#
def load_config_in_dataset_ini():
    """Loads all configurations present in the 3W dataset's main
    configuration file.

    Raises:
        Exception: Error if the configuration file is not found.
        Exception: Error if the configuration file cannot be loaded.

    Returns:
        dict: Dict with all configurations present in the 3W dataset's
            main configuration file. This dict is formated with the
            basic configuration language used by the configparser
            module.
    """
    # Check if the configuration file exists in the expected path
    if not exists(PATH_DATASET_INI):
        raise Exception(
            f"the 3w dataset's main configuration file was not found "
            f"in {PATH_DATASET_INI}"
        )

    # Load the configuration file
    dataset_ini = configparser.ConfigParser()
    dataset_ini.optionxform = lambda option: option
    try:
        dataset_ini.read(PATH_DATASET_INI)
    except Exception as e:
        raise Exception(
            f"the 3w dataset's main configuration file "
            f"({PATH_DATASET_INI}) could not be loaded. {e}"
        )

    return dict(dataset_ini)


# Loads all configurations present in the 3W dataset's main
# configuration file and provides specific configurations in different
# granularity and formats
#
DATASET_INI = load_config_in_dataset_ini()

DATASET_VERSION = DATASET_INI.get("Versions").get("DATASET")

COLUMNS_DESCRIPTIONS = dict(DATASET_INI.get("Columns of CSV Data Files"))
COLUMNS_DATA_FILES = list(COLUMNS_DESCRIPTIONS.keys())
VARS = COLUMNS_DATA_FILES[1:-1]
CLASS = COLUMNS_DATA_FILES[-1]

events_section = DATASET_INI.get("Events")
EVENT_NAMES = [n.strip() for n in events_section.get("NAMES").split(",")]
EXTRA_INSTANCES_TRAINING = events_section.getint("EXTRA_INSTANCES_TRAINING")
TRANSIENT_OFFSET = events_section.getint("TRANSIENT_OFFSET")

NORMAL_LABEL = DATASET_INI.get("NORMAL").getint("LABEL")

LABELS_DESCRIPTIONS = {}
EVENT_NAMES_LABELS = {}
EVENT_NAMES_DESCRIPTIONS = {}
EVENT_NAMES_OBSERVATION_LABELS = {}
for n in EVENT_NAMES:
    s = DATASET_INI.get(n)
    l = s.getint("LABEL")
    d = s.get("DESCRIPTION")
    LABELS_DESCRIPTIONS[l] = d
    EVENT_NAMES_LABELS[n] = l
    EVENT_NAMES_DESCRIPTIONS[n] = d
    if s.getboolean("TRANSIENT"):
        EVENT_NAMES_OBSERVATION_LABELS[n] = {
            NORMAL_LABEL,
            l,
            l + TRANSIENT_OFFSET,
        }
    else:
        EVENT_NAMES_OBSERVATION_LABELS[n] = {NORMAL_LABEL, l}

# Classes
#
class EventType:
    """This class encapsulates properties (constants and default values)
    for each type of event covered by the 3W project."""

    def __init__(self, event_name):
        """Initializes an event.

        Args:
            event_name (srt): Event type name to be initialized. This
                name must be a section name in the 3W dataset's main
                configuration file.
        """
        event_section = DATASET_INI.get(event_name)
        self.LABEL = event_section.getint("LABEL")
        self.OBSERVATION_LABELS = EVENT_NAMES_OBSERVATION_LABELS[event_name]
        self.DESCRIPTION = event_section.get("DESCRIPTION")
        self.TRANSIENT = event_section.getboolean("TRANSIENT")
        self.window = event_section.getint("WINDOW")
        self.step = event_section.getint("STEP")
