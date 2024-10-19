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
    """Loads all configurations present in the 3W Dataset's main
    configuration file.

    Raises:
        Exception: Error if the configuration file is not found.
        Exception: Error if the configuration file cannot be loaded.

    Returns:
        dict: Dict with all configurations present in the 3W Dataset's
            main configuration file. This dict is formated with the
            basic configuration language used by the configparser
            module.
    """
    # Check if the configuration file exists in the expected path
    if not exists(PATH_DATASET_INI):
        raise Exception(
            f"the 3w Dataset's main configuration file was not found "
            f"in {PATH_DATASET_INI}"
        )

    # Load the configuration file
    dataset_ini = configparser.ConfigParser()
    dataset_ini.optionxform = lambda option: option
    try:
        dataset_ini.read(PATH_DATASET_INI)
    except Exception as e:
        raise Exception(
            f"the 3w Dataset's main configuration file "
            f"({PATH_DATASET_INI}) could not be loaded. {e}"
        )

    return dict(dataset_ini)


def load_3w_dataset(data_type='real', base_path=PATH_DATASET):
    """
    Load the 3W Dataset 2.0.

    Parameters
    ----------
    data_type : str, optional
        Type of data to be loaded ('real', 'simulated' or 'imputed').
        The default is 'real'.
    base_path : str, optional
        Path to the root folder of the dataset. The default is PATH_DATASET.

    Returns
    -------
    pandas.DataFrame
        DataFrame with the 3W Dataset 2.0 data.
    """

    dataframes = []
    for i in range(10):  # Loop through folders 0 to 9
        folder_path = os.path.join(base_path, str(i))
        if os.path.exists(folder_path):
            parquet_files = [f for f in os.listdir(folder_path) if f.endswith('.parquet')]
            for file in parquet_files:
                file_path = os.path.join(folder_path, file)
                try:
                    df = pd.read_parquet(file_path)

                    # Filter data by specified type
                    if data_type == 'real':
                        df_filtered = df[df['state'] == 0]  # Real data
                    elif data_type == 'simulated':
                        df_filtered = df[df['state'] == 1]  # Simulated data
                    elif data_type == 'imputed':
                        df_filtered = df[df['state'] == 2]  # Imputed data
                    else:
                        raise ValueError("Invalid data type. Choose between 'real', 'simulated' or 'imputed'.")

                    dataframes.append(df_filtered)
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")
        else:
            print(f"Folder {folder_path} not found.")

    # Concatenate all DataFrames into a single DataFrame
    if dataframes:
        df = pd.concat(dataframes, ignore_index=True)
        return df
    else:
        print("No data found.")
        return None


# Loads all configurations present in the 3W Dataset's main
# configuration file and provides specific configurations in different
# granularity and formats
#
DATASET_INI = load_config_in_dataset_ini()

DATASET_VERSION = DATASET_INI.get("VERSION").get("DATASET")

COLUMNS_DESCRIPTIONS = dict(DATASET_INI.get("PARQUET_FILE_PROPERTIES"))
COLUMNS_DATA_FILES = list(COLUMNS_DESCRIPTIONS.keys())
VARS = COLUMNS_DATA_FILES[1:-1]
CLASS = COLUMNS_DATA_FILES[-1]

events_section = DATASET_INI.get("EVENTS")
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

parquet_settings = DATASET_INI.get("PARQUET_SETTINGS")
PARQUET_EXTENSION = parquet_settings.get("PARQUET_EXTENSION")
PARQUET_ENGINE = parquet_settings.get("PARQUET_ENGINE")
PARQUET_COMPRESSION = parquet_settings.get("PARQUET_COMPRESSION")


# Classes
#
class EventType:
    """This class encapsulates properties (constants and default values)
    for each type of event covered by the 3W Project."""

    def __init__(self, event_name):
        """Initializes an event.

        Args:
            event_name (srt): Event type name to be initialized. This
                name must be a section name in the 3W Dataset's main
                configuration file.
        """
        event_section = DATASET_INI.get(event_name)
        self.LABEL = event_section.getint("LABEL")
        self.OBSERVATION_LABELS = EVENT_NAMES_OBSERVATION_LABELS[event_name]
        self.DESCRIPTION = event_section.get("DESCRIPTION")
        self.TRANSIENT = event_section.getboolean("TRANSIENT")
        self.window = event_section.getint("WINDOW")
        self.step = event_section.getint("STEP")
        