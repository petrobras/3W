"""This is the 3W toolkit, a software package written in Python 3 that 
is one of the 3W project's major components.

This toolkit contains resources that make the following easier:

- 3W dataset overview generation;
- Experimentation and comparative analysis of Machine Learning-based 
approaches and algorithms for specific problems related to undesirable 
events that occur in offshore oil wells during their respective 
production phases;
* Standardization of key points of the Machine Learning-based algorithm 
development pipeline.

All these resources are implemented in the following sub-modules:

- **base**: groups the objects used by the other sub-modules;
- **dev**: has all the resources related to development of Machine 
Learning models;
- **misc**: brings together diverse resources that do not fit in the 
other sub-modules;
- **rolling_window**: creates a view of array which for every point 
gives the n-dimensional neighbourhood of size window. New dimensions are 
added at the end of array or after the corresponding original dimension.

Specific problems will be incorporated into this toolkit gradually. At 
this time, models can be developed for the following problems:

- Binary Classifier of Spurious Closure of DHSV.

Examples of how to use this toolkit will be incremented throughout its 
development. Please, check the project's README.md file for more details.

It is important to note that there are arbitrary choices in this 
toolkit, but they have been carefully made to allow adequate comparative 
analysis without compromising the ability to experiment with different 
approaches and algorithms.

This toolkit's documentation is generated in english and in Google format 
with [autoDocstring - Python Docstring Generator
](https://github.com/NilsJPWerner/autoDocstring), which follows [PEP 257
](https://peps.python.org/pep-0257/), and [pdoc3
](https://pdoc3.github.io/pdoc/).

Its source code is implemented according to the style guide established 
by [PEP 8](https://peps.python.org/pep-0008/). This is guaranteed with 
the use of the [Black formatter](https://github.com/psf/black).
"""

__status__ = "Development"
__version__ = "1.0"
__license__ = "Apache License 2.0"
__copyright__ = "Copyright 2022, Petróleo Brasileiro S.A."
__authors__ = [
    "Ricardo Emanuel Vaz Vargas",
    "Bruno Guberfain do Amaral",
    "Jean Carlos Dias de Araújo",
    "Lucas Pierezan Magalhães",
]
__maintainer__ = ["Ricardo Emanuel Vaz Vargas <ricardo.vargas@petrobras.com.br>"]

# Imports and exposes objects from the 'base' sub-module
from .base import (
    CLASS,
    COLUMNS_DATA_FILES,
    COLUMNS_DESCRIPTIONS,
    DATASET_INI,
    DATASET_VERSION,
    EVENT_NAMES,
    EVENT_NAMES_DESCRIPTIONS,
    EVENT_NAMES_LABELS,
    EVENT_NAMES_OBSERVATION_LABELS,
    EXTRA_INSTANCES_TRAINING,
    EventType,
    LABELS_DESCRIPTIONS,
    NORMAL_LABEL,
    PATH_3W_PROJECT,
    PATH_DATASET,
    PATH_DATASET_INI,
    PATH_FOLDS,
    PATH_TOOLKIT,
    TRANSIENT_OFFSET,
    VARS,
    load_config_in_dataset_ini,
)

# Imports and exposes objects from the 'dev' sub-module
from .dev import (
    Experiment,
    EventFold,
    EventFolds,
)

# Imports and exposes objects from the 'misc' sub-module
from .misc import (
    calc_stats_instances,
    count_properties_instance,
    count_properties_instances,
    create_and_plot_scatter_map,
    create_table_of_instances,
    filter_rare_undesirable_events,
    get_all_labels_and_files,
    label_and_file_generator,
    load_instance,
    load_instances,
)
