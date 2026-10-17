"""Frozen parameters of protocol 3W-HW v1.

The values were fixed before the split was drawn (see
``protocol/COMMIT-A.md``). Changing any of them defines a new protocol
version.
"""

from __future__ import annotations

PROTOCOL_ID = "3W-HW-v1"

# Decision grid, in rows of the 1 Hz series.
STRIDE = 30
FIRST_ROW = 3600
LOOKBACK = 3600
ELIG_WINDOW = 900
ELIG_FRAC = 0.9

# Episodes, alarms and onsets.
MIN_EPISODE_ROWS = 300
ONSET_TOL = 300
ON_K = 2
OFF_K = 2

# Uncertainty: bootstrap over wells.
BOOT_B = 2000
BOOT_SEED = 611

# Split.
N_TEST_WELLS = 10
N_DEV_FOLDS = 5

# Labels.
N_CLASSES = 10
TRANSIENT_OFFSET = 100
NA = -1
CLASS_NAMES = {
    0: "normal",
    1: "abrupt_increase_bsw",
    2: "spurious_closure_dhsv",
    3: "severe_slugging",
    4: "flow_instability",
    5: "rapid_productivity_loss",
    6: "quick_restriction_pck",
    7: "scaling_pck",
    8: "hydrate_production_line",
    9: "hydrate_service_line",
}

# Event scores are quantised to this many logit bins when they are
# summed per well, so that a bootstrap replicate costs milliseconds.
NBINS = 4096
LOGIT_CLIP = 12.0

# The path prefix used in the committed label index.
DATASET_DIR = "dataset"

PROCESS_CHANNELS = [
    "P-ANULAR",
    "P-JUS-BS",
    "P-JUS-CKGL",
    "P-JUS-CKP",
    "P-MON-CKGL",
    "P-MON-CKP",
    "P-MON-SDV-P",
    "P-PDG",
    "PT-P",
    "P-TPT",
    "QBS",
    "QGL",
    "T-JUS-CKP",
    "T-MON-CKP",
    "T-PDG",
    "T-TPT",
    "ABER-CKGL",
    "ABER-CKP",
]

STATE_CHANNELS = [
    "ESTADO-DHSV",
    "ESTADO-M1",
    "ESTADO-M2",
    "ESTADO-PXO",
    "ESTADO-SDV-GL",
    "ESTADO-SDV-P",
    "ESTADO-W1",
    "ESTADO-W2",
    "ESTADO-XO",
]
