import numpy as np
from sklearn.model_selection import BaseCrossValidator, StratifiedKFold


class StratifiedGroupKFold(BaseCrossValidator):
    """
    GroupKFold with stratification based on the event type.

    * Constructor arguments:
        - **n_splits: int** -- Number of folds   

    * Methods:

    - **split()**

    - **get_n_splits()**
    """

    def __init__(self, n_splits, event_types):
        self.base_splitter = StratifiedKFold(n_splits)
        self.event_types = event_types

    def split(self, X, y, groups):
        """
        Create the splits.

        * Parameters:
            - **X: np.ndarray** - Data

            - **y: np.array** - Labels

            - **groups: np.array** - Groups

        * Yields:
            - **splits**: [TUPLE] - Tuple with the index of training and test samples for each fold.
        """
        unique_g = np.unique(groups)
        event_y = np.array([self.event_types[_] for _ in unique_g])
        indices = np.arange(groups.size)

        # stratified split of events
        for _, test_gidx in self.base_splitter.split(unique_g, event_y):
            test_groups = unique_g[test_gidx]
            test_mask = np.isin(groups, test_groups)
            yield indices[~test_mask], indices[test_mask]

    def get_n_splits(self, X, y, groups):
        return self.base_splitter.n_splits
