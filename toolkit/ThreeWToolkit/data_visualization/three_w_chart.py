import pandas as pd
import matplotlib.colors as mcolors
import plotly.graph_objects as go

from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from ..utils.data_utils import get_config_dataset_ini
from ..data_visualization.base_visualizer import BaseVisualizer


class ThreeWChart(BaseVisualizer):
    """A class to generate interactive visualizations for 3W dataset files using Plotly.

    Notes
    -----
    Developed by: Yan Tavares (2025) | Github: https://github.com/yantavares
    Adapted by: Matheus Ferreira (2025) | Github: https://github.com/Mathtzt
    """

    def __init__(
        self,
        file_path: str,
        title: str = "ThreeW Chart",
        y_axis: str = "P-MON-CKP",
        use_dropdown: bool = False,
        dropdown_position: tuple = (0.4, 1.4),
    ):
        """Initializes the ThreeWChart class with the given parameters.

        Args:
            file_path (str): Path to the Parquet file containing the dataset.
            title (str, optional): Title of the chart. Defaults to "ThreeW Chart".
            y_axis (str, optional): olumn name to be plotted on the y-axis. Defaults to "P-MON-CKP".
            use_dropdown (bool, optional):  Whether to show a dropdown for selecting the y-axis (default is False). Defaults to False.
            dropdown_position (tuple, optional): Position of the dropdown button on the chart. Defaults to (0.4, 1.4).
        """
        self.file_path: str = file_path
        self.title: str = title
        self.y_axis: str = y_axis
        self.use_dropdown: bool = use_dropdown
        self.dropdown_position: tuple = dropdown_position

        self.dataset_ini: dict = get_config_dataset_ini()
        self.class_mapping: dict[int, str] = self._generate_class_mapping()
        self.class_colors: dict[int, str] = self._generate_class_colors()

    def _generate_class_mapping(self) -> dict[int, str]:
        """Generate a combined mapping of event labels (including transient states) to their descriptions.

        Returns:
            dict[int, str]: Mapping of event labels to their descriptions.
        """
        return {
            **self.dataset_ini["LABELS_DESCRIPTIONS"],
            **self.dataset_ini["TRANSIENT_LABELS_DESCRIPTIONS"],
        }

    def _generate_class_colors(self) -> dict[int, str]:
        """Automatically generate a color mapping for event labels using a colormap.
        For transient states, the color is the event color with lower opacity.

        Returns:
            dict[int, str]: Mapping of event labels to their colors.
        """
        cmap = plt.get_cmap("tab10")
        colors = {}

        def apply_transparency(color: str, opacity: float) -> str:
            rgb = mcolors.to_rgb(color)
            r, g, b = int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
            return f"rgba({r}, {g}, {b}, {opacity})"

        for idx, (label, _) in enumerate(
            self.dataset_ini["LABELS_DESCRIPTIONS"].items()
        ):
            if label == 0:
                base_color = "white"
            else:
                base_color = mcolors.rgb2hex(cmap(idx % cmap.N))
            colors[label] = base_color

            transient_label = label + self.dataset_ini["TRANSIENT_OFFSET"]
            colors[transient_label] = (
                "white" if label == 0 else apply_transparency(base_color, opacity=0.4)
            )
        return colors

    def _load_data(self) -> pd.DataFrame:
        """Loads and preprocesses the dataset using the load_instance function.

        Returns:
            pd.DataFrame: Preprocessed DataFrame with sorted timestamps and no missing values.
        """
        instance = (int(Path(self.file_path).parent.name), Path(self.file_path))
        df = self._load_instance(instance)
        df.reset_index(inplace=True)
        df = df.dropna(subset=["timestamp"]).drop_duplicates("timestamp").fillna(0)
        return df.sort_values(by="timestamp")

    def _get_non_zero_columns(self, df: pd.DataFrame) -> list[str]:
        """Returns the list of columns that are not all zeros or NaN.

        Args:
            df (pd.DataFrame): DataFrame to check for non-zero columns.

        Returns:
            list[str]: List of column names that are not all zeros or NaN.
        """
        return [
            col
            for col in df.columns
            if df[col].astype(bool).sum() > 0 and col not in ["timestamp", "class"]
        ]

    def _get_background_shapes(self, df: pd.DataFrame) -> list[dict]:
        """Creates background shapes to highlight class transitions in the chart.

        Args:
            df (pd.DataFrame): DataFrame containing the class data.

        Returns:
            list[dict]: List of shape dictionaries for Plotly.
        """
        shapes = []
        prev_class = None
        start_idx = 0

        for i in range(len(df)):
            current_class = df.iloc[i]["class"]

            if pd.isna(current_class):
                print(f"Warning: NaN class value at index {i}")
                continue

            if prev_class is not None and current_class != prev_class:
                shapes.append(
                    dict(
                        type="rect",
                        x0=df.iloc[start_idx]["timestamp"],
                        x1=df.iloc[i - 1]["timestamp"],
                        y0=0,
                        y1=1,
                        xref="x",
                        yref="paper",
                        fillcolor=self.class_colors.get(prev_class, "white"),
                        opacity=0.2,
                        line_width=0,
                    )
                )
                start_idx = i

            prev_class = current_class

        if prev_class is not None:
            shapes.append(
                dict(
                    type="rect",
                    x0=df.iloc[start_idx]["timestamp"],
                    x1=df.iloc[len(df) - 1]["timestamp"],
                    y0=0,
                    y1=1,
                    xref="x",
                    yref="paper",
                    fillcolor=self.class_colors.get(prev_class, "white"),
                    opacity=0.2,
                    line_width=0,
                )
            )

        return shapes

    def _add_custom_legend(self, fig: go.Figure, present_classes: list[int]) -> None:
        """Adds a custom legend to the chart for only those classes present in the data.

        Args:
            fig (go.Figure): The Plotly figure to which the legend will be added.
            present_classes (list[int]): The unique class values present in the DataFrame.
        """
        for class_value in present_classes:
            if class_value in self.class_mapping:
                event_name = self.class_mapping[class_value]
                fig.add_trace(
                    go.Scatter(
                        x=[None],
                        y=[None],
                        mode="markers",
                        marker=dict(
                            size=12,
                            color=self.class_colors.get(class_value, "white"),
                            line=dict(width=1, color="black"),
                        ),
                        name=f"{class_value} - {event_name}",
                        showlegend=True,
                    )
                )

    def _load_instance(self, instance):
        """Loads all data and metadata from a specific `instance`.

        Args:
            instance (tuple): This tuple must refer to a specific `instance`
                and contain its label (int) and its full path (Path).

        Raises:
            Exception: Error if the Parquet file passed as arg cannot be
            read.

        Returns:
            pandas.DataFrame: Its index contains the timestamps loaded from
                the Parquet file. Its columns contain data loaded from the
                other columns of the Parquet file and metadata loaded from
                the argument `instance` (label, well, and id).
        """
        # Loads label metadata from the argument `instance`
        label, fp = instance

        try:
            # Loads well and id metadata from the argument `instance`
            well, id = fp.stem.split("_")

            # Loads data from the Parquet file
            df = pd.read_parquet(fp, engine="pyarrow")
            expected = self.dataset_ini["COLUMNS_DATA_FILES"][1:]
            if not all(df.columns == expected):
                raise ValueError(
                    f"Invalid columns in the file {fp}: {df.columns.tolist()}"
                )

        except Exception as e:
            raise Exception(f"error reading file {fp}: {e}")

        # Incorporates the loaded metadata
        df["label"] = label
        df["well"] = well
        df["id"] = id

        # Incorporates the loaded data and ordenates the df's columns
        df = df[["label", "well", "id"] + self.dataset_ini["COLUMNS_DATA_FILES"][1:]]

        return df

    def plot(self, ax=None) -> tuple[Figure, None]:
        """Generate and display an interactive Plotly chart.

        This method creates a Plotly figure based on the available data and
        configuration options. Since Plotly does not use Matplotlib axes,
        the returned Axes object is always None.

        Returns:
            Tuple[Figure, Optional[Axes]]:
                - Figure: The generated Plotly figure.
                - Axes: Always None (not applicable for Plotly).

        Raises:
            ValueError: If no valid columns are available for plotting.
        """
        df = self._load_data()

        present_classes = df["class"].dropna().unique().tolist()

        if self.use_dropdown:
            available_y_axes = self._get_non_zero_columns(df)
            if available_y_axes:
                dropdown_buttons = [
                    dict(
                        args=[{"y": [df[col]]}, {"yaxis.title": col}],
                        label=col,
                        method="update",
                    )
                    for col in available_y_axes
                ]
                fig = go.Figure()
                if self.y_axis not in available_y_axes:
                    print(
                        f"Warning: Default y-axis '{self.y_axis}' not found in available columns."
                    )
                    print("Using the first available column as the default y-axis.")
                    self.y_axis = available_y_axes[0]
                fig.add_trace(
                    go.Scatter(
                        x=df["timestamp"],
                        y=df[self.y_axis],
                        mode="lines",
                        name="Selected Variable",
                    )
                )
                active_index = available_y_axes.index(self.y_axis)
                fig.update_layout(
                    updatemenus=[
                        dict(
                            buttons=dropdown_buttons,
                            direction="down",
                            showactive=True,
                            x=self.dropdown_position[0],
                            y=self.dropdown_position[1],
                            active=active_index,
                        )
                    ]
                )
            else:
                raise ValueError("No available columns to plot.")
        else:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"], y=df[self.y_axis], mode="lines", name=self.y_axis
                )
            )

        fig.update_xaxes(rangeslider_visible=True)
        fig.update_layout(
            shapes=self._get_background_shapes(df),
            xaxis_title="Timestamp",
            yaxis_title=self.y_axis if not self.use_dropdown else df[self.y_axis].name,
            title=self.title,
            legend=dict(
                x=1.05, y=1, title="Legend", itemclick=False, itemdoubleclick=False
            ),
        )

        self._add_custom_legend(fig, present_classes)

        return fig, None
