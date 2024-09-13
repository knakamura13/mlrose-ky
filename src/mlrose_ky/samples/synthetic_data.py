"""Class defining a synthetic dataset generator and a function to visualize a dataset."""

from os import makedirs
from typing import Any

import matplotlib.colors as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


class SyntheticData:
    """
    Class for generating synthetic datasets.

    Parameters
    ----------
    seed : int, optional, default=42
        Random seed for reproducibility.
    root_directory : str, optional, default=None
        Directory to save the generated data.
    """

    def __init__(self, seed: int = 42, root_directory: str = None):
        self.seed: int = seed
        self.root_directory: str | None = root_directory

    @staticmethod
    def get_synthetic_features_and_classes(with_redundant_column: bool = False) -> tuple[list[str], list[str]]:
        """
        Get the feature names and class labels for the synthetic dataset.

        Parameters
        ----------
        with_redundant_column : bool, optional, default=False
            Whether to include a redundant column.

        Returns
        -------
        tuple[list[str], list[str]]
            A tuple containing the list of feature names and class labels.
        """
        features = ["(1) A", "(2) B"]
        if with_redundant_column:
            features.append("(3) R")

        classes = ["RED", "BLUE"]

        return features, classes

    def get_synthetic_data(
        self, x_dim: int = 20, y_dim: int = 20, add_noise: float = 0.0, add_redundant_column: bool = False
    ) -> tuple[np.ndarray, list[str], list[str], str | None]:
        """
        Generate synthetic data.

        Parameters
        ----------
        x_dim : int, optional, default=20
            Dimension of the x-axis.
        y_dim : int, optional, default=20
            Dimension of the y-axis.
        add_noise : float, optional, default=0.0
            Amount of noise to add.
        add_redundant_column : bool, optional, default=False
            Whether to add a redundant column.

        Returns
        -------
        tuple[np.ndarray, list[str], list[str], str | None]
            A tuple containing the synthetic data, feature names, class labels, and output directory.
        """
        synthetic_data = self.__create_synthetic_data(x_dim, y_dim, add_noise, add_redundant_column)
        synthetic_data_array = synthetic_data.values

        output_directory = None
        if self.root_directory is not None:
            output_directory = (
                self.root_directory
                + f"/synthetic__sz_{x_dim * y_dim}__n_{1 if add_noise else 0}__rc_{add_redundant_column}/".lower().replace(".", "_")
            )
            try:
                makedirs(output_directory)
            except OSError:
                pass

        features, classes = self.get_synthetic_features_and_classes(add_redundant_column)

        return synthetic_data_array, features, classes, output_directory

    def setup_synthetic_data_test_train(
        self, data: np.ndarray, test_size: float = 0.30
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split the synthetic data into training and testing sets and normalize it.

        Parameters
        ----------
        data : np.ndarray
            The synthetic data.
        test_size : float, optional, default=0.30
            Proportion of the dataset to include in the test split.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            A tuple containing the normalized full dataset, labels, training data, testing data, training labels, and testing labels.
        """
        x = np.array(data[:, :-1])
        y = np.array(data[:, -1])

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=self.seed, stratify=y)

        # Normalize
        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        x = scaler.transform(x)

        return x, y, x_train, x_test, y_train, y_test

    def __create_synthetic_data(self, x_dim: int, y_dim: int, add_noise: float = 0.0, add_redundant_column: bool = False) -> pd.DataFrame:
        """
        Create synthetic data.

        Parameters
        ----------
        x_dim : int
            Dimension of the x-axis.
        y_dim : int
            Dimension of the y-axis.
        add_noise : float, optional, default=0.0
            Amount of noise to add.
        add_redundant_column : bool, optional, default=False
            Whether to add a redundant column.

        Returns
        -------
        pd.DataFrame
            The synthetic data as a DataFrame.
        """
        np.random.seed(self.seed)
        data = []

        x_mid = x_dim / 2
        y_mid = y_dim / 2
        x_mid_right_low = int(1 + x_mid + x_mid / 4)
        x_mid_right_high = int(x_mid_right_low + x_mid / 2)
        y_mid_right_low = int(1 + y_mid + y_mid / 4)
        y_mid_right_high = int(y_mid_right_low + y_mid / 2)

        x_mid_left_low = int(x_mid / 4)
        x_mid_left_high = x_mid_left_low + int(x_mid / 2)
        y_mid_left_low = int(x_mid / 4)
        y_mid_left_high = y_mid_left_low + int(y_mid / 2)

        for x in range(0, x_dim):
            for y in range(0, y_dim):
                value = 0 if (x + y) < (x_dim + y_dim) / 2 else 1
                random_value = np.random.random(1)[0]
                if (x_mid_right_low < x < x_mid_right_high and y_mid_right_low < y < y_mid_right_high) or (
                    x_mid_left_low < x < x_mid_left_high and y_mid_left_low < y < y_mid_left_high
                ):
                    data.append([x, y, random_value, 1 - value])
                else:
                    data.append([x, y, random_value, value])

        if add_noise > 0:
            noise_count = int((add_noise * x_dim * y_dim) + 0.5)
            # flip some point values
            for _ in range(0, noise_count):
                x = np.random.randint(x_dim)
                y = np.random.randint(y_dim)
                xy = y + y_dim * x
                abrc = data[xy]
                random_value = np.random.random(1)[0]
                data[xy] = [x, y, random_value, 1 - abrc[-1]]

            # duplicate some rows and randomly flip the data for those rows
            for _ in range(0, noise_count * 2):
                x = np.random.randint(x_dim)
                y = np.random.randint(y_dim)
                random_value = np.random.random(1)[0]
                xy = y + y_dim * x
                abrc = data[xy]
                value = abrc[-1]
                random_value_old = abrc[-2]
                data.append([x, y, random_value_old, value if random_value < 0.5 else 1 - value])

        df = pd.DataFrame.from_records(data)
        df.rename(columns={0: "A", 1: "B", 2: "R", 3: "C"}, errors="raise", inplace=True)
        if not add_redundant_column:
            df.drop(columns=["R"], inplace=True)

        return df


def plot_synthetic_dataset(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    classifier: Any = None,
    transparent_bg: bool = False,
    bg_color: str = "white",
):
    """
    Plot the synthetic dataset.

    Parameters
    ----------
    x_train : np.ndarray
        Training data.
    x_test : np.ndarray
        Testing data.
    y_train : np.ndarray
        Training labels.
    y_test : np.ndarray
        Testing labels.
    classifier : Any, optional, default=None
        Classifier to plot decision boundary.
    transparent_bg : bool, optional, default=False
        Whether to make the background transparent.
    bg_color : str, optional, default="white"
        Background color.
    """
    offset = 0.05

    x_min_train, x_max_train = x_train[:, 0].min() - offset, x_train[:, 0].max() + offset
    y_min_train, y_max_train = x_train[:, 1].min() - offset, x_train[:, 1].max() + offset
    x_min_test, x_max_test = x_test[:, 0].min() - offset, x_test[:, 0].max() + offset
    y_min_test, y_max_test = x_test[:, 1].min() - offset, x_test[:, 1].max() + offset

    x_min = min(x_min_test, x_min_train)
    y_min = max(y_min_test, y_min_train)
    x_max = min(x_max_test, x_max_train)
    y_max = max(y_max_test, y_max_train)

    has_3_columns = x_train.shape[1] == 3

    h = 0.02
    rr = None
    if has_3_columns:
        xx, yy, rr = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h), np.arange(0, 1, h))
    else:
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Plot the dataset first
    cm = plt.get_cmap("RdBu")
    cm_bright = mpl.ListedColormap(["#FF0000", "#0000FF"])

    ax = plt.gca()
    if classifier is not None:
        dd = np.c_[xx.ravel(), yy.ravel()] if not has_3_columns else np.c_[xx.ravel(), yy.ravel(), rr.ravel()]
        if hasattr(classifier, "decision_function"):
            Z = classifier.decision_function(dd)
        else:
            Z = classifier.predict_proba(dd)[:, 1]

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        if has_3_columns:
            xx = xx[:, :, 0]
            yy = yy[:, :, 0]
            Z = Z.mean(axis=2)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=0.8)

    # Plot the training and testing points
    ax.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=cm_bright, edgecolor="darkgreen")
    ax.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap=cm_bright, edgecolor="white", alpha=0.6)

    ax.patch.set_facecolor(bg_color)
    if transparent_bg:
        ax.patch.set_alpha(0)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())

    plt.show()
