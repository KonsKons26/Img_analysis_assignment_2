import os
import numpy as np
import random
import pandas as pd
import cv2 as cv
from scipy.stats import pearsonr, spearmanr, kendalltau
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import seaborn as sns
import plotly.graph_objects as go
from src.utils import get_paths, read_image
from typing import List, Dict


def plot_dataset_distribution(base_path: str) -> dict[str, dict[str, int]]:
    """
    Creates a bar plot showing the distribution of samples across train, test,
    and validation sets for both normal and pneumonia classes.
    
    Parameters
    ----------
    base_path : str
        The base path where the dataset directories are located.

    Returns
    -------
    counts : dict[str, dict[str, int]]
        A dictionary containing the counts of normal and pneumonia samples in
        train, test, and validation datasets.
    """

    # Get the paths using your existing function
    paths = get_paths(base_path)

    # Count files in each directory
    counts = {
        "train": {"NORMAL": 0, "PNEUMONIA": 0},
        "test": {"NORMAL": 0, "PNEUMONIA": 0},
        "val": {"NORMAL": 0, "PNEUMONIA": 0}
    }

    for dataset in ["train", "test", "val"]:
        for label in ["NORMAL", "PNEUMONIA"]:
            path = paths[dataset][label]
            if os.path.exists(path):
                counts[dataset][label] = len([
                    f for f in os.listdir(path)
                    if os.path.isfile(os.path.join(path, f))
                ])

    # Prepare data for plotting
    datasets = list(counts.keys())
    normal_counts = [
        counts[dataset]["NORMAL"] for dataset in datasets
    ]
    pneumonia_counts = [
        counts[dataset]["PNEUMONIA"] for dataset in datasets
    ]
    total_counts = [
        normal_counts[i] + pneumonia_counts[i] for i in range(len(datasets))
    ]

    # Calculate overall totals
    total_normal = sum(normal_counts)
    total_pneumonia = sum(pneumonia_counts)
    grand_total = total_normal + total_pneumonia

    # Set up the plot
    x = np.arange(len(datasets))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    # Create bars
    bars1 = ax.bar(
        x - (width / 2), normal_counts, width, label="Normal",
        color="indianred", alpha=0.8,
        edgecolor='black', linewidth=0.5,
        zorder=2,
    )
    bars2 = ax.bar(
        x + (width / 2), pneumonia_counts, width, label="Pneumonia",
        color="dodgerblue", alpha=0.8,
        zorder=2,
        edgecolor='black', linewidth=0.5,
    )
    bars3 = ax.bar(
        x, total_counts, width * 2.25, label="Total",
        color="darkorchid", alpha=0.8,
        edgecolor='black', linewidth=0.5,
        zorder=1,
    )

    # Customize the plot
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Number of Samples")
    ax.set_title("Distribution of Samples Across Datasets")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend()

    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f'{int(height)}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center", va="bottom"
            )

    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)

    # Add grid for better readability
    ax.grid(axis='y', alpha=0.3)

    # Create text box with total counts
    textstr = f"Normal: {total_normal:,}\nPneumonia: {total_pneumonia:,}\nTotal: {grand_total:,}"

    # Create a text box in the upper right corner
    props = dict(
        boxstyle='round', facecolor='lightgray', alpha=0.8, edgecolor='black'
    )
    ax.text(
        0.98, 0.75, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', horizontalalignment='right', bbox=props
    )

    # Adjust layout and show
    plt.tight_layout()
    plt.show()

    return counts


def plot_image_grid_all_subsets(
        base_dir: str,
        n: int
    ) -> None:
    """
    Plots a grid of randomly selected images from the NORMAL and PNEUMONIA
    classes sampled across all subsets (train, test, val). Left half shows
    NORMAL, right half PNEUMONIA.

    Parameters
    ----------
    base_path : str
        Path to the dataset base directory containing 'train', 'test', and
        'val' folders.
    n : int
        Total number of images to display. Must be divisible by 2.
    """
    assert n % 2 == 0, "The number of images (n) must be divisible by 2."

    paths = get_paths(base_dir)

    # Collect all image paths for both classes across subsets
    normal_paths = []
    pneumonia_paths = []

    for subset in ['train', 'test', 'val']:
        normal_dir = paths[subset]["NORMAL"]
        pneumonia_dir = paths[subset]["PNEUMONIA"]

        normal_paths += [
            os.path.join(normal_dir, f)
            for f in os.listdir(normal_dir)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ]
        pneumonia_paths += [
            os.path.join(pneumonia_dir, f)
            for f in os.listdir(pneumonia_dir)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ]

    # Ensure enough images are available
    assert len(
        normal_paths
    ) >= n // 2, f"Not enough NORMAL images (found {len(normal_paths)})."
    assert len(
        pneumonia_paths
    ) >= n // 2, f"Not enough PNEUMONIA images (found {len(pneumonia_paths)})."

    # Randomly sample
    normal_sample = random.sample(normal_paths, n // 2)
    pneumonia_sample = random.sample(pneumonia_paths, n // 2)

    # Read images
    normal_images = [read_image(p) for p in normal_sample]
    pneumonia_images = [read_image(p) for p in pneumonia_sample]

    # Combine for plotting
    all_images = normal_images + pneumonia_images
    labels = ['NORMAL'] * (n // 2) + ['PNEUMONIA'] * (n // 2)

    # Plotting
    cols = n
    rows = 1
    if n > 6:
        cols = n // 2
        rows = 2

    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))
    axes = axes.flatten() if n > 1 else [axes]

    for ax, img, label in zip(axes, all_images, labels):
        ax.imshow(img, cmap='gray')
        ax.set_title(label, fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def plot_intensity_distributions(
        base_path: str,
        ylim: tuple[float, float]
    ) -> None:
    """
    Plots the average intensity distribution of images in the dataset,
    showing the mean histogram with ±1 standard deviation as dotted lines.

    Parameters
    ----------
    base_path : str
        The base path where the dataset directories are located.

    Returns
    -------
    None
        Plots the average intensity distribution of images in the dataset.

    Raises
    ------
    Exception
        If there is an error retrieving paths or processing images.
    """
    try:
        dataset_paths: Dict[str, Dict[str, str]] = get_paths(base_path)
    except Exception as e:
        print(f"Error during path retrieval: {e}")
        return

    all_histograms: Dict[str, List[np.ndarray]] = {
        "NORMAL": [],
        "PNEUMONIA": []
    }
    labels_to_process = ["NORMAL", "PNEUMONIA"]
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

    # --- Image loading and histogram calculation (same as before) ---
    for set_type in dataset_paths:
        for label in labels_to_process:
            label_path = dataset_paths[set_type].get(label)
            if not label_path or not os.path.isdir(label_path):
                continue
            image_files = []
            try:
                for f_name in os.listdir(label_path):
                    if f_name.lower().endswith(image_extensions):
                        image_files.append(os.path.join(label_path, f_name))
            except FileNotFoundError:
                print(
                    f"Warning: Directory '{label_path}' "
                     "not found during image listing. Skipping."
                )
                continue
            except Exception as e:
                print(f"Error listing files in '{label_path}': {e}. Skipping.")
                continue
            if not image_files: continue
            for img_path in image_files:
                try:
                    image = read_image(img_path)
                    hist = cv.calcHist([image], [0], None, [256], [0, 256])
                    sum_hist = np.sum(hist)
                    normalized_hist = hist.flatten(
                    ) / sum_hist if sum_hist > 0 else np.zeros(256)
                    all_histograms[label].append(normalized_hist)
                except FileNotFoundError:
                    print(
                        f"Warning: Image file not found at '{img_path}'. "
                        "Skipping."
                    )
                except Exception as e:
                    print(
                        f"Warning: Could not process image '{img_path}': {e}."
                         " Skipping."
                    )

    if not any(all_histograms.values()):
        print("No image data was processed. Cannot generate plot.")
        return

    plt.figure(figsize=(14, 8))
    intensity_values = np.arange(256)

    for label, histograms_list in all_histograms.items():
        print(f"\n--- Class: {label} ---")
        if not histograms_list:
            print("  No histograms collected for this class. Skipping.")
            continue

        hist_array = np.array(histograms_list)
        n_images = hist_array.shape[0]
        print(f"  Number of images (N): {n_images}")

        if n_images == 0:
            print(
                "  No images for this class after processing. "
                "Skipping plot for this class."
            )
            continue

        mean_hist = np.mean(hist_array, axis=0)
        print(
            f"  Mean histogram range: min={np.min(mean_hist):.4e}, "
            "max={np.max(mean_hist):.4e}"
        )

        effective_std_dev = np.zeros_like(mean_hist) 
        if n_images == 1:
            print(
                "  Warning: Only one image. Standard deviation is undefined "
                "(or zero); Std Dev lines will overlay mean line."
            )
        elif n_images > 1:
            std_dev_hist = np.std(hist_array, axis=0, ddof=1)
            effective_std_dev = std_dev_hist
            print(
                f"  Std Dev of normalized histograms range: "
                f"min={np.min(std_dev_hist):.4e}, "
                f"max={np.max(std_dev_hist):.4e}"
            )
            if np.all(std_dev_hist < 1e-9):
                print(
                    "  Warning: Standard deviation of histograms is zero "
                    "or extremely small for all bins."
                )

        lower_band = mean_hist - effective_std_dev
        upper_band = mean_hist + effective_std_dev
        lower_band = np.maximum(0, lower_band) 

        actual_band_width = upper_band - lower_band
        print(
            "  Distance between upper/lower Std Dev lines (Mean ± SD, "
            f"floored at 0) range: min={np.min(actual_band_width):.4e}, "
            f"max={np.max(actual_band_width):.4e}"
        )
        
        if np.all(actual_band_width < 1e-9) and n_images > 1:
             print(
                 "  Info: Calculated Std Dev is zero or extremely small. "
                 "Dotted lines will overlay or be very close to the mean line."
                )
        print("---")

        # Plot the mean line
        line, = plt.plot(
            intensity_values, mean_hist,
            label=f'{label} (Mean, N={n_images})', linewidth=2
        )
        line_color = line.get_color()

        # Plot the Mean ± 1 Standard Deviation as dotted lines
        # Label only one of these (e.g., the upper one) for a cleaner legend
        plt.plot(intensity_values, upper_band, 
                 color=line_color, linestyle=':', linewidth=1.2, 
                 label=f'{label} (±1 Std Dev)')
        plt.plot(intensity_values, lower_band, 
                 color=line_color, linestyle=':', linewidth=1.2)

    plt.title(
        'Average Intensity Distribution (Mean with ±1 Std Dev Dotted Lines)',
        fontsize=16
    )
    plt.xlabel('Pixel Intensity (0-255)', fontsize=14)
    plt.ylabel('Normalized Frequency (Density)', fontsize=14)
    plt.legend(fontsize=12)
    plt.ylim(ylim) # Adjusted to a reasonable range for visibility
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_processed_image_grid_one_class(in_path: str, n: int) -> None:
    im_paths = [
        os.path.join(in_path, f)
        for f in os.listdir(in_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    ims = random.sample(im_paths, n)
    images = [read_image(p) for p in ims]

    rows = cols = round(n**0.5)

    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))
    axes = axes.flatten() if n > 1 else [axes]
    for ax, img in zip(axes, images):
        ax.imshow(img, cmap='gray')
        ax.axis('off')
    for ax in axes[n:]:
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def plot_correlation_coefficients(
        corr_df,
        title: str = "Correlation Coefficients",
    ) -> None:
    """
    Plot the correlation coefficients of the features against the target.

    Parameters
    ----------
    corr_df : pandas.DataFrame
        The DataFrame containing the correlation coefficients. The columns of
        the DataFrame are the features and the index is the correlation
        coefficient method.
    title : str, default "Correlation Coefficients"
        The title of the plot.

    Returns
    -------
    None
        Displays the plot of the correlation coefficients.
    """

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=corr_df.columns,
        y=corr_df.loc["point_biserial"],
        mode="markers",
        marker=dict(size=12),
        name="Point Biserial"
    ))

    fig.add_trace(go.Scatter(
        x=corr_df.columns,
        y=corr_df.loc["spearman"],
        mode="markers",
        marker=dict(size=12),
        name="Spearman"
    ))

    fig.add_trace(go.Scatter(
        x=corr_df.columns,
        y=corr_df.loc["kendall"],
        mode="markers",
        marker=dict(size=12),
        name="Kendall"
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Features",
        yaxis_title="Correlation Coefficient",
        template="plotly_white",
        height=700
    )

    fig.show()


def heatmap_correlations(
        matrix,
        labels: list[str],
        title: str = "Pearson correlation Heatmap",
        figsize: tuple[int, int] = (16, 16),
        cmap: str = "crest"
    ) -> None:
    """
    Plot a heatmap of the correlations between the features in the given
    dataframe.

    Parameters
    ----------
    matrix : pd.DataFrame
        The dataframe containing the features to plot the correlations for. The
        columns of the dataframe are the features.
    labels : list of str
        The labels for the features to plot the correlations for.
    figsize : tuple of int, default (20, 20)
        The size of the figure to create for the heatmap.
    title : str, default "Correlation Heatmap"
        The title of the heatmap.
    cmap : str, default "crest"
        The colormap to use for the heatmap.

    Returns
    -------
    None
        Displays the heatmap of the correlations between the features.
    """

    plt.figure(figsize=figsize)

    sns.heatmap(
        matrix,
        annot=False,
        fmt=".2f",
        cmap=cmap,
        cbar_kws={"shrink": .8},
        linewidths=0.5,
        linecolor="black",
        square=True
    )
    plt.xticks(
        ticks=np.arange(len(labels)) + 0.5,
        labels=labels,
        rotation=45,
        ha="right"
    )
    plt.yticks(
        ticks=np.arange(len(labels)) + 0.5,
        labels=labels,
        rotation=0,
        va="center"
    )
    plt.title(title, fontsize=20)
    plt.show()


def pairplot(
        data,
        title: str,
        kde_color: str = "#421f6e",
        scatter_color: str = "#7a4db0",
        hue = None,
        cmap: str = "cool",
        height: int = 5
    ):
    """
    Plot a pairplot of the given data.

    Parameters
    ----------
    data : pd.DataFrame
        The data to plot the pairplot for. The columns of the dataframe are the
        features.
    title : str
        The title of the pairplot.
    kde_color : str
        The color to use for the KDE plots.
    scatter_color : str
        The color to use for the scatter plots.
    hue : pd.Series
        The hue variable to use for the pairplot. If None, the pairplot will
        not be colored by hue.
    cmap : str
        The colormap to use for the pairplot. If hue is None, this will be
        ignored.
    height : int, default 5
        The height of each subplot in the pairplot.

    Returns
    -------
    None
        Displays the pairplot of the given data.
    """

    def plot_mean_median(x, **kwargs):
        """Plot the mean and median of the data on the diagonal of the
        pairplot."""

        mean_val = np.mean(x)
        plt.axvline(
            mean_val,
            color="blue",
            linestyle="-",
            label=f"Mean: {mean_val:.2f}"
        )

        median_val = np.median(x)
        plt.axvline(
            median_val,
            color="red",
            linestyle="--",
            label=f"Median {median_val:.2f}"
        )
        plt.legend()


    if hue is not None:
        plot_kws = {"hue": hue, "palette": cmap, "alpha": 1, "s": 1.5}
    else:
        plot_kws = {"color": scatter_color, "alpha": 1, "s": 1.5}

    g = sns.pairplot(
        data,
        diag_kind="kde",
        plot_kws=plot_kws,
        diag_kws={"color": kde_color},
        height=height
    )

    g.map_lower(sns.kdeplot, levels=4, color="black")
    g.map_diag(plot_mean_median)

    if hue is not None:
        # Create a normalized scalar mappable for the color bar
        norm = Normalize(vmin=hue.min(), vmax=hue.max())
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])

        g.figure.subplots_adjust(right=0.85)
        cbar_ax = g.figure.add_axes([0.88, 0.15, 0.02, 0.7])
        g.figure.colorbar(sm, cax=cbar_ax, label=hue.name)

    plt.suptitle(title, y=1.02)
    plt.show()