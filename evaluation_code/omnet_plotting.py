"""Plotting utilities and experiment plotting script for OMNeT++ post-processing results.

The file contains reusable plotting helpers at the top and the concrete plotting
workflows below. The helpers are intentionally kept independent from the
experiment sections so they can also be imported from notebooks or smaller
analysis scripts.
"""

from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# The plotting script is intentionally self-contained. The helper functions below
# are copied from the post-processing workflow where they are also useful for
# plotting-only analysis. This avoids importing the full post-processing module
# just to load, filter, and visualize already generated result CSV files.

# -----------------------------------------------------------------------------
# Plot labels and global configuration
# -----------------------------------------------------------------------------

#      ____                                _            
#     |  _ \ __ _ _ __ __ _ _ __ ___   ___| |_ ___ _ __ 
#     | |_) / _` | '__/ _` | '_ ` _ \ / _ \ __/ _ \ '__|
#     |  __/ (_| | | | (_| | | | | | |  __/ ||  __/ |   
#     |_|   \__,_|_|  \__,_|_| |_| |_|\___|\__\___|_|   
#    
#                              

DISTRIBUTED_HYRASEL_DESCRIPTOR = "HyRASeL (Distr.)"
CENTRALIZED_HYRASEL_DESCRIPTOR = "HyRASeL (Centr.)"
DISTRIBUTED_VTWR_DESCRIPTOR = "v-TWR (Distr.)"
CENTRALIZED_VTWR_DESCRIPTOR = "v-TWR (Centr.)"
TWR_DESCRIPTOR = "TWR (Baseline)"
CHIASSON_DESCRIPTOR = "Chiasson et al. (Baseline)"

ACCURACY_DESCRIPTOR = "Localization Error [m]"
COVERAGE_DESCRIPTOR = "Coverage Fraction"
DURATION_DESCRIPTOR = "Absolute Localization Duration Per Node [s]"
DURATION_RATIO_DESCRIPTOR = "Localization Service Ratio"
INTER_UPDATE_TIME_DESCRIPTOR = "Inter-Update Interval [s]"
FREQUENCY_DESCRIPTOR = "Update Frequency [Hz]"
NUM_NODES_DESCRIPTOR = "Number of Nodes"
NUM_SLOTS_DESCRIPTOR = "Number of v-Anchor Slots"
DENSITY_DESCRIPTOR = "Inter-Anchor Distance [m]"
INSTANT_FREQUENCY_DESCRIPTOR = "Instantaneous Update Frequency [Hz]"
CDF_DESCRIPTOR = "CDF"

TIMESTAMP_FORMAT = "noisy_timestamp"
K = 3
DEFAULT_SPEED = 25

FONT_SIZE = 9
COL_WIDTH = 3.5
COL_WIDTH_2 = 7.16
ROW_HEIGHT = COL_WIDTH / 1.2

FIGURE_DIR = Path("tmp/figs")

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE - 1,
    "ytick.labelsize": FONT_SIZE - 1,
    "legend.fontsize": FONT_SIZE,
})


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------

def _build_timestamped_output_path(save_path: str, extension: str = "pdf") -> Path:
    """ Build a timestamped output path for generated figures

    Args:
        save_path (str): Base name of the output file without timestamp or extension
        extension (str): File extension used for the saved figure

    Returns:
        Path: Output path in the configured figure directory
    """
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%y%m%d_%H%M")
    return FIGURE_DIR / f"{timestamp}_{save_path}.{extension}"


def _apply_axis_options(
    ax: plt.Axes,
    xlabel: str | None = None,
    ylabel: str | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    xticks: list[float] | None = None,
    yticks: list[float] | None = None,
    xlog: bool = False,
    ylog: bool = False,
) -> None:
    """ Apply common axis labels, scales, limits, ticks, and grid settings

    Args:
        ax (plt.Axes): Matplotlib axis to configure
        xlabel (str | None): Label for the x-axis
        ylabel (str | None): Label for the y-axis
        xlim (tuple[float, float] | None): Optional x-axis limits
        ylim (tuple[float, float] | None): Optional y-axis limits
        xticks (list[float] | None): Optional x-axis tick positions
        yticks (list[float] | None): Optional y-axis tick positions
        xlog (bool): Whether to use a logarithmic x-axis
        ylog (bool): Whether to use a logarithmic y-axis

    Returns:
        None
    """
    ax.set_xlabel(xlabel or "", fontsize=FONT_SIZE)
    ax.set_ylabel(ylabel or "", fontsize=FONT_SIZE)
    ax.tick_params(labelsize=FONT_SIZE - 1)
    ax.grid(True, alpha=0.3)

    if xlog:
        ax.set_xscale("log")
    if ylog:
        ax.set_yscale("log")
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)


def _add_legend(
    ax: plt.Axes,
    legend_title: str | None = None,
    legend_ncol: int = 2,
    legend_above: bool = True,
) -> None:
    """ Add a consistently styled legend to an axis

    Args:
        ax (plt.Axes): Matplotlib axis that contains labeled artists
        legend_title (str | None): Optional legend title
        legend_ncol (int): Number of legend columns
        legend_above (bool): Whether to place the legend above the plot area

    Returns:
        None
    """
    if legend_above:
        ax.legend(
            title=legend_title,
            ncol=legend_ncol,
            bbox_to_anchor=(0.5, 1.0),
            loc="lower center",
            columnspacing=0.4,
            handletextpad=0.3,
            borderpad=0.3,
            frameon=True,
            fontsize=FONT_SIZE,
        )
    else:
        ax.legend(title=legend_title, ncol=legend_ncol, fontsize=FONT_SIZE)



def compute_inter_measurement_time(
    df: pd.DataFrame,
    filter_col: str,
    timestamp_col: str = "sim_timestamp",
) -> pd.DataFrame:
    """ Compute inter-measurement time and update frequency for valid localization rows

    Args:
        df (pd.DataFrame): Input dataframe containing localization results
        filter_col (str): Column used to select rows with valid localization results
        timestamp_col (str): Timestamp column used to compute time differences

    Returns:
        pd.DataFrame: Filtered dataframe with inter_measurement_time and update_frequency columns
    """
    result = df.loc[df[filter_col].notna()].copy()
    result["inter_measurement_time"] = result[timestamp_col].diff()
    result["update_frequency"] = 1 / result["inter_measurement_time"]
    return result.loc[result["inter_measurement_time"] > 0].copy()


def load_results_from_csv(
    files: list[str],
    folder: str = "tmp",
    append_parameters: list[Any] | None = None,
    append_value: str | None = None,
) -> pd.DataFrame:
    """ Load one or more result CSV files and concatenate them into one dataframe

    Args:
        files (list[str]): File stems without .csv extension
        folder (str): Directory that contains the result CSV files
        append_parameters (list[Any] | None): Values appended to each loaded file, if provided
        append_value (str | None): Column name used for appended parameter values

    Returns:
        pd.DataFrame: Concatenated dataframe containing all successfully loaded files
    """
    dataframes = []
    file_parameter_pairs = zip(files, append_parameters) if append_parameters else ((file, None) for file in files)

    for file_stem, appended_parameter in file_parameter_pairs:
        file_path = Path(folder) / f"{file_stem}.csv"
        if not file_path.exists():
            print(f"Warning: {file_path} not found, skipping.")
            continue

        data_df = pd.read_csv(file_path)

        if append_parameters is not None and append_value is not None:
            data_df[append_value] = appended_parameter

        if "residuals" in data_df.columns:
            data_df["residuals"] = data_df["residuals"].apply(
                lambda value: np.fromstring(str(value).strip("[]"), sep=" ") if pd.notna(value) else None
            )

        dataframes.append(data_df)

    if not dataframes:
        return pd.DataFrame()

    result_df = pd.concat(dataframes, ignore_index=True)
    print(f"Loaded {files} in {len(result_df)} rows")
    return result_df


def merge_exclusive_columns(
    df: pd.DataFrame,
    col1: str = "self_loc_error",
    col2: str = "v_twr_error",
    new_col: str = "hyrasel_pos_error",
) -> pd.DataFrame:
    """ Merge two mutually exclusive value columns into a single output column

    Args:
        df (pd.DataFrame): Input dataframe containing both source columns
        col1 (str): First source column
        col2 (str): Second source column
        new_col (str): Output column that receives the merged values

    Returns:
        pd.DataFrame: Copy of the input dataframe with the merged output column

    Raises:
        KeyError: If one of the source columns is missing
        ValueError: If a row contains values in both source columns
    """
    for column in [col1, col2]:
        if column not in df.columns:
            raise KeyError(f"Column '{column}' not found in dataframe")

    overlap_mask = df[col1].notna() & df[col2].notna()
    if overlap_mask.any():
        raise ValueError(
            f"{overlap_mask.sum()} row(s) contain values in both '{col1}' and '{col2}'"
        )

    result = df.copy()
    result[new_col] = result[col1].combine_first(result[col2])
    return result


def filter_residual_outliers_per_parameter(
    df: pd.DataFrame,
    group_cols: list[str],
    residuals_column: str = "residuals",
    k: float = 3.0,
    verbose: bool = True,
) -> pd.DataFrame:
    """ Remove residual outliers independently for each parameter group

    Args:
        df (pd.DataFrame): Input dataframe containing residual vectors
        group_cols (list[str]): Columns used to group rows before robust filtering
        residuals_column (str): Column containing residual arrays or residual-like values
        k (float): Robust sigma multiplier used as the outlier threshold
        verbose (bool): Whether to print filtering statistics

    Returns:
        pd.DataFrame: Dataframe with residual outliers removed and rows without residuals preserved
    """
    df = df.copy()

    if residuals_column not in df.columns:
        if verbose:
            print(f"Column '{residuals_column}' not found; residual filtering skipped.")
        return df

    valid_mask = df[residuals_column].notna()
    df_valid = df.loc[valid_mask]
    df_other = df.loc[~valid_mask]

    filtered_groups = []
    removed_per_group = []
    n_valid_total = len(df_valid)
    n_removed_total = 0

    for group_key, group in df_valid.groupby(group_cols):
        if group.empty:
            removed_per_group.append((group_key, 0, 0))
            continue

        all_residuals = np.hstack([np.ravel(row) for row in group[residuals_column]])
        median = np.median(all_residuals)
        mad = np.median(np.abs(all_residuals - median))
        sigma_robust = 1.4826 * mad

        if sigma_robust == 0:
            keep_mask = pd.Series(True, index=group.index)
        else:
            keep_mask = group[residuals_column].apply(
                lambda row: np.all(np.abs(np.ravel(row) - median) <= k * sigma_robust)
            )

        filtered_group = group.loc[keep_mask]
        n_removed_group = len(group) - len(filtered_group)
        n_removed_total += n_removed_group
        removed_per_group.append((group_key, len(group), n_removed_group))
        filtered_groups.append(filtered_group)

    filtered_valid_df = pd.concat(filtered_groups, ignore_index=True) if filtered_groups else pd.DataFrame()
    final_df = pd.concat([filtered_valid_df, df_other], ignore_index=True)

    if verbose:
        print("\n--- Outlier Removal per Group ---")
        for group_key, n_valid, n_removed in removed_per_group:
            print(f"Group {group_key}: removed {n_removed} of {n_valid} (kept {n_valid - n_removed})")

        print("\n--- Overall ---")
        print(f"Valid rows with residuals: {n_valid_total}")
        print(f"Removed: {n_removed_total}")
        print(f"Kept: {n_valid_total - n_removed_total}")
        print(f"Unchanged rows without residuals: {len(df_other)}")
        print(f"Final total rows: {len(final_df)}")

    return final_df


def plot_ecdf(
    datasets: list[dict[str, Any]],
    xlabel: str = "Position Error [m]",
    ylabel: str = "CDF",
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    xticks: list[float] | None = None,
    yticks: list[float] | None = None,
    xscale: str | None = None,
    save_path: str | None = None,
    fig_width: float = COL_WIDTH,
    fig_height: float = ROW_HEIGHT,
    show_fig: bool = True,
) -> None:
    """ Plot one or more empirical cumulative distribution functions

    Args:
        datasets (list[dict[str, Any]]): Dataset specifications with df, x, label, and optional filter_col and min_val
        xlabel (str): X-axis label
        ylabel (str): Y-axis label
        xlim (tuple[float, float] | None): Optional x-axis limits
        ylim (tuple[float, float] | None): Optional y-axis limits
        xticks (list[float] | None): Optional x-axis tick positions
        yticks (list[float] | None): Optional y-axis tick positions
        xscale (str | None): Optional x-axis scale, for example "log"
        save_path (str | None): Optional output file stem
        fig_width (float): Figure width in inches
        fig_height (float): Figure height in inches
        show_fig (bool): Whether to display the figure interactively

    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)
    linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]

    for index, dataset in enumerate(datasets):
        filter_col = dataset.get("filter_col", dataset["x"])
        data = dataset["df"].loc[dataset["df"][filter_col].notna()].copy()

        if "min_val" in dataset:
            data = data.loc[data[dataset["x"]] > dataset["min_val"]]

        sns.ecdfplot(
            data=data,
            x=dataset["x"],
            label=dataset["label"],
            linestyle=linestyles[index % len(linestyles)],
            ax=ax,
        )

    _apply_axis_options(ax, xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim, xticks=xticks, yticks=yticks)

    if xscale is not None:
        ax.set_xscale(xscale)

    ax.legend(fontsize=FONT_SIZE)

    if save_path is not None:
        fig.savefig(_build_timestamped_output_path(save_path), bbox_inches="tight")

    if show_fig:
        plt.show()
    else:
        plt.close(fig)


def rmse(values: pd.Series | np.ndarray) -> float:
    """ Calculate the root mean square error of a numeric vector

    Args:
        values (pd.Series | np.ndarray): Numeric error values

    Returns:
        float: Root mean square error
    """
    array = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(array**2)))


def aggregate_info_per_node(
    df: pd.DataFrame,
    grouping: list[str],
    error_col: str,
    method_display_name: str,
    timestamp_format: str = "sim_timestamp",
) -> pd.DataFrame:
    """ Aggregate localization statistics per node and experiment group

    Args:
        df (pd.DataFrame): Input dataframe containing localization results
        grouping (list[str]): Columns used for grouping, typically including Rep and NodeId
        error_col (str): Error column used for mean and RMSE statistics
        method_display_name (str): Display name written to the Method column
        timestamp_format (str): Timestamp column used to compute localization duration

    Returns:
        pd.DataFrame: Aggregated dataframe with count, mean error, RMSE, duration, and frequency
    """
    grouped_df = (
        df.groupby(grouping)
        .agg(
            loc_count=(error_col, "count"),
            mean_error=(error_col, "mean"),
            rmse_error=(error_col, rmse),
            first_loc=(timestamp_format, "min"),
            last_loc=(timestamp_format, "max"),
        )
        .reset_index()
    )

    grouped_df["Method"] = method_display_name
    grouped_df["Loc_Duration"] = grouped_df["last_loc"] - grouped_df["first_loc"]
    grouped_df["Mean_Frequency"] = grouped_df["loc_count"] / grouped_df["Loc_Duration"]

    return grouped_df


def filter_results(
    df: pd.DataFrame,
    grouping: list[str] | None = None,
    k: float = 10.0,
    filter_residuals: bool = True,
    filter_pdop: bool = True,
    pdop_factor: float = 2.0,
    filter_anchor_stitching: bool = True,
    merge_vtwr_and_self_loc: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """ Apply the standard result filters used before plotting

    Args:
        df (pd.DataFrame): Input dataframe containing localization results
        grouping (list[str] | None): Columns used for grouped residual outlier filtering
        k (float): Robust residual outlier threshold factor
        filter_residuals (bool): Whether to remove residual outliers
        filter_pdop (bool): Whether to filter rows by PDOP
        pdop_factor (float): Maximum accepted PDOP value
        filter_anchor_stitching (bool): Whether to remove anchor-stitching estimates
        merge_vtwr_and_self_loc (bool): Whether to merge self-localization and v-TWR error columns
        verbose (bool): Whether to print filtering statistics

    Returns:
        pd.DataFrame: Filtered dataframe ready for plotting
    """
    grouping = grouping or ["Method"]
    filtered_df = df.copy()

    if filter_residuals:
        filtered_df = filter_residual_outliers_per_parameter(
            filtered_df,
            grouping,
            k=k,
            verbose=verbose,
        )

    if filter_pdop:
        filtered_df = filter_pdop_values(
            filtered_df,
            pdop_factor=pdop_factor,
            verbose=verbose,
        )

    if filter_anchor_stitching:
        filtered_df = filter_anchor_stitching_values(filtered_df, verbose=verbose)

    if merge_vtwr_and_self_loc:
        filtered_df = merge_exclusive_columns(
            filtered_df,
            col1="self_loc_error",
            col2="v_twr_error",
        )

    return filtered_df


def filter_pdop_values(
    df: pd.DataFrame,
    pdop_factor: float = 2.0,
    verbose: bool = True,
) -> pd.DataFrame:
    """ Remove rows with PDOP values above the configured threshold

    Args:
        df (pd.DataFrame): Input dataframe containing a PDOP column
        pdop_factor (float): Maximum accepted PDOP value
        verbose (bool): Whether to print the number of removed rows

    Returns:
        pd.DataFrame: Dataframe with high-PDOP rows removed while keeping rows without PDOP
    """
    len_before = len(df)
    pdop_ok = (df["PDOP"] < pdop_factor) | df["PDOP"].isna()
    df_filtered = df.loc[pdop_ok].copy()
    len_after = len(df_filtered)

    if verbose:
        print(f"Removed {len_before - len_after} rows due to PDOP; from {len_before} to {len_after} rows.")

    return df_filtered


def filter_anchor_stitching_values(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """ Remove rows that were produced with anchor stitching

    Args:
        df (pd.DataFrame): Input dataframe containing a using_anchor_stitching column
        verbose (bool): Whether to print the number of removed rows

    Returns:
        pd.DataFrame: Dataframe without anchor-stitching rows
    """
    len_before = len(df)
    keep_mask = df["using_anchor_stitching"] == False
    df_filtered = df.loc[keep_mask].copy()
    len_after = len(df_filtered)

    if verbose:
        print(f"Removed {len_before - len_after} rows that used anchor stitching; from {len_before} to {len_after} rows.")

    return df_filtered


def plot_boxplot(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str,
    xlabel: str | None = None,
    ylabel: str | None = None,
    xlog: bool = False,
    ylog: bool = False,
    showfliers: bool = True,
    legend_ncol: int = 2,
    legend_above: bool = True,
    save_path: str | None = None,
    legend_title: str | None = None,
    title: str | None = None,
    fig_width: float = COL_WIDTH,
    fig_height: float = ROW_HEIGHT,
    show_fig: bool = True,
) -> None:
    """ Plot a grouped boxplot and optionally save it as a PDF

    Args:
        df (pd.DataFrame): Input dataframe to plot
        x (str): Column used for the x-axis
        y (str): Column used for the y-axis
        hue (str): Column used to split boxes by method or category
        xlabel (str | None): Optional x-axis label
        ylabel (str | None): Optional y-axis label
        xlog (bool): Whether to use a logarithmic x-axis
        ylog (bool): Whether to use a logarithmic y-axis
        showfliers (bool): Whether to show outlier markers
        legend_ncol (int): Number of legend columns
        legend_above (bool): Whether to place the legend above the plot
        save_path (str | None): Optional output file stem
        legend_title (str | None): Optional legend title
        title (str | None): Optional plot title
        fig_width (float): Figure width in inches
        fig_height (float): Figure height in inches
        show_fig (bool): Whether to display the figure interactively

    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)
    sns.boxplot(data=df, x=x, y=y, hue=hue, showfliers=showfliers, ax=ax)

    if title:
        ax.set_title(title, fontsize=FONT_SIZE)

    _apply_axis_options(
        ax,
        xlabel=xlabel or x,
        ylabel=ylabel or y,
        xlog=xlog,
        ylog=ylog,
    )
    _add_legend(
        ax, 
        legend_title=legend_title, 
        legend_ncol=legend_ncol, 
        legend_above=legend_above
    )

    if save_path is not None:
        fig.savefig(_build_timestamped_output_path(save_path), dpi=300, bbox_inches="tight")

    if show_fig:
        plt.show()
    else:
        plt.close(fig)


def prepare_coverage_error_df(
    dfs_with_method: list[tuple[pd.DataFrame, str]],
    timestamp_col: str,
    error_col_map: dict[str, str],
    bin_width: float = 0.04,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """ Prepare coverage and error dataframes for coverage/error ECDF plots

    Args:
        dfs_with_method (list[tuple[pd.DataFrame, str]]): Input dataframes paired with raw method names
        timestamp_col (str): Timestamp column used to assign localizations to time bins
        error_col_map (dict[str, str]): Mapping from method name to the source error column
        bin_width (float): Width of the time bins in seconds

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Coverage ECDF data, error ECDF data, and aggregated plot data
    """
    frames = []
    for df, method in dfs_with_method:
        d = df.copy()
        d["Method"] = method
        error_col = error_col_map.get(method, "hyrasel_pos_error")
        d["_error"] = d[error_col]
        frames.append(d)

    df_all = pd.concat(frames, ignore_index=True)
    df_valid = df_all.loc[df_all["_error"].notna()].copy()
    df_valid["time_bin"] = np.floor(df_valid[timestamp_col] / bin_width) * bin_width

    num_nodes = df_all["NodeId"].nunique()

    # Coverage
    df_node = (
        df_valid.groupby(["Method", "Rep", "interAnchorDist", "time_bin", "NodeId"])
        .size()
        .reset_index(name="n_loc")
    )
    df_node["localized"] = df_node["n_loc"] >= 1

    df_cov_run = (
        df_node.groupby(["Method", "Rep", "interAnchorDist", "time_bin"], as_index=False)
        .agg(n_nodes_localized=("localized", "sum"))
    )
    df_cov_run["coverage"] = df_cov_run["n_nodes_localized"] / num_nodes

    # Error
    df_error_ecdf = (
        df_valid[["Method", "Rep", "interAnchorDist", "time_bin", "NodeId", "_error"]]
        .rename(columns={"_error": "hyrasel_pos_error"})
        .copy()
    )

    # Aggregated plot-df
    df_run = (
        df_valid
        .groupby(["Method", "Rep", "interAnchorDist", "time_bin"], as_index=False)
        .agg(mean_error=("_error", "mean"), median_error=("_error", "median"))
    )

    df_cov_plot = (
        df_cov_run
        .groupby(["Method", "interAnchorDist", "time_bin"], as_index=False)
        .agg(coverage_mean=("coverage", "mean"), coverage_sd=("coverage", "std"))
    )

    df_error_plot = (
        df_run
        .groupby(["Method", "interAnchorDist", "time_bin"], as_index=False)
        .agg(
            error_mean=("mean_error", "mean"),
            error_sd=("mean_error", "std"),
            error_median=("median_error", "mean"),
            error_q25=("median_error", lambda x: x.quantile(0.25)),
            error_q75=("median_error", lambda x: x.quantile(0.75)),
        )
    )

    df_plot = pd.merge(
        df_cov_plot,
        df_error_plot,
        on=["Method", "interAnchorDist", "time_bin"],
        how="inner",
    )

    return df_cov_run, df_error_ecdf, df_plot


def plot_coverage_ecdf(
    df_cov_ecdf: pd.DataFrame,
    df_error_ecdf: pd.DataFrame,
    hyperparams: list[float] | None = None,
    descriptors: list[str] | tuple[str, ...] = ("Dense", "Semi-Dense", "Sparse"),
    label_map: dict[str, str] | None = None,
    xlabel_cov: str = "Coverage",
    xlabel_err: str = "Localization Error [m]",
    ylabel: str = "CDF",
    xlim_err: tuple[float, float] = (0, 1),
    save_path: str | None = None,
    fig_width: float = COL_WIDTH_2,
    fig_height: float = ROW_HEIGHT * 1.8,
    legend_ncol: int = 2,
    show_fig: bool = False,
) -> None:
    """ Plot coverage and localization-error ECDFs in a two-row facet grid

    Args:
        df_cov_ecdf (pd.DataFrame): Coverage data with one row per method, repetition, density, and time bin
        df_error_ecdf (pd.DataFrame): Localization-error data with one row per localization event
        hyperparams (list[float] | None): Inter-anchor distances to plot as columns
        descriptors (list[str] | tuple[str, ...]): Human-readable density labels for each column
        label_map (dict[str, str] | None): Mapping from raw method names to display names
        xlabel_cov (str): X-axis label for the coverage ECDF
        xlabel_err (str): X-axis label for the error ECDF
        ylabel (str): Y-axis label for both ECDF rows
        xlim_err (tuple[float, float]): X-axis limits for localization error
        save_path (str | None): Optional output file stem
        fig_width (float): Figure width in inches
        fig_height (float): Figure height in inches
        legend_ncol (int): Number of legend columns
        show_fig (bool): Whether to display the figure interactively

    Returns:
        None
    """
    sns.set_theme(style="whitegrid")

    hyperparams = hyperparams or sorted(df_cov_ecdf["interAnchorDist"].unique())
    methods = sorted(df_cov_ecdf["Method"].unique())
    palette = sns.color_palette("tab10", len(methods))
    linestyles = ["-", "--", "-.", ":"]
    ecdf_ticks = np.arange(0, 1.01, 0.25)

    fig, axes = plt.subplots(
        nrows=2,
        ncols=len(hyperparams),
        figsize=(fig_width, fig_height),
        constrained_layout=True,
        sharey="row",
        sharex=False,
    )

    if len(hyperparams) == 1:
        axes = np.asarray(axes).reshape(2, 1)

    # fig.set_constrained_layout_pads(
    #     w_pad=0.01, 
    #     h_pad=0.01, 
    #     wspace=0.01, 
    #     hspace=0.03
    # )

    for j, hp in enumerate(hyperparams):
        descriptor = descriptors[j] if j < len(descriptors) else str(hp)
        ax_cov, ax_err = axes[0, j], axes[1, j]

        for i, method in enumerate(methods):
            line_kwargs = {
                "color": palette[i],
                "linestyle": linestyles[i % len(linestyles)],
                "linewidth": 2,
            }
            cov_mask = (df_cov_ecdf["interAnchorDist"] == hp) & (df_cov_ecdf["Method"] == method)
            err_mask = (df_error_ecdf["interAnchorDist"] == hp) & (df_error_ecdf["Method"] == method)

            d_cov = df_cov_ecdf.loc[cov_mask]
            if not d_cov.empty:
                sns.ecdfplot(data=d_cov, x="coverage", ax=ax_cov, label=method, **line_kwargs)

            d_err = df_error_ecdf.loc[err_mask]
            if not d_err.empty:
                sns.ecdfplot(data=d_err, x="hyrasel_pos_error", ax=ax_err, **line_kwargs)

        ax_cov.set_title(f"{descriptor} ({hp} m)", fontsize=FONT_SIZE)
        _apply_axis_options(ax_cov, xlabel=xlabel_cov, ylabel=ylabel if j == 0 else "", xlim=(0, 1.01), ylim=(0, 1.01))
        _apply_axis_options(ax_err, xlabel=xlabel_err, ylabel=ylabel if j == 0 else "", xlim=xlim_err, ylim=(0, 1.01))
        ax_cov.set_xticks(ecdf_ticks)
        ax_cov.set_yticks(ecdf_ticks)
        ax_err.set_yticks(ecdf_ticks)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if label_map:
        labels = [label_map.get(label, label) for label in labels]

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=legend_ncol,
        frameon=True,
        columnspacing=0.2,
        bbox_to_anchor=(0.5, 1.00),
        fontsize=FONT_SIZE,
    )

    if save_path:
        fig.savefig(_build_timestamped_output_path(save_path), bbox_inches="tight")

    if show_fig:
        plt.show()
    else:
        plt.close(fig)


def plot_mean_with_errorbar(
    datasets: list[dict[str, Any]],
    xlabel: str | None = None,
    ylabel: str | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    xticks: list[float] | None = None,
    yticks: list[float] | None = None,
    show_errorbars: bool = True,
    verbose: bool = True,
    save_path: str | None = None,
    use_rep_means: bool = False,
    legend_above: bool = True,
    legend_title: str | None = None,
    legend_ncol: int = 2,
    FIG_WIDTH: float = COL_WIDTH,
    FIG_HEIGHT: float = ROW_HEIGHT,
    show_fig: bool = False,
) -> None:
    """ Plot mean values with optional standard-error bars for multiple datasets

    Args:
        datasets (list[dict[str, Any]]): Dataset specifications with df, x, y, label, and optional filter_col
        xlabel (str | None): X-axis label
        ylabel (str | None): Y-axis label
        xlim (tuple[float, float] | None): Optional x-axis limits
        ylim (tuple[float, float] | None): Optional y-axis limits
        xticks (list[float] | None): Optional x-axis tick positions
        yticks (list[float] | None): Optional y-axis tick positions
        show_errorbars (bool): Whether to draw SEM error bars
        verbose (bool): Whether to print grouped summary statistics
        save_path (str | None): Optional output file stem
        use_rep_means (bool): Whether to compute statistics from per-repetition means
        legend_above (bool): Whether to place the legend above the plot
        legend_title (str | None): Optional legend title
        legend_ncol (int): Number of legend columns
        FIG_WIDTH (float): Figure width in inches
        FIG_HEIGHT (float): Figure height in inches
        show_fig (bool): Whether to display the figure interactively

    Returns:
        None
    """
    linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
    markers = ["o", "s", "^", "D", "x"]

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), constrained_layout=True)

    for i, ds in enumerate(datasets):
        filter_col = ds.get("filter_col", ds["y"])
        data = ds["df"].loc[ds["df"][filter_col].notna()].copy()

        if use_rep_means:
            rep_means = data.groupby([ds["x"], "Rep"])[ds["y"]].mean().unstack("Rep")
            means = rep_means.mean(axis=1)
            stds = rep_means.std(axis=1)
            counts = rep_means.count(axis=1)
            sem = stds / np.sqrt(counts)
        else:
            grouped = data.groupby(ds["x"])[ds["y"]]
            means = grouped.mean()
            stds = grouped.std()
            counts = grouped.count()
            sem = stds / np.sqrt(counts)

        if verbose:
            print(f"\n--- {ds['label']} ---")
            print(f"Rows per group:\n{counts}")
            print(f"Mean:\n{means}")
            print(f"Std:\n{stds}")
            print(f"SEM:\n{sem}")

        plot_kwargs = {
            "label": ds["label"],
            "linestyle": linestyles[i % len(linestyles)],
            "marker": markers[i % len(markers)],
        }

        if show_errorbars:
            ax.errorbar(means.index, means.values, yerr=sem.values, capsize=4, **plot_kwargs)
        else:
            ax.plot(means.index, means.values, **plot_kwargs)

    _apply_axis_options(ax, xlabel=xlabel or "", ylabel=ylabel or "", xlim=xlim, ylim=ylim, xticks=xticks, yticks=yticks)
    _add_legend(ax, legend_title=legend_title, legend_ncol=legend_ncol, legend_above=legend_above)

    if save_path is not None:
        fig.savefig(_build_timestamped_output_path(save_path), dpi=300, bbox_inches="tight")

    if show_fig:
        plt.show()
    else:
        plt.close(fig)



#################################################################################
#   _   _                 _                        __   ____  _       _         #
#  | \ | |_   _ _ __ ___ | |__   ___ _ __    ___  / _| / ___|| | ___ | |_ ___   #
#  |  \| | | | | '_ ` _ \| '_ \ / _ \ '__|  / _ \| |_  \___ \| |/ _ \| __/ __|  #
#  | |\  | |_| | | | | | | |_) |  __/ |    | (_) |  _|  ___) | | (_) | |_\__ \  #
#  |_| \_|\__,_|_| |_| |_|_.__/ \___|_|     \___/|_|   |____/|_|\___/ \__|___/  #
#################################################################################                                                                         

# %% Extracting
centralized_slots_df = load_results_from_csv(["CENTRALIZED_" + TIMESTAMP_FORMAT + "_numSlots_2_with_vtwr",
                                        "CENTRALIZED_" + TIMESTAMP_FORMAT + "_numSlots_3_with_vtwr",
                                        "CENTRALIZED_" + TIMESTAMP_FORMAT + "_numSlots_4_with_vtwr",
                                        "CENTRALIZED_" + TIMESTAMP_FORMAT + "_numSlots_5_with_vtwr"])

distributed_slots_df = load_results_from_csv(["DISTRIBUTED_" + TIMESTAMP_FORMAT + "_numSlots_2_with_vtwr",
                                        "DISTRIBUTED_" + TIMESTAMP_FORMAT + "_numSlots_3_with_vtwr",
                                        "DISTRIBUTED_" + TIMESTAMP_FORMAT + "_numSlots_4_with_vtwr",
                                        "DISTRIBUTED_" + TIMESTAMP_FORMAT + "_numSlots_5_with_vtwr"])


# %% filtering
k = K
grouping_list = ["numSlots"]
distributed_slots_filtered_df = filter_results(distributed_slots_df, 
                                     grouping_list, 
                                     k=k, 
                                     filter_residuals=True, 
                                     filter_pdop=True, 
                                     pdop_factor=2.0,
                                     filter_anchor_stitching=False,
                                     merge_vtwr_and_self_loc=True, 
                                     verbose=True
                                     )

centralized_slots_filtered_df = filter_results(centralized_slots_df, 
                                     grouping_list, 
                                     k=k, 
                                     filter_residuals=True, 
                                     filter_pdop=True, 
                                     pdop_factor=2.0,
                                     filter_anchor_stitching=False,
                                     merge_vtwr_and_self_loc=True, 
                                     verbose=True
                                     )

# %% grouping
groupby_list = ["numSlots", "Rep", "NodeId"]
centralized_slots_grouped_df = aggregate_info_per_node(centralized_slots_filtered_df, groupby_list, "hyrasel_pos_error" , "HyRASeL (Centralized)", TIMESTAMP_FORMAT)
distributed_slots_grouped_df = aggregate_info_per_node(distributed_slots_filtered_df, groupby_list, "hyrasel_pos_error" , "HyRASeL (Distributed)", TIMESTAMP_FORMAT)

df_all = pd.concat([centralized_slots_grouped_df, distributed_slots_grouped_df], ignore_index=True)

# %% Accuracy
plot_mean_with_errorbar(
    datasets=[
        {"df": centralized_slots_filtered_df, "x": "numSlots", "y": "hyrasel_pos_error", "label": CENTRALIZED_HYRASEL_DESCRIPTOR},
        {"df": distributed_slots_filtered_df, "x": "numSlots", "y": "hyrasel_pos_error", "label": DISTRIBUTED_HYRASEL_DESCRIPTOR},
        {"df": centralized_slots_filtered_df, "x": "numSlots", "y": "v_twr_error", "label": CENTRALIZED_VTWR_DESCRIPTOR},
        {"df": distributed_slots_filtered_df, "x": "numSlots", "y": "v_twr_error", "label": DISTRIBUTED_VTWR_DESCRIPTOR},
    ],
    xlabel=NUM_SLOTS_DESCRIPTOR,
    ylabel=ACCURACY_DESCRIPTOR,
    xticks=[2, 3, 4, 5],
    save_path="Figure_04a_Error_vs_numSlots", 
    use_rep_means = True,
    legend_ncol=2,
    FIG_HEIGHT=ROW_HEIGHT*1.1,
    )


# %% Inter Loc Time
plot_mean_with_errorbar(
    datasets=[
        {"df": compute_inter_measurement_time(centralized_slots_filtered_df, "hyrasel_pos_error", TIMESTAMP_FORMAT), "x": "numSlots", "y": "inter_measurement_time", "label": CENTRALIZED_HYRASEL_DESCRIPTOR},
        {"df": compute_inter_measurement_time(distributed_slots_filtered_df, "hyrasel_pos_error", TIMESTAMP_FORMAT), "x": "numSlots", "y": "inter_measurement_time", "label": DISTRIBUTED_HYRASEL_DESCRIPTOR},
        {"df": compute_inter_measurement_time(centralized_slots_filtered_df, "v_twr_error", TIMESTAMP_FORMAT), "x": "numSlots", "y": "inter_measurement_time", "label": CENTRALIZED_VTWR_DESCRIPTOR},
        {"df": compute_inter_measurement_time(distributed_slots_filtered_df, "v_twr_error", TIMESTAMP_FORMAT), "x": "numSlots", "y": "inter_measurement_time", "label": DISTRIBUTED_VTWR_DESCRIPTOR},
    ],
    xlabel=NUM_SLOTS_DESCRIPTOR,
    ylabel=INTER_UPDATE_TIME_DESCRIPTOR,
    xticks=[2, 3, 4, 5],
    save_path="Figure_04b_ILT_vs_numSlots_with_vtwr", 
    use_rep_means = True,
    legend_ncol=2,
    FIG_HEIGHT=ROW_HEIGHT*1.1,
    )


#########################################################################################
#   _   _                 _                        __   _   _           _               #
#  | \ | |_   _ _ __ ___ | |__   ___ _ __    ___  / _| | \ | | ___   __| | ___  ___     #
#  |  \| | | | | '_ ` _ \| '_ \ / _ \ '__|  / _ \| |_  |  \| |/ _ \ / _` |/ _ \/ __|    #
#  | |\  | |_| | | | | | | |_) |  __/ |    | (_) |  _| | |\  | (_) | (_| |  __/\__ \    #
#  |_| \_|\__,_|_| |_| |_|_.__/ \___|_|     \___/|_|   |_| \_|\___/ \__,_|\___||___/    #
#########################################################################################                                                                                 


# %% NUMBER OF NODES
twr_nodes_df = load_results_from_csv(["TWR_" + TIMESTAMP_FORMAT + "_numNodes_4", 
                                        "TWR_" + TIMESTAMP_FORMAT + "_numNodes_9", 
                                        "TWR_" + TIMESTAMP_FORMAT + "_numNodes_16", 
                                        "TWR_" + TIMESTAMP_FORMAT + "_numNodes_25", 
                                        "TWR_" + TIMESTAMP_FORMAT + "_numNodes_36",
                                        "TWR_" + TIMESTAMP_FORMAT + "_numNodes_49"
                                        ], 
                                        append_parameters=[4, 9, 16, 25, 36, 49], 
                                        append_value="numNodes"
                                        )
distributed_nodes_df = load_results_from_csv(["DISTRIBUTED_" + TIMESTAMP_FORMAT + "_numNodes_4_with_vtwr", 
                                        "DISTRIBUTED_" + TIMESTAMP_FORMAT + "_numNodes_9_with_vtwr", 
                                        "DISTRIBUTED_" + TIMESTAMP_FORMAT + "_numNodes_16_with_vtwr", 
                                        "DISTRIBUTED_" + TIMESTAMP_FORMAT + "_numNodes_25_with_vtwr", 
                                        "DISTRIBUTED_" + TIMESTAMP_FORMAT + "_numNodes_36_with_vtwr",
                                        "DISTRIBUTED_" + TIMESTAMP_FORMAT + "_numNodes_49_with_vtwr",
                                        ])
centralized_nodes_df = load_results_from_csv(["CENTRALIZED_" + TIMESTAMP_FORMAT + "_numNodes_4_with_vtwr", 
                                       "CENTRALIZED_" + TIMESTAMP_FORMAT + "_numNodes_9_with_vtwr", 
                                       "CENTRALIZED_" + TIMESTAMP_FORMAT + "_numNodes_16_with_vtwr", 
                                       "CENTRALIZED_" + TIMESTAMP_FORMAT + "_numNodes_25_with_vtwr", 
                                       "CENTRALIZED_" + TIMESTAMP_FORMAT + "_numNodes_36_with_vtwr",
                                       "CENTRALIZED_" + TIMESTAMP_FORMAT + "_numNodes_49_with_vtwr"
                                       ])


# %% Filtering
k = K
grouping_list = ["numNodes"]
distributed_nodes_filtered_df = filter_results(distributed_nodes_df, 
                            grouping_list, 
                            k=k, 
                            filter_residuals=True, 
                            filter_pdop=True, 
                            pdop_factor=2.0,
                            filter_anchor_stitching=False,
                            merge_vtwr_and_self_loc=True, 
                            verbose=True
                            )

centralized_nodes_filtered_df = filter_results(centralized_nodes_df, 
                            grouping_list, 
                            k=k, 
                            filter_residuals=True, 
                            filter_pdop=True, 
                            pdop_factor=2.0,
                            filter_anchor_stitching=False,
                            merge_vtwr_and_self_loc=True, 
                            verbose=True
                            )


# %% Accuracy vs. number of nodes
plot_mean_with_errorbar(
    datasets=[
        {"df": centralized_nodes_filtered_df,"x": "numNodes", "y": "hyrasel_pos_error", "label": CENTRALIZED_HYRASEL_DESCRIPTOR},
        {"df": distributed_nodes_filtered_df,"x": "numNodes", "y": "hyrasel_pos_error", "label": DISTRIBUTED_HYRASEL_DESCRIPTOR},
        {"df": centralized_nodes_filtered_df,"x": "numNodes", "y": "v_twr_error", "label": CENTRALIZED_VTWR_DESCRIPTOR},
        {"df": distributed_nodes_filtered_df,"x": "numNodes", "y": "v_twr_error", "label": DISTRIBUTED_VTWR_DESCRIPTOR},
        {"df": twr_nodes_df, "x": "numNodes", "y": "pos_error", "label": TWR_DESCRIPTOR},
    ],
    xlabel=NUM_NODES_DESCRIPTOR,
    ylabel=ACCURACY_DESCRIPTOR,
    xticks=[4, 9, 16, 25, 36, 49],
    show_errorbars=True, 
    save_path="Figure_05a_error_vs_nodeNum", 
    use_rep_means = True,
    legend_ncol=2,
    FIG_HEIGHT=ROW_HEIGHT*1.2
)

# %% Inter-Loc Time vs. Number of Nodes
plot_mean_with_errorbar(
    datasets=[
        {"df": compute_inter_measurement_time(centralized_nodes_filtered_df, "hyrasel_pos_error", TIMESTAMP_FORMAT), "x": "numNodes", "y": "inter_measurement_time", "label": CENTRALIZED_HYRASEL_DESCRIPTOR},
        {"df": compute_inter_measurement_time(distributed_nodes_filtered_df, "hyrasel_pos_error", TIMESTAMP_FORMAT), "x": "numNodes", "y": "inter_measurement_time", "label": DISTRIBUTED_HYRASEL_DESCRIPTOR}, # hyrasel_pos_error
        {"df": compute_inter_measurement_time(centralized_nodes_filtered_df, "v_twr_error", TIMESTAMP_FORMAT), "x": "numNodes", "y": "inter_measurement_time", "label": CENTRALIZED_VTWR_DESCRIPTOR},
        {"df": compute_inter_measurement_time(distributed_nodes_filtered_df, "v_twr_error", TIMESTAMP_FORMAT), "x": "numNodes", "y": "inter_measurement_time", "label": DISTRIBUTED_VTWR_DESCRIPTOR},
        {"df": compute_inter_measurement_time(twr_nodes_df,"pos_error", TIMESTAMP_FORMAT), "x": "numNodes", "y": "inter_measurement_time", "label": TWR_DESCRIPTOR},
    ],
    xlabel=NUM_NODES_DESCRIPTOR,
    ylabel=INTER_UPDATE_TIME_DESCRIPTOR,
    xticks=[4, 9, 16, 25, 36, 49],
    save_path="Figure_05b_ILT_vs_numNodes", 
    use_rep_means = True,
    legend_ncol=2,
    FIG_HEIGHT=ROW_HEIGHT*1.2
    )



#############################################################################
#      _               _                  ____                 _ _          #
#     / \   _ __   ___| |__   ___  _ __  |  _ \  ___ _ __  ___(_) |_ _   _  #
#    / _ \ | '_ \ / __| '_ \ / _ \| '__| | | | |/ _ \ '_ \/ __| | __| | | | #
#   / ___ \| | | | (__| | | | (_) | |    | |_| |  __/ | | \__ \ | |_| |_| | #
#  /_/   \_\_| |_|\___|_| |_|\___/|_|    |____/ \___|_| |_|___/_|\__|\__, | #
#                                                                    |___/  #
#############################################################################


# %% ANCHOR DENSITY: 
twr_density_df = load_results_from_csv(["TWR_" + TIMESTAMP_FORMAT + "_interAnchorDist_130",
                                        "TWR_" + TIMESTAMP_FORMAT + "_interAnchorDist_200",
                                        "TWR_" + TIMESTAMP_FORMAT + "_interAnchorDist_270"],
                                        append_parameters=[130, 200, 270], append_value="interAnchorDist")

self_loc_density_df = load_results_from_csv(["HYPERBOLIC_SELF_LOCALIZATION_" + TIMESTAMP_FORMAT + "_interAnchorDist_130_with_vtwr",
                                            "HYPERBOLIC_SELF_LOCALIZATION_" + TIMESTAMP_FORMAT + "_interAnchorDist_200_with_vtwr",
                                            "HYPERBOLIC_SELF_LOCALIZATION_" + TIMESTAMP_FORMAT + "_interAnchorDist_270_with_vtwr",])

centralized_density_df = load_results_from_csv(["CENTRALIZED_" + TIMESTAMP_FORMAT + "_interAnchorDist_130_with_vtwr",
                                           "CENTRALIZED_" + TIMESTAMP_FORMAT + "_interAnchorDist_200_with_vtwr", 
                                           "CENTRALIZED_" + TIMESTAMP_FORMAT + "_interAnchorDist_270_with_vtwr", ])

distributed_density_df = load_results_from_csv(["DISTRIBUTED_" + TIMESTAMP_FORMAT + "_interAnchorDist_130_with_vtwr",
                                         "DISTRIBUTED_" + TIMESTAMP_FORMAT + "_interAnchorDist_200_with_vtwr", 
                                         "DISTRIBUTED_" + TIMESTAMP_FORMAT + "_interAnchorDist_270_with_vtwr",])


# %% Filtering
k = K
grouping_list = ["interAnchorDist"]
self_loc_density_filtered_df = filter_results(self_loc_density_df, 
                                     grouping_list, 
                                     k=k, 
                                     filter_residuals=True, 
                                     filter_pdop=True, 
                                     pdop_factor=2.0,
                                     filter_anchor_stitching=True,
                                     merge_vtwr_and_self_loc=True, # No effect tbh
                                     verbose=True
                                     )

centralized_density_filtered_df = filter_results(centralized_density_df, 
                                     grouping_list, 
                                     k=k, 
                                     filter_residuals=True, 
                                     filter_pdop=True, 
                                     pdop_factor=2.0,
                                     filter_anchor_stitching=True,
                                     merge_vtwr_and_self_loc=True, 
                                     verbose=True
                                     )

distributed_density_filtered_df = filter_results(distributed_density_df, 
                                     grouping_list, 
                                     k=k, 
                                     filter_residuals=True, 
                                     filter_pdop=True, 
                                     pdop_factor=2.0,
                                     filter_anchor_stitching=True,
                                     merge_vtwr_and_self_loc=True, 
                                     verbose=True
                                     )

centralized_density_filtered_df_w_stitch = filter_results(centralized_density_df, 
                                     grouping_list, 
                                     k=k, 
                                     filter_residuals=True, 
                                     filter_pdop=True, 
                                     pdop_factor=2.0,
                                     filter_anchor_stitching=False,
                                     merge_vtwr_and_self_loc=True, 
                                     verbose=True
                                     )

distributed_density_filtered_df_w_stitch = filter_results(distributed_density_df, 
                                     grouping_list, 
                                     k=k, 
                                     filter_residuals=True, 
                                     filter_pdop=True, 
                                     pdop_factor=2.0,
                                     filter_anchor_stitching=False,
                                     merge_vtwr_and_self_loc=True, 
                                     verbose=True
                                     )

self_loc_density_filtered_df_w_stitch = filter_results(self_loc_density_df, 
                                     grouping_list, 
                                     k=k, 
                                     filter_residuals=True, 
                                     filter_pdop=True, 
                                     pdop_factor=2.0,
                                     filter_anchor_stitching=False,
                                     merge_vtwr_and_self_loc=True, # No effect tbh
                                     verbose=True
                                     )

# %% Group by node
groupby_list = ["interAnchorDist", "Rep", "NodeId"]
twr_grouped_df = aggregate_info_per_node(twr_density_df, groupby_list, "pos_error", "TWR (Baseline)", TIMESTAMP_FORMAT)

self_loc_grouped_df_w_stitch = aggregate_info_per_node(self_loc_density_filtered_df_w_stitch, groupby_list, "self_loc_error", "Chiasson et al. (Baseline)", TIMESTAMP_FORMAT)
self_loc_grouped_df = aggregate_info_per_node(self_loc_density_filtered_df, groupby_list, "self_loc_error", "Chiasson et al. (Baseline)", TIMESTAMP_FORMAT)

distributed_grouped_df_w_stitch = aggregate_info_per_node(distributed_density_filtered_df_w_stitch, groupby_list, "hyrasel_pos_error", "HyRASeL (Distributed)", TIMESTAMP_FORMAT)
distributed_grouped_df = aggregate_info_per_node(distributed_density_filtered_df, groupby_list, "hyrasel_pos_error", "HyRASeL (Distributed)", TIMESTAMP_FORMAT)

centralized_grouped_w_stitch_df =  aggregate_info_per_node(centralized_density_filtered_df_w_stitch, groupby_list, "hyrasel_pos_error", "HyRASeL (Centralized)", TIMESTAMP_FORMAT)
centralized_grouped_df = aggregate_info_per_node(centralized_density_filtered_df, groupby_list, "hyrasel_pos_error", "HyRASeL (Centralized)", TIMESTAMP_FORMAT)

df_all_wo_stitching = pd.concat([twr_grouped_df,
                                 self_loc_grouped_df, 
                                 distributed_grouped_df, 
                                 centralized_grouped_df], ignore_index=True)
df_all_w_stitching = pd.concat([twr_grouped_df,
                                self_loc_grouped_df_w_stitch, 
                                distributed_grouped_df_w_stitch, 
                                centralized_grouped_w_stitch_df], ignore_index=True)

df_all_w_stitching["Localization_Duration_Ratio"] = (df_all_w_stitching["Loc_Duration"]/(df_all_w_stitching["interAnchorDist"]/DEFAULT_SPEED))
df_all_wo_stitching["Localization_Duration_Ratio"] = (df_all_wo_stitching["Loc_Duration"]/(df_all_wo_stitching["interAnchorDist"]/DEFAULT_SPEED))


# %% Coverage
df_cov_ecdf, df_error_ecdf, _ = prepare_coverage_error_df(
    dfs_with_method=[
        (centralized_density_filtered_df_w_stitch,   "CENTRALIZED"),
        (distributed_density_filtered_df_w_stitch,   "DISTRIBUTED"),
        (twr_density_df,                             "TWR"),
        (self_loc_density_filtered_df_w_stitch,      "HYPERBOLIC_SELF_LOCALIZATION"),
    ],
    timestamp_col=TIMESTAMP_FORMAT,
    error_col_map={"TWR": "pos_error"},   # all other methods use "hyrasel_pos_error"
    bin_width=0.04,
)


# %% Median Localization Error Pivot
method_label_map = {
    "HYPERBOLIC_SELF_LOCALIZATION": CHIASSON_DESCRIPTOR,
    "CENTRALIZED": CENTRALIZED_HYRASEL_DESCRIPTOR,
    "DISTRIBUTED": DISTRIBUTED_HYRASEL_DESCRIPTOR,
    "TWR": TWR_DESCRIPTOR,
}


coverage_stats = (
    df_cov_ecdf
    .groupby(["interAnchorDist", "Method"])["coverage"]
    .agg(
        n="count",
        mean="mean",
        std="std",
        min="min",
        q25=lambda x: x.quantile(0.25),
        median="median",
        q75=lambda x: x.quantile(0.75),
        max="max",
    )
    .reset_index()
)

coverage_stats["Method"] = (
    coverage_stats["Method"]
    .map(method_label_map)
    .fillna(coverage_stats["Method"])
)

print("\nCoverage Statistics:")
print(
    coverage_stats.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}"
    )
)


# %% Plot EDCF
plot_coverage_ecdf(
    df_cov_ecdf=df_cov_ecdf,
    df_error_ecdf=df_error_ecdf,
    xlabel_cov=COVERAGE_DESCRIPTOR,
    xlabel_err=ACCURACY_DESCRIPTOR,
    ylabel=CDF_DESCRIPTOR,
    label_map={
        "CENTRALIZED":                  CENTRALIZED_HYRASEL_DESCRIPTOR,
        "DISTRIBUTED":                  DISTRIBUTED_HYRASEL_DESCRIPTOR,
        "HYPERBOLIC_SELF_LOCALIZATION": CHIASSON_DESCRIPTOR,
        "TWR":                          TWR_DESCRIPTOR,
    },
    save_path="Figure_06_Coverage_Facet",
    fig_height=ROW_HEIGHT*1.5,
    show_fig=True,
    legend_ncol=4,
)

#####################################   
#   ____                      _     #
#  / ___| _ __   ___  ___  __| |    #
#  \___ \| '_ \ / _ \/ _ \/ _` |    #
#   ___) | |_) |  __/  __/ (_| |    #
#  |____/| .__/ \___|\___|\__,_|    #
#        |_|                        #
#####################################

# %% SPEED
twr_speed_df = load_results_from_csv(["TWR_" + TIMESTAMP_FORMAT + "_speed_2", 
                                      "TWR_" + TIMESTAMP_FORMAT + "_speed_5", 
                                      "TWR_" + TIMESTAMP_FORMAT + "_speed_10", 
                                      "TWR_" + TIMESTAMP_FORMAT + "_speed_25",
                                      "TWR_" + TIMESTAMP_FORMAT + "_speed_50"
                                      ], 
                                      append_parameters=[2, 5, 10, 25, 50], 
                                      append_value="speed"
                                      )
distributed_speed_df = load_results_from_csv(["DISTRIBUTED_" + TIMESTAMP_FORMAT + "_speed_2_with_vtwr", 
                                       "DISTRIBUTED_" + TIMESTAMP_FORMAT + "_speed_5_with_vtwr", 
                                       "DISTRIBUTED_" + TIMESTAMP_FORMAT + "_speed_10_with_vtwr", 
                                       "DISTRIBUTED_" + TIMESTAMP_FORMAT + "_speed_25_with_vtwr",
                                       "DISTRIBUTED_" + TIMESTAMP_FORMAT + "_speed_50_with_vtwr"
                                       ])
centralized_speed_df = load_results_from_csv(["CENTRALIZED_" + TIMESTAMP_FORMAT + "_speed_2_with_vtwr", 
                                       "CENTRALIZED_" + TIMESTAMP_FORMAT + "_speed_5_with_vtwr", 
                                       "CENTRALIZED_" + TIMESTAMP_FORMAT + "_speed_10_with_vtwr", 
                                       "CENTRALIZED_" + TIMESTAMP_FORMAT + "_speed_25_with_vtwr",
                                       "CENTRALIZED_" + TIMESTAMP_FORMAT + "_speed_50_with_vtwr"
                                       ])

# %% Filtering
k = K
grouping_list = ["speed"]
distributed_speed_filtered_df = filter_results(distributed_speed_df, 
                            grouping_list, 
                            k=k, 
                            filter_residuals=True, 
                            filter_pdop=True, 
                            pdop_factor=2.0,
                            filter_anchor_stitching=False,
                            merge_vtwr_and_self_loc=True, 
                            verbose=True
                            )

centralized_speed_filtered_df = filter_results(centralized_speed_df, 
                            grouping_list, 
                            k=k, 
                            filter_residuals=True, 
                            filter_pdop=True, 
                            pdop_factor=2.0,
                            filter_anchor_stitching=False,
                            merge_vtwr_and_self_loc=True, 
                            verbose=True
                            )

# %%
plot_mean_with_errorbar(
    datasets=[
        {"df": centralized_speed_filtered_df, "x": "speed", "y": "hyrasel_pos_error", "label": CENTRALIZED_HYRASEL_DESCRIPTOR},
        {"df": distributed_speed_filtered_df, "x": "speed", "y": "hyrasel_pos_error", "label": DISTRIBUTED_HYRASEL_DESCRIPTOR},
        {"df": centralized_speed_filtered_df, "x": "speed", "y": "v_twr_error", "label": CENTRALIZED_VTWR_DESCRIPTOR},
        {"df": distributed_speed_filtered_df, "x": "speed", "y": "v_twr_error", "label": DISTRIBUTED_VTWR_DESCRIPTOR},
        {"df": twr_speed_df,  "x": "speed", "y": "pos_error", "label": TWR_DESCRIPTOR},
    ],
    xlabel="Speed [m/s]",
    ylabel=ACCURACY_DESCRIPTOR,
    xticks=[2, 5, 10, 25, 50],
    show_errorbars=True, 
    save_path="Figure_07_error_vs_speed", 
    use_rep_means = True, 
    legend_ncol= 2, 
    FIG_HEIGHT=ROW_HEIGHT*1.1,
    FIG_WIDTH=COL_WIDTH
    )
