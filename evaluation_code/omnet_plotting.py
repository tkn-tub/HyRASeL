#############################################
#   ___                            _        #
#  |_ _|_ __ ___  _ __   ___  _ __| |_ ___  #
#   | || '_ ` _ \| '_ \ / _ \| '__| __/ __| #
#   | || | | | | | |_) | (_) | |  | |_\__ \ #
#  |___|_| |_| |_| .__/ \___/|_|   \__|___/ #
#                |_|                        #
#############################################
# %%
from omnet_data_visualization import *
import matplotlib as mpl

#      ____                                _            
#     |  _ \ __ _ _ __ __ _ _ __ ___   ___| |_ ___ _ __ 
#     | |_) / _` | '__/ _` | '_ ` _ \ / _ \ __/ _ \ '__|
#     |  __/ (_| | | | (_| | | | | | |  __/ ||  __/ |   
#     |_|   \__,_|_|  \__,_|_| |_| |_|\___|\__\___|_|   
#    
#                                                    
# %% 
DISTRIBUTED_HYRASEL_DESCRIPTOR = "HyRASeL (Distr.)" #"HyRASeL (Distributed)"
CENTRALIZED_HYRASEL_DESCRIPTOR = "HyRASeL (Centr.)" #"HyRASeL (Centralized)"
DISTRIBUTED_VTWR_DESCRIPTOR = "v-TWR (Distr.)" # HyRASeL 
CENTRALIZED_VTWR_DESCRIPTOR = "v-TWR (Centr.)" #HyRASeL
TWR_DESCRIPTOR = "TWR (Baseline)"
CHIASSON_DESCRIPTOR = "Chiasson et al. (Baseline)"


ACCURACY_DESCRIPTOR = "Localization-Error [m]"
COVERAGE_DESCRIPTOR = "Coverage Fraction"
DURATION_DESCRIPTOR = "Absolute Localization Duration Per Node [s]"
DURATION_RATIO_DESCRIPTOR = "Localization Service Ratio"
INTER_UPDATE_TIME_DESCRIPTOR = "Inter-Update Interval [s]"
FREQUENCY_DESCRIPTOR = "Update Frequency [Hz]"
NUM_NODES_DESCRIPTOR = "Number of Nodes"
NUM_SLOTS_DESCRIPTOR = "Number of v-Anchor Slots"
DENSITY_DESCRIPTIOR = "Inter-Anchor Distance [m]"
INSTANT_FREQUENCY_DESCRIPTOR = "Instantaneous Update Frequency [Hz]"
CDF_DESCRIPTOR = "CDF"
TIMESTAMP_FORMAT = "noisy_timestamp" # clock_timestamp; sim_timestamp
K = 3
DEFAULT_SPEED = 25

FONT_SIZE      = 10          # pt 
COL_WIDTH      = 3.5         # in  – single column
COL_WIDTH_2    = 7.16        # in  – full-width
ROW_HEIGHT     = COL_WIDTH/1.2         # in 


mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
})

#####################################################
#   _____                 _   _                     #
#  |  ___|   _ _ __   ___| |_(_) ___  _ __  ___     #
#  | |_ | | | | '_ \ / __| __| |/ _ \| '_ \/ __|    #
#  |  _|| |_| | | | | (__| |_| | (_) | | | \__ \    #
#  |_|   \__,_|_| |_|\___|\__|_|\___/|_| |_|___/    #
#####################################################                                      

# %%

def rmse(x):
    return np.sqrt(np.mean(x**2))


def aggregate_info_per_node(
        df: pd.DataFrame,
        grouping: list, 
        error_col: str,
        method_display_name: str,
        timestamp_format: str = "sim_timestamp" 
) -> pd.DataFrame:
    
    # Group
    grouped_df = df.groupby(
        grouping
    ).agg(
        loc_count=(error_col, "count"),
        mean_error=(error_col, "mean"),
        rmse_error=(error_col, rmse),
        first_loc = (timestamp_format, "min"),
        last_loc = (timestamp_format, "max"),
    ).reset_index()
    
    grouped_df["Method"] = method_display_name
    grouped_df["Loc_Duration"] = grouped_df["last_loc"] - grouped_df["first_loc"]
    grouped_df["Mean_Frequency"] = grouped_df["loc_count"] / grouped_df["Loc_Duration"]

    return grouped_df






def filter_results(
        df: pd.DataFrame,
        grouping: list[str] = ["Method"],
        k: float = 10.0,
        filter_residuals: bool = True, 
        filter_pdop: bool = True,
        pdop_factor: float = 2.0,
        filter_anchor_stitching: bool = True, 
        merge_vtwr_and_self_loc: bool = True,
        verbose: bool = True,
):
    filtered_df = df.copy()
    if filter_residuals:
        filtered_df = filter_residual_outliers_per_parameter(filtered_df, grouping, k=k, verbose=verbose)

    if filter_pdop:
        filtered_df = filter_pdop_values(filtered_df, pdop_factor=pdop_factor, verbose=verbose)

    if filter_anchor_stitching: 
        filtered_df = filter_anchor_stitching_values(filtered_df, verbose=verbose)

    if merge_vtwr_and_self_loc: 
        filtered_df = merge_exclusive_columns(filtered_df, col1 = "self_loc_error", col2="v_twr_error")

    return filtered_df

def filter_pdop_values(df: pd.DataFrame, pdop_factor: float = 2.0, verbose: bool = True):
    len_before = len(df)
    pdop_ok = (df["PDOP"] < pdop_factor) | df["PDOP"].isna()
    df_filtered = df.loc[pdop_ok]
    len_after = len(df_filtered)
    if verbose:
        print(f"Removed {len_before-len_after} rows due to PDOP; (from {len_before} to {len_after})")
    return df_filtered

def filter_anchor_stitching_values(df: pd.DataFrame, verbose: bool = True):
    len_before = len(df)
    keep_mask = df["using_anchor_stitching"] == False
    df_filtered = df.loc[keep_mask]
    len_after = len(df_filtered)
    if verbose:
        print(f"Removed {len_before-len_after} rows that used anchor stitching; (from {len_before} to {len_after})")
    return df_filtered

def plot_boxplot(df: pd.DataFrame, 
                 x: str, 
                 y: str, 
                 hue: str, 
                 xlabel: str = None,
                 ylabel: str = None, 
                 xlog: bool = False,
                 ylog: bool = False, 
                 showfliers: bool = True, 
                 legend_ncol: int = 2,
                 legend_above: bool = True, 
                 save_path: str = None,
                 legend_title: str = None, 
                 title: str = None 
                 )-> None:
    plt.figure()
    sns.boxplot(data=df, x=x, y=y, hue=hue, showfliers=showfliers)
 
    plt.title(title)
    if xlog: 
        plt.xscale("log")
    if ylog: 
        plt.yscale("log")

    plt.xlabel(xlabel or x)
    plt.ylabel(ylabel or y)
 
    if legend_above:
        plt.legend(
            title=legend_title,
            ncol=legend_ncol,
            bbox_to_anchor=(0.5, 1.0),
            loc="lower center",
        )
    else:
        plt.legend(title=legend_title, ncol=legend_ncol)
 
    if save_path is not None:
        Path("tmp/figs").mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%y%m%d_%H%M")
        plt.savefig(
            f"tmp/figs/{timestamp}_{save_path}.pdf",
            dpi=300,
            bbox_inches="tight",
        )
 
    plt.show()


def prepare_coverage_error_df(
    dfs_with_method: list[tuple[pd.DataFrame, str]],
    timestamp_col: str,
    error_col_map: dict[str, str],          # {"TWR": "pos_error", ...: "hyrasel_pos_error"}
    bin_width: float = 0.04,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns (df_cov_ecdf, df_error_ecdf, df_plot) ready for plot_coverage_ecdf().
    
    dfs_with_method: [(df, "TWR"), (df, "CENTRALIZED"), ...]
    error_col_map:   maps method name → source error column
    """
    frames = []
    for df, method in dfs_with_method:
        d = df.copy()
        d["Method"] = method
        src = error_col_map.get(method, "hyrasel_pos_error")
        d["_error"] = d[src]
        frames.append(d)

    df = pd.concat(frames, ignore_index=True)
    df_valid = df[df["_error"].notna()].copy()
    df_valid["time_bin"] = np.floor(df_valid[timestamp_col] / bin_width) * bin_width

    N_nodes = df["NodeId"].nunique()

    # Coverage
    df_node = (
        df_valid
        .groupby(["Method", "Rep", "interAnchorDist", "time_bin", "NodeId"])
        .size()
        .reset_index(name="n_loc")
    )
    df_node["localized"] = df_node["n_loc"] >= 1

    df_cov_run = (
        df_node
        .groupby(["Method", "Rep", "interAnchorDist", "time_bin"], as_index=False)
        .agg(n_nodes_localized=("localized", "sum"))
    )
    df_cov_run["coverage"] = df_cov_run["n_nodes_localized"] / N_nodes

    #  Error 
    df_error_ecdf = df_valid[
        ["Method", "Rep", "interAnchorDist", "time_bin", "NodeId", "_error"]
    ].rename(columns={"_error": "hyrasel_pos_error"}).copy()

    #  Aggregated plot-df 
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
            error_mean   =("mean_error",   "mean"),
            error_sd     =("mean_error",   "std"),
            error_median =("median_error", "mean"),
            error_q25    =("median_error", lambda x: x.quantile(0.25)),
            error_q75    =("median_error", lambda x: x.quantile(0.75)),
        )
    )

    df_plot = pd.merge(df_cov_plot, df_error_plot,
                       on=["Method", "interAnchorDist", "time_bin"], how="inner")

    return df_cov_run, df_error_ecdf, df_plot


def plot_coverage_ecdf(
    df_cov_ecdf:   pd.DataFrame,
    df_error_ecdf: pd.DataFrame,
    hyperparams:   list | None = None,
    descriptors:   list[str]   = ("Dense", "Semi-Dense", "Sparse"),
    label_map:     dict | None = None,
    xlabel_cov:    str = "Coverage",
    xlabel_err:    str = "Localization Error [m]",
    ylabel:        str = "CDF",
    xlim_err:      tuple = (0, 1),
    save_path:     str | None = None,
    fig_width_per_col: float = 5,
    FIG_WIDTH = COL_WIDTH_2,
    FIG_HEIGHT = ROW_HEIGHT*1.8,
    legend_ncol = 2,
    show_fig = False
):
    """
    2-row ECDF grid  (top: coverage, bottom: error).

    Parameters
    ----------
    df_cov_ecdf    Output of prepare_coverage_error_df() – one row per (Method, Rep, IAD, bin).
    df_error_ecdf  Output of prepare_coverage_error_df() – one row per localization.
    hyperparams    List of interAnchorDist values to plot (default: all, sorted).
    descriptors    Human-readable subtitle per column.
    label_map      Maps raw Method strings to display names.
    save_path      Stem of the output file (saved under tmp/figs/).  None → no save.
    """
    # Generated with AI
    sns.set_theme(style="whitegrid")

    hyperparams = hyperparams or sorted(df_cov_ecdf["interAnchorDist"].unique())
    methods     = sorted(df_cov_ecdf["Method"].unique())
    palette     = sns.color_palette("tab10", len(methods))
    linestyles  = ["-", "--", "-.", ":"]
    ecdf_ticks  = np.arange(0, 1.01, 0.25)

    fig, axes = plt.subplots(
        nrows=2, ncols=len(hyperparams),
        figsize=(FIG_WIDTH, FIG_HEIGHT),
        constrained_layout=True,
        sharey="row", 
        sharex=False,
    )

    fig.set_constrained_layout_pads(
        w_pad=0.01,
        h_pad=0.01,
        wspace=0.01,
        hspace=0.03,
    )

    for j, (hp, descr) in enumerate(zip(hyperparams, descriptors)):
        ax_cov, ax_err = axes[0, j], axes[1, j]

        for i, method in enumerate(methods):
            kw = dict(color=palette[i], linestyle=linestyles[i % 4], linewidth=2)
            mask = lambda d: (d["interAnchorDist"] == hp) & (d["Method"] == method)

            d_cov = df_cov_ecdf[mask(df_cov_ecdf)]
            if not d_cov.empty:
                sns.ecdfplot(data=d_cov, x="coverage",           ax=ax_cov, label=method, **kw)

            d_err = df_error_ecdf[mask(df_error_ecdf)]
            if not d_err.empty:
                sns.ecdfplot(data=d_err, x="hyrasel_pos_error",  ax=ax_err, **kw)

        ax_cov.set_title(f"{descr} ({hp} m)", fontsize=FONT_SIZE-2)
        ax_cov.set_xlabel(xlabel_cov, fontsize=FONT_SIZE-2)  
        ax_err.set_xlabel(xlabel_err, fontsize=FONT_SIZE-2)
        # Tick-Labels
        ax_cov.tick_params(labelsize=FONT_SIZE-2)
        ax_err.tick_params(labelsize=FONT_SIZE-2)

        if j == 0:
            ax_cov.set_ylabel(ylabel, fontsize=FONT_SIZE-2)  
            ax_err.set_ylabel(ylabel, fontsize=FONT_SIZE-2)

        for ax, xlim in [(ax_cov, (0, 1.01)), (ax_err, xlim_err)]:
            ax.set_xlim(xlim)
            ax.set_ylim((0, 1.01))
            ax.set_yticks(ecdf_ticks) 
            ax.grid(True, alpha=0.3)

        ax_cov.set_xticks(ecdf_ticks)

    # Shared legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if label_map:
        labels = [label_map.get(l, l) for l in labels]
    fig.legend(
        handles, 
        labels, 
        loc="lower center",
        # ncol=len(methods),
        ncol=legend_ncol, 
        frameon=False, 
        columnspacing=0.2,
        bbox_to_anchor=(0.5, 1.00),
        fontsize=FONT_SIZE - 2,
        )

    if save_path:
        Path("tmp/figs").mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%y%m%d_%H%M")
        plt.savefig(f"tmp/figs/{ts}_{save_path}.pdf", bbox_inches="tight")
    if show_fig: 
        plt.show()



def plot_mean_with_errorbar(
    datasets: list[dict],
    xlabel=None, ylabel=None,
    xlim=None, ylim=None,
    xticks=None, yticks=None,
    show_errorbars: bool = True, 
    verbose: bool = True, 
    save_path: str= None,
    use_rep_means: bool =  False,
    legend_above: bool= True,
    legend_title: str = None,
    legend_ncol: int = 2, 
    FIG_WIDTH = COL_WIDTH,
    FIG_HEIGHT = ROW_HEIGHT
):
    linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
    markers    = ["o", "s", "^", "D", "x"]

    fig, ax = plt.subplots(
        figsize=(FIG_WIDTH, FIG_HEIGHT),
        constrained_layout=True,
    )

    for i, ds in enumerate(datasets):
        filter_col = ds.get("filter_col", ds["y"])
        df = ds["df"].loc[ds["df"][filter_col].notna()]


        if not use_rep_means: 
            grouped = df.groupby(ds["x"])[ds["y"]]
            means   = grouped.mean()
            stds    = grouped.std()
            counts  = grouped.count()
            sem = stds / np.sqrt(counts)
            if verbose: 
                print(f"\n--- {ds['label']} ---")
                print(f"Rows pro Gruppe:\n{counts}")
                print(f"Mean:\n{means}")
                print(f"Std:\n{stds}")
                print(f"SEM:\n{sem}")

        else: 
            rep_means = df.groupby([ds["x"], "Rep"])[ds["y"]].mean().unstack("Rep")
            means   = rep_means.mean(axis=1)
            stds    = rep_means.std(axis=1)
            n_reps = rep_means.shape[1]
            sem = stds / np.sqrt(n_reps)
            if verbose: 
                print(f"\n--- {ds['label']} ---")
                print(f"Reps:\n{n_reps}")
                print(f"Mean:\n{means}")
                print(f"Std:\n{stds}")
                print(f"SEM:\n{sem}")

        if show_errorbars:
            ax.errorbar(
                means.index, means.values, yerr=sem.values,
                label=ds["label"],
                linestyle=linestyles[i % len(linestyles)],
                marker=markers[i % len(markers)],
                capsize=4,
            )
        else:
            ax.plot(
                means.index, means.values,
                label=ds["label"],
                linestyle=linestyles[i % len(linestyles)],
                marker=markers[i % len(markers)],
            )

    ax.set_xlabel(xlabel or "", fontsize=FONT_SIZE-2)
    ax.set_ylabel(ylabel or "", fontsize=FONT_SIZE-2)
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
            fontsize=FONT_SIZE-2
        )
    else:
        ax.legend(title=legend_title, ncol=legend_ncol)

    ax.tick_params(labelsize=FONT_SIZE-2)
    ax.grid(True)
    if xticks is not None: ax.set_xticks(xticks)
    if yticks is not None: ax.set_yticks(yticks)
    if xlim   is not None: ax.set_xlim(xlim)
    if ylim   is not None: ax.set_ylim(ylim)

    if save_path is not None:
        Path("tmp/figs").mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%y%m%d_%H%M")
        fig.savefig(
            f"tmp/figs/{timestamp}_{save_path}.pdf", 
            dpi=300,
            bbox_inches="tight",
            )

#############################################
#   ____                 _ _                #
#  | __ )  __ _ ___  ___| (_)_ __   ___     #
#  |  _ \ / _` / __|/ _ \ | | '_ \ / _ \    #   
#  | |_) | (_| \__ \  __/ | | | | |  __/    #
#  |____/ \__,_|___/\___|_|_|_| |_|\___|    #
#############################################                                     

# %% BASELINE
twr_response_df = load_results_from_csv(["TWR_"+TIMESTAMP_FORMAT])
centralized_self_loc_response_df = load_results_from_csv(["CENTRALIZED_" + TIMESTAMP_FORMAT +"_with_vtwr"])
distributed_self_loc_response_df = load_results_from_csv(["DISTRIBUTED_"+ TIMESTAMP_FORMAT +"_with_vtwr"])


# Sensitivity for outlier removal
k = K

filtered_distributed_speed_df = filter_results(distributed_self_loc_response_df, 
                                     ["Method"], 
                                     k=k, 
                                     filter_residuals=True, 
                                     filter_pdop=True, 
                                     pdop_factor=2.0,
                                     filter_anchor_stitching=True,
                                     merge_vtwr_and_self_loc=True, 
                                     verbose=True
                                     )

filtered_centralized_speed_df = filter_results(centralized_self_loc_response_df, 
                                     ["Method"], 
                                     k=k, 
                                     filter_residuals=True, 
                                     filter_pdop=True, 
                                     pdop_factor=2.0,
                                     filter_anchor_stitching=True,
                                     merge_vtwr_and_self_loc=True, 
                                     verbose=True
                                     )
filtered_centralized_speed_w_stitch_df = filter_results(centralized_self_loc_response_df, 
                                     ["Method"], 
                                     k=k, 
                                     filter_residuals=True, 
                                     filter_pdop=True, 
                                     pdop_factor=2.0,
                                     filter_anchor_stitching=False,
                                     merge_vtwr_and_self_loc=True, 
                                     verbose=True
                                     )

# %% Accuracy/Error BASELINE
#! PLOT CDF of localzation error / Maybe CDF per node (to show differences)
plot_ecdf(
    datasets=[
        {"df": twr_response_df, "x": "pos_error", "label": "TWR (Baseline)"},
        {"df": filtered_centralized_speed_df, "x": "hyrasel_pos_error", "label": "HyRASeL (Centralized)"},
        {"df": filtered_centralized_speed_w_stitch_df, "x": "hyrasel_pos_error", "label": "HyRASeL (Centralized w/ stitching)"},
        {"df": filtered_distributed_speed_df, "x": "hyrasel_pos_error", "label": "HyRASeL (Distributed)"},
        {"df": centralized_self_loc_response_df, "x": "v_twr_error", "label": "HyRASeL v-TWR (Centralized)"},
    ],
    xlabel=ACCURACY_DESCRIPTOR,
    ylabel=CDF_DESCRIPTOR,
    yticks=[0, 0.25, 0.5, 0.75, 1.0],
    # save_path="BASELINE_Accuracy_CDF",
    # xticks=[x/10 for x in range(0, 21, 2)],
    xlim=(0, 1.0)
)

# %% Inter-Localization-Time BASELINE
plot_ecdf(
    datasets=[
        {"df": compute_inter_measurement_time(twr_response_df, "pos_error", TIMESTAMP_FORMAT), "x": "inter_measurement_time", "label": "TWR (Baseline)"},
        {"df": compute_inter_measurement_time(filtered_centralized_speed_df, "hyrasel_pos_error", TIMESTAMP_FORMAT), "x": "inter_measurement_time", "label": "HyRASeL (Centralized)"},
        {"df": compute_inter_measurement_time(filtered_centralized_speed_w_stitch_df, "hyrasel_pos_error", TIMESTAMP_FORMAT), "x": "inter_measurement_time", "label": "HyRASeL (Centralized w/stitching)"},
        {"df": compute_inter_measurement_time(filtered_distributed_speed_df, "hyrasel_pos_error", TIMESTAMP_FORMAT), "x": "inter_measurement_time", "label": "HyRASeL (Distributed)"},
        # {"df": compute_inter_measurement_time(centralized_self_loc_response_df, "v_twr_error", TIMESTAMP_FORMAT), "x": "inter_measurement_time", "label": "HyRASeL v-TWR (Centralized)"},
    ],
    xlabel=INTER_UPDATE_TIME_DESCRIPTOR,
    ylabel=CDF_DESCRIPTOR,
    yticks=[0, 0.25, 0.5, 0.75, 1.0],
    # save_path="BASELINE_ILT_CDF",
    # xticks=[x/10 for x in range(0, 21, 2)],
    xlim=(0, 1.0)
)
# %% Update Frequency BASELINE
plot_ecdf(
    datasets=[
        {"df": compute_inter_measurement_time(twr_response_df, "pos_error", TIMESTAMP_FORMAT), "x": "update_frequency", "label": "TWR (Baseline)"},
        {"df": compute_inter_measurement_time(centralized_self_loc_response_df, "self_loc_error", TIMESTAMP_FORMAT), "x": "update_frequency", "label": "HyRASeL (Centralized)"},
        {"df": compute_inter_measurement_time(distributed_self_loc_response_df, "self_loc_error", TIMESTAMP_FORMAT), "x": "update_frequency", "label": "HyRASeL (Distributed)"},
        {"df": compute_inter_measurement_time(centralized_self_loc_response_df, "v_twr_error", TIMESTAMP_FORMAT), "x": "update_frequency", "label": "HyRASeL v-TWR (Centralized)"},
    ],
    xlabel=INSTANT_FREQUENCY_DESCRIPTOR,
    ylabel=CDF_DESCRIPTOR,
    yticks=[0, 0.25, 0.5, 0.75, 1.0],
    # save_path="BASELINE_instant_frequency_CDF",
    # xticks=[x/10 for x in range(0, 21, 2)],
    # xlim=(0, 0.4)
)


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
        # {"df": distributed_slots_filtered_df, "x": "numSlots", "y": "self_loc_error", "label": "HyRASeL (Distributed/Self-Loc only)"},
        # {"df": centralized_slots_filtered_df, "x": "numSlots", "y": "self_loc_error", "label": "HyRASeL (Centralized/Self-Loc only)"},
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
    # legend_above=False
    )


# %% Inter Loc Time
plot_mean_with_errorbar(
    datasets=[
        {"df": compute_inter_measurement_time(centralized_slots_filtered_df, "hyrasel_pos_error", TIMESTAMP_FORMAT), "x": "numSlots", "y": "inter_measurement_time", "label": CENTRALIZED_HYRASEL_DESCRIPTOR},
        {"df": compute_inter_measurement_time(distributed_slots_filtered_df, "hyrasel_pos_error", TIMESTAMP_FORMAT), "x": "numSlots", "y": "inter_measurement_time", "label": DISTRIBUTED_HYRASEL_DESCRIPTOR},
        # {"df": compute_inter_measurement_time(distributed_slots_filtered_df, "self_loc_error", TIMESTAMP_FORMAT), "x": "numSlots", "y": "inter_measurement_time", "label": "HyRASeL (Distributed/Self-Loc only)"},
        # {"df": compute_inter_measurement_time(centralized_slots_filtered_df, "self_loc_error", TIMESTAMP_FORMAT), "x": "numSlots", "y": "inter_measurement_time", "label": "HyRASeL (Centralized/Self-Loc only)"},
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
    # FIG_WIDTH=10
    # legend_above=False
    )

# %%
# plot_boxplot(df_all, 
#              x="numSlots", 
#              y="loc_count",
#              hue="Method",
#              xlabel= "Number of v-Anchor Slots",
#              ylabel="Successful localizations per Node",
#             #  save_path="Method_vs_NumberOfLocs_for_numSlots"
#              )
# # %% Mean Error 
# plot_boxplot(df_all, 
#              x="numSlots", 
#              y="mean_error",
#              hue="Method",
#              xlabel= "Number of v-Anchor Slots",
#              ylabel="Mean Localization Error per Node [m]", 
#              showfliers=False, 
#             #  save_path="Method_vs_Error_for_numSlots"
#              )

# %% Mean Localization Duration
# plot_boxplot(df_all, 
#              x="numSlots", 
#              y="Loc_Duration",
#              hue="Method",
#              xlabel= "Number of v-Anchor Slots",
#              ylabel="Absolute localization duration per node [s]", 
#              showfliers=True, 
#              save_path="Method_vs_Duration_for_NumSlots"
#              )

# # %% Mean Frequency 
# plot_boxplot(df_all, 
#              x="numSlots", 
#              y="Mean_Frequency",
#              hue="Method",
#              xlabel= "Number of v-Anchor Slots",
#              ylabel="Mean Location Update Frequency [Hz]", 
#              showfliers=False, 
#              save_path="Method_vs_Frequency_for_numSlots"
#              )



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
    save_path="Figure_05a_ILT_vs_numNodes", 
    use_rep_means = True,
    legend_ncol=2,
    FIG_HEIGHT=ROW_HEIGHT*1.2
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
    save_path="Figure_05b_error_vs_nodeNum", 
    use_rep_means = True,
    legend_ncol=2,
    FIG_HEIGHT=ROW_HEIGHT*1.2
)


# # %% instant update frequency vs. number of nodes
# plot_mean_with_errorbar(
#     datasets=[
#         {"df": compute_inter_measurement_time(twr_nodes_df,"pos_error", TIMESTAMP_FORMAT), "x": "numNodes", "y": "update_frequency", "label": "TWR (Baseline)"},
#         {"df": compute_inter_measurement_time(distributed_nodes_filtered_df, "hyrasel_pos_error", TIMESTAMP_FORMAT), "x": "numNodes", "y": "update_frequency", "label": "HyRASeL (Distributed)"},
#         {"df": compute_inter_measurement_time(centralized_nodes_filtered_df, "hyrasel_pos_error", TIMESTAMP_FORMAT), "x": "numNodes", "y": "update_frequency", "label": "HyRASeL (Centralized)"},
#     ],
#     xlabel="Number of Nodes",
#     ylabel="Update Frequency [Hz]",
#     xticks=[4, 9, 16, 25, 36],
#     save_path="frequency_vs_numNodes", 
#     use_rep_means = True
#     )


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

# %% Mean Localization Duration with and without anchor stitching

# plot_boxplot(df_all_w_stitching, 
#              x="interAnchorDist", 
#              y="Loc_Duration",
#              hue="Method",
#              xlabel= DENSITY_DESCRIPTIOR,
#              ylabel= DURATION_DESCRIPTOR, 
#              showfliers=True, 
#             #  save_path="Figure_05_Method_vs_Duration_for_IADs_w_stitching"
#              )
# plot_boxplot(df_all_wo_stitching, 
#              x="interAnchorDist", 
#              y="Loc_Duration",
#              hue="Method",
#              xlabel= DENSITY_DESCRIPTIOR,
#              ylabel= DURATION_DESCRIPTOR, 
#              showfliers=True, 
#             #  save_path="Figure_06_Method_vs_Duration_for_IADs_wo_stitching"
#              )


plot_boxplot(df_all_w_stitching, 
             x="interAnchorDist", 
             y="Localization_Duration_Ratio",
             hue="Method",
             xlabel= DENSITY_DESCRIPTIOR,
             ylabel= DURATION_RATIO_DESCRIPTOR, 
             showfliers=True, 
            #  save_path="Figure_05_Method_vs_Duration_Ratio_for_IADs_w_stitching"
             )
plot_boxplot(df_all_wo_stitching, 
             x="interAnchorDist", 
             y="Localization_Duration_Ratio",
             hue="Method",
             xlabel= DENSITY_DESCRIPTIOR,
             ylabel= DURATION_RATIO_DESCRIPTOR, 
             showfliers=True, 
            #  save_path="Figure_06_Method_vs_Duration_Ratio_for_IADs_wo_stitching"
             )


# %% Mean Error with and without anchor stitching
plot_boxplot(df_all_w_stitching, 
             x="interAnchorDist", 
             y="mean_error",
             hue="Method",
             xlabel= DENSITY_DESCRIPTIOR,
             ylabel=ACCURACY_DESCRIPTOR, 
             showfliers=False, 
            #  save_path="Figure_07_Method_vs_Error_for_IADs_w_stitching",
            #  ylog=True
             )
plot_boxplot(df_all_wo_stitching, 
             x="interAnchorDist", 
             y="mean_error",
             hue="Method",
             xlabel= DENSITY_DESCRIPTIOR,
             ylabel= ACCURACY_DESCRIPTOR, 
             showfliers=False, 
            #  save_path="Figure_08_Method_vs_Error_for_IADs_wo_stitching"
             )



# %% Mean Frequency with and without stitching
plot_boxplot(df_all_w_stitching, 
             x="interAnchorDist", 
             y="Mean_Frequency",
             hue="Method",
             xlabel= DENSITY_DESCRIPTIOR,
             ylabel= FREQUENCY_DESCRIPTOR, 
             showfliers=False, 
            #  save_path="Figure_09_Method_vs_Frequency_for_IADs_w_stitching"
             )
plot_boxplot(df_all_wo_stitching, 
             x="interAnchorDist", 
             y="Mean_Frequency",
             hue="Method",
             xlabel= DENSITY_DESCRIPTIOR,
             ylabel=FREQUENCY_DESCRIPTOR, 
             showfliers=False, 
            #  save_path="Figure_10_Method_vs_Frequency_for_IADs_wo_stitching"
             )


# %% Coverage
df_cov_ecdf, df_error_ecdf, _ = prepare_coverage_error_df(
    dfs_with_method=[
        (centralized_density_filtered_df_w_stitch,   "CENTRALIZED"),
        (distributed_density_filtered_df_w_stitch,   "DISTRIBUTED"),
        (twr_density_df,                             "TWR"),
        (self_loc_density_filtered_df_w_stitch,      "HYPERBOLIC_SELF_LOCALIZATION"),
    ],
    timestamp_col=TIMESTAMP_FORMAT,
    error_col_map={"TWR": "pos_error"},   # alle anderen → "hyrasel_pos_error"
    bin_width=0.04,
)

# %%

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
    FIG_HEIGHT=ROW_HEIGHT*1.4,
    show_fig=True,
    legend_ncol=4,
)

# %% Direct plotting of inter-measurement-time without mean 
all_list = []
twr_density_df_cp = twr_density_df.copy()
twr_density_df_cp["Method"] = "TWR"

all_list.append(compute_inter_measurement_time(twr_density_df_cp, "pos_error", "noisy_timestamp"))

for dataset in [distributed_density_filtered_df_w_stitch, centralized_density_filtered_df_w_stitch, self_loc_density_filtered_df_w_stitch]:
    all_list.append(compute_inter_measurement_time(dataset, "hyrasel_pos_error", "noisy_timestamp"))

all_list_df = pd.concat(all_list, ignore_index=True)

plot_boxplot(all_list_df, 
             x="interAnchorDist", 
             y="inter_measurement_time", 
             hue="Method", 
             showfliers=False,
             )

plot_boxplot(all_list_df,
             x="interAnchorDist", 
             y="inter_measurement_time", 
             hue="Method", 
             showfliers=True, 
             ylog=True,
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
# for speed in [2, 5, 10, 20]:
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
    # FIG_WIDTH=10
    # legend_above=False
    )


# %%
# plot_mean_with_errorbar(
#     datasets=[
#         {"df": compute_inter_measurement_time(twr_speed_df,"pos_error", TIMESTAMP_FORMAT), "x": "speed", "y": "inter_measurement_time", "label": "TWR (Baseline)"},
#         {"df": compute_inter_measurement_time(distributed_speed_filtered_df, "hyrasel_pos_error", TIMESTAMP_FORMAT), "x": "speed", "y": "inter_measurement_time", "label": "HyRASeL (Distributed)"},
#         {"df": compute_inter_measurement_time(centralized_speed_filtered_df, "hyrasel_pos_error", TIMESTAMP_FORMAT), "x": "speed", "y": "inter_measurement_time", "label": "HyRASeL (Centralized)"},
#     ],
#     xlabel="Speed [m/s]",
#     ylabel="Mean Inter-Localization-Time [s]",
#     xticks=[1, 5, 10, 20],
#     # save_path="interLocTime_vs_numNodes", 
#     use_rep_means = True
#     # show_errorbars=False
#     )


# %% Node Density/Deployment Area Size

twr_node_density_df = load_results_from_csv(["TWR_" + TIMESTAMP_FORMAT + "_size_80",
                                        "TWR_" + TIMESTAMP_FORMAT + "_size_60",
                                        "TWR_" + TIMESTAMP_FORMAT + "_size_40",
                                        "TWR_" + TIMESTAMP_FORMAT + "_size_20"],
                                        append_parameters=[80, 60, 40, 20], append_value="size")

centralized_node_density_df = load_results_from_csv(["CENTRALIZED_" + TIMESTAMP_FORMAT + "_size_80_with_vtwr",
                                           "CENTRALIZED_" + TIMESTAMP_FORMAT + "_size_60_with_vtwr",
                                           "CENTRALIZED_" + TIMESTAMP_FORMAT + "_size_40_with_vtwr", 
                                           "CENTRALIZED_" + TIMESTAMP_FORMAT + "_size_20_with_vtwr", ])

distributed_node_density_df = load_results_from_csv(["DISTRIBUTED_" + TIMESTAMP_FORMAT + "_size_80_with_vtwr",
                                         "DISTRIBUTED_" + TIMESTAMP_FORMAT + "_size_60_with_vtwr",
                                         "DISTRIBUTED_" + TIMESTAMP_FORMAT + "_size_40_with_vtwr", 
                                         "DISTRIBUTED_" + TIMESTAMP_FORMAT + "_size_20_with_vtwr",])


# %% Filtering
k = K
grouping_list = ["size"]

centralized_node_density_filtered_df_w_stitch = filter_results(centralized_node_density_df, 
                                     grouping_list, 
                                     k=k, 
                                     filter_residuals=True, 
                                     filter_pdop=True, 
                                     pdop_factor=2.0,
                                     filter_anchor_stitching=False,
                                     merge_vtwr_and_self_loc=True, 
                                     verbose=True
                                     )

distributed_node_density_filtered_df_w_stitch = filter_results(distributed_node_density_df, 
                                     grouping_list, 
                                     k=k, 
                                     filter_residuals=True, 
                                     filter_pdop=True, 
                                     pdop_factor=2.0,
                                     filter_anchor_stitching=False,
                                     merge_vtwr_and_self_loc=True, 
                                     verbose=True
                                     )


# %% Group by node
groupby_list = ["size", "Rep", "NodeId"]

twr_node_density_grouped_df = aggregate_info_per_node(twr_node_density_df, groupby_list, "pos_error", TWR_DESCRIPTOR, TIMESTAMP_FORMAT)

distributed_node_density_grouped_df_w_stitch = aggregate_info_per_node(distributed_node_density_filtered_df_w_stitch, groupby_list, "hyrasel_pos_error" ,DISTRIBUTED_HYRASEL_DESCRIPTOR, TIMESTAMP_FORMAT)

centralized_node_density_grouped_w_stitch_df =  aggregate_info_per_node(centralized_node_density_filtered_df_w_stitch, groupby_list, "hyrasel_pos_error", CENTRALIZED_HYRASEL_DESCRIPTOR, TIMESTAMP_FORMAT)

df_all_w_stitching = pd.concat([twr_node_density_grouped_df,
                                distributed_node_density_grouped_df_w_stitch, 
                                centralized_node_density_grouped_w_stitch_df], ignore_index=True)
DURATION = 10.8

df_all_w_stitching["Localization_Duration_Ratio"] = df_all_w_stitching["Loc_Duration"]/DURATION

# %% Mean Localization Duration with and without anchor stitching
plot_boxplot(df_all_w_stitching, 
             x="size", 
             y="Localization_Duration_Ratio",
             hue="Method",
             xlabel= "Node Deployment Area Size",
             ylabel= DURATION_RATIO_DESCRIPTOR, 
             showfliers=True, 
             legend_ncol= 3, 
            #  save_path = "Localization_service_ratio_vs_node_Density"
             )

# %% Mean Error with and without anchor stitching
plot_boxplot(df_all_w_stitching, 
             x="size", 
             y="mean_error",
             hue="Method",
             xlabel= "Node Deployment Area Size",
             ylabel=ACCURACY_DESCRIPTOR, 
             showfliers=False, 
             legend_ncol= 3, 
            #  save_path = "Mean_error_vs_node_Density"
            #  ylog=True
             )


# %% RMSE with and without anchor stitching
plot_boxplot(df_all_w_stitching, 
             x="size", 
             y="rmse_error",
             hue="Method",
             xlabel= "Node Deployment Area Size",
             ylabel="RMSE [m]", 
             showfliers=False, 
             legend_ncol= 3, 
            #  save_path = "RMSE_vs_node_Density"
            #  ylog=True
             )


# %% Mean Frequency with and without stitching
plot_boxplot(df_all_w_stitching, 
             x="size", 
             y="Mean_Frequency",
             hue="Method",
             xlabel= "Node Deployment Area Size",
             ylabel= FREQUENCY_DESCRIPTOR, 
             showfliers=False, 
             legend_ncol= 3, 
            #  save_path = "Mean_freq_vs_node_density"
             )