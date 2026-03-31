import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.colors import ListedColormap
import os

PATH_EXPLOG_PREFIX = "/home/amaloo/reucl_moe/.exp_result" # TODO: replace this string with the absolute path of the ".exp_result" folder

def _get_tr_loss(exp_header, typestr='theory_stats', criteria=2):
    d=np.load(f'{PATH_EXPLOG_PREFIX}/tmoecl/log/{exp_header}/stats_run_0.npz')[typestr]
    return d[:, criteria].tolist()

def _get_conv_index(exp_header):
    d=np.load(f'{PATH_EXPLOG_PREFIX}/tmoecl/log/{exp_header}/stats_run_0.npz')['theory_stats'][:,3]
    try:
        _idx = next(i for i, value in enumerate(d) if int(value) == 1)
    except StopIteration:
        _idx = -1
    return _idx

def comp_criterias(exp_header, criteria_dict, is_transparent=False):
    """
    Sample usage:
        exp_header="20241120_C013524K"
        criteria_dict = [
            ('tr_loss', 2, 'Training Loss'),
            ('theory_stats', 2, 'Val1 Loss'),
        ]
        plot.comp_criterias(exp_header, criteria_dict)
    """
    if is_transparent:
        plt.figure(figsize=(10, 6),dpi=300)
    else:
        plt.figure(figsize=(10, 6),dpi=300, facecolor='white')
    
    conv_idx = None
    for typestr, criteria, notes in criteria_dict:
        data = _get_tr_loss(exp_header, typestr=typestr, criteria=criteria)
        data_x = list(range(1,len(data)+1))
        conv_idx = _get_conv_index(exp_header)

        plt.plot(data_x, data, label=notes)
    plt.axvline(x=conv_idx, color='gray', linestyle='--', linewidth=2, label="Convergence Round")

    plt.xlabel('Round')
    plt.ylabel('Error G_t (%)')
    plt.title(f'Comparison of criterias')

    plt.legend()
    plt.grid(True)
    plt.show()

def plot_discrete_data(exp_header_dict, is_transparent=False, title_notes="", typestr='theory_stats', criteria=9, ls=False, is_export=False, export_g="", export_filename=""):
    if is_transparent:
        plt.figure(figsize=(10, 6),dpi=300)
    else:
        plt.figure(figsize=(10, 6),dpi=300, facecolor='white')

    for exp_header, note in exp_header_dict.items():
        data = _get_tr_loss(exp_header, typestr=typestr, criteria=criteria)
        non_zero_data = [value for value in data if value != 0.0]
        x_list = [index for index, value in enumerate(data) if value != 0]
        # x_list = list(range(1,len(data)+1))
        conv_idx = _get_conv_index(exp_header)

        plt.plot(x_list, non_zero_data, 'x--', label=note, marker='2')
        # if conv_idx!=-1:
        #     plt.plot(x_list[conv_idx], data[conv_idx], 'x', color='red', markersize=8)

    plt.xlabel('Round')
    if ls:
        plt.yscale('log')  # Use logarithmic scale for the y-axis

    if criteria==9:
        plt.ylabel('Error G_t (%)')
    elif criteria==10:
        plt.ylabel('Avg. Accuracy (%)')
    if title_notes=="":
        plt.title(f'Discrete Data')
    else:
        # plt.title(f'Training Loss averaged by round ({title_notes})')
        plt.title(f'{title_notes}')

    plt.legend()
    plt.grid(True)

    if is_export:
        if not os.path.exists(export_g):
            os.makedirs(export_g)
        _export_path = os.path.join(export_g, export_filename, '.pdf')
        plt.savefig(os.path.join(), format='pdf')
        print(f"Plot saved to {_export_path}")
    plt.show()


def _plot_testing_accuracy(exp_header_dict, is_transparent=False, title_notes="", ref_acc=-1.0,typestr='theory_stats', criteria=2, ls=False, is_export=False, export_g="", export_filename=""):

    """
    Plot the testing accuracy (indicating generalization) of multi runs (potentially from multi exps) on a single plot for comparison.
    # NOTE REQUIREMENT: all run/exps in this list have the same number of task.
    """

    if is_transparent:
        plt.figure(figsize=(10, 6),dpi=300)
    else:
        plt.figure(figsize=(10, 6),dpi=300, facecolor='white')

    for exp_header, note in exp_header_dict.items():
        data = _get_tr_loss(exp_header, typestr=typestr, criteria=criteria)
        data_x = list(range(1,len(data)+1))
        conv_idx = _get_conv_index(exp_header)

        plt.plot(data_x, data, label=note)
        if conv_idx!=-1:
            plt.plot(data_x[conv_idx], data[conv_idx], 'x', color='red', markersize=8)


    if ref_acc != -1.0:
        plt.axhline(y=ref_acc, color='gold', linestyle='-', linewidth=0.8)

    if ls:
        plt.yscale('log')  # Use logarithmic scale for the y-axis

    plt.xlabel('Round')
    plt.ylabel('Error G_t (%)')
    if title_notes!="":
        plt.title(f'{title_notes}')

    plt.legend()
    plt.grid(True)
    
    if is_export:
        if not os.path.exists(export_g):
            os.makedirs(export_g)
        _export_path = os.path.join(export_g, export_filename, '.pdf')
        plt.savefig(os.path.join(), format='pdf')
        print(f"Plot saved to {_export_path}")
    plt.show()

def compare_plots(exp_header_dict, title_notes="", metric=('tr_loss', 2), ls=False):
    _plot_testing_accuracy(exp_header_dict, title_notes=title_notes, typestr=metric[0], criteria=metric[1], ls=ls)

def plot_expert_usage_over_time(exp_header_dict, is_transparent=False, title_notes=""):
    """
    For each round, plot bars for each expert showing cumulative usage up to that round.
    exp_header_dict: {exp_header: note}
    """
    for exp_header, note in exp_header_dict.items():
        stats_path = f"{PATH_EXPLOG_PREFIX}/tmoecl/log/{exp_header}/stats_run_0.npz"
        stats = np.load(stats_path)
        expert_usage = stats['expert_usage']  # shape: (nb_rounds, nb_experts)
        nb_rounds, nb_experts = expert_usage.shape

        # Compute cumulative usage for each expert up to each round
        cumulative_usage = np.cumsum(expert_usage, axis=0)  # shape: (nb_rounds, nb_experts)

        if is_transparent:
            plt.figure(figsize=(max(10, nb_rounds // 2), 6), dpi=300)
        else:
            plt.figure(figsize=(max(10, nb_rounds // 2), 6), dpi=300, facecolor='white')

        rounds = np.arange(nb_rounds)
        bar_width = 0.8 / nb_experts

        for expert_id in range(nb_experts):
            plt.bar(
                rounds + expert_id * bar_width,
                cumulative_usage[:, expert_id],
                width=bar_width,
                label=f'Expert {expert_id}'
            )

        plt.xlabel('Round')
        plt.ylabel('Cumulative Usage Count')
        plt.title(title_notes if title_notes else f'Cumulative Expert Usage per Round: {note}')
        plt.xticks(rounds + bar_width * (nb_experts - 1) / 2, rounds)
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        plt.tight_layout()
        plt.show()

def plot_expert_usage_total(exp_header_dict, is_transparent=False, title_notes=""):
    """
    Plots a bar chart of total usage count for each expert across all rounds.
    exp_header_dict: {exp_header: note}
    """
    for exp_header, note in exp_header_dict.items():
        stats_path = f"{PATH_EXPLOG_PREFIX}/tmoecl/log/{exp_header}/stats_run_0.npz"
        stats = np.load(stats_path)
        expert_usage = stats['expert_usage']  # shape: (nb_rounds, nb_experts)
        total_usage = expert_usage.sum(axis=0)  # sum over rounds for each expert

        nb_experts = total_usage.shape[0]
        expert_ids = np.arange(nb_experts)

        if is_transparent:
            plt.figure(figsize=(8, 5), dpi=300)
        else:
            plt.figure(figsize=(8, 5), dpi=300, facecolor='white')

        plt.bar(expert_ids, total_usage, color='skyblue')
        plt.xlabel('Expert ID')
        plt.ylabel('Total Usage Count')
        plt.title(title_notes if title_notes else f'Total Expert Usage: {note}')
        plt.xticks(expert_ids)
        plt.tight_layout()
        plt.show()

def plot_expert_usage_lines(exp_header_dict, is_transparent=False, title_notes=""):
    """
    Plots a line chart: each expert is a line, y-axis is cumulative usage, x-axis is round.
    exp_header_dict: {exp_header: note}
    """
    for exp_header, note in exp_header_dict.items():
        stats_path = f"{PATH_EXPLOG_PREFIX}/tmoecl/log/{exp_header}/stats_run_0.npz"
        stats = np.load(stats_path)
        expert_usage = stats['expert_usage']  # shape: (nb_rounds, nb_experts)
        nb_rounds, nb_experts = expert_usage.shape

        cumulative_usage = np.cumsum(expert_usage, axis=0)  # shape: (nb_rounds, nb_experts)
        rounds = np.arange(nb_rounds)

        if is_transparent:
            plt.figure(figsize=(10, 8), dpi=300)
        else:
            plt.figure(plt.figure(figsize=(10, 8)), dpi=300, facecolor='white')

        for expert_id in range(nb_experts):
            plt.plot(rounds, cumulative_usage[:, expert_id], label=f'Expert {expert_id}')

        plt.xlabel('Round')
        plt.ylabel('Cumulative Usage Count')
        plt.title(title_notes if title_notes else f'Cumulative Expert Usage per Round: {note}')
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def plot_expert_load_ratio_vs_parameter(exp_header_dict, param_dict, param_type, is_transparent=False, title_notes=""):
    """
    Plots a line chart: each expert is a line, x-axis is the parameter, y-axis is expert load ratio.
    exp_header_dict: {exp_header: note}
    param_dict: {exp_header: parameter_value}
    """
    # Gather usage ratios for each experiment
    expert_ratios_by_parameter = {}  # {parameter: [ratios for each expert]}
    for exp_header, note in exp_header_dict.items():
        stats_path = f"{PATH_EXPLOG_PREFIX}/tmoecl/log/{exp_header}/stats_run_0.npz"
        stats = np.load(stats_path)
        expert_usage = stats['expert_usage']  # shape: (nb_rounds, nb_experts)
        total_usage = expert_usage.sum(axis=0)
        usage_ratio = total_usage / total_usage.sum()
        param = param_dict[exp_header]
        expert_ratios_by_parameter[param] = usage_ratio

    # Sort by parameter
    sorted_param = sorted(expert_ratios_by_parameter.keys())
    ratios_matrix = np.array([expert_ratios_by_parameter[a] for a in sorted_param])  # shape: (num_params, num_experts)
    num_experts = ratios_matrix.shape[1]

    if is_transparent:
        plt.figure(figsize=(8, 8), dpi=300)
    else:
        plt.figure(figsize=(8, 8), dpi=300, facecolor='white')

    for expert_id in range(num_experts):
        plt.plot(sorted_param, ratios_matrix[:, expert_id], marker='o', label=f'Expert {expert_id}')

    plt.xlabel(param_type)
    plt.ylabel('Expert Load Ratio')
    plt.title(title_notes if title_notes else 'Expert Load Ratio vs' + param_type)
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    

def compute_expert_entropy(exp_id, base_path='/Users/leahparparov/Downloads/reucl/.exp_result/tmoecl/log'):
    """Compute entropy of expert usage from stats_run_0.npz"""
    path = os.path.join(base_path, exp_id, 'stats_run_0.npz')
    if not os.path.exists(path):
        print(f"[WARNING] Missing stats for {exp_id}")
        return None

    stats = np.load(path)
    if 'expert_usage' not in stats:
        print(f"[WARNING] No 'expert_usage' in {exp_id}")
        return None

    usage_counts = stats['expert_usage'].sum(axis=0)  # sum over rounds
    if usage_counts.sum() == 0:
        return 0.0

    usage_probs = usage_counts / usage_counts.sum()
    return entropy(usage_probs)

def plot_entropy_vs_parameter(exp_header_dict, param_dict, param_type, title_notes=""):
    """
    Plot entropy of expert usage vs alpha values.

    Args:
        exp_header_dict: dict mapping experiment_id -> note
        alpha_dict: dict mapping experiment_id -> alpha_value
    """
    param = []
    entropies = []

    for exp_id in exp_header_dict.keys():
        ent = compute_expert_entropy(exp_id)
        if ent is not None and exp_id in param_dict:
            param.append(param_dict[exp_id])
            entropies.append(ent)

    if not param:
        print("No valid entropy data found.")
        return

    # Sort by alpha
    param, entropies = zip(*sorted(zip(param, entropies)))

    plt.figure(figsize=(6, 6), dpi=300)
    plt.plot(param, entropies, marker='o', linestyle='-', color='tab:purple')
    plt.xlabel(param_type)
    plt.ylabel('Expert Usage Entropy')
    plt.title("Load Balance Entropy vs " + param_type)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_test_loss_vs_parameter(exp_header_dict, param_dict, param_type, is_transparent=False, title_notes=""):
    """
    Plots test loss vs parameter values.
    
    Args:
        exp_header_dict: dict mapping experiment_id -> note
        param_dict: dict mapping experiment_id -> parameter_value
        param_type: string indicating the type of parameter (e.g., 'alpha')
    """
    test_losses = []
    parameters = []

    for exp_id, note in exp_header_dict.items():
        stats_path = f"{PATH_EXPLOG_PREFIX}/tmoecl/log/{exp_id}/stats_run_0.npz"
        stats = np.load(stats_path)
        if 'theory_stats' not in stats:
            print(f"[WARNING] No 'theory_stats' in {exp_id}")
            continue

        theory_stats = stats['theory_stats']
        test_loss = theory_stats[:, 16] 
        if test_loss.size == 0:
            continue

        avg_test_loss = np.mean(test_loss)
        test_losses.append(avg_test_loss)
        parameters.append(param_dict[exp_id])

    if not parameters or not test_losses:
        print("No valid data found for plotting.")
        return

    # Sort by parameter
    sorted_indices = np.argsort(parameters)
    parameters = np.array(parameters)[sorted_indices]
    test_losses = np.array(test_losses)[sorted_indices]

    if is_transparent:
        plt.figure(figsize=(8, 6), dpi=300)
    else:
        plt.figure(figsize=(8, 6), dpi=300, facecolor='white')

    plt.plot(parameters, test_losses, marker='o', linestyle='-', color='tab:blue')
    plt.xlabel(param_type)
    plt.ylabel('Average Test Loss')
    plt.title(title_notes if title_notes else 'Test Loss vs ' + param_type)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
def plot_forgetting_vs_expert_load(exp_header, expert_ids, forget_criteria=20, load_criteria_start=30, title_notes="", is_transparent=False, max_rounds=None):
    """
    Dual-axis plot of cumulative forgetting and expert load across rounds, optionally clipped to max_rounds.
    """
    stats = np.load(f'{PATH_EXPLOG_PREFIX}/tmoecl/log/{exp_header}/stats_run_0.npz')['theory_stats']
    forgetting_curve = stats[:, forget_criteria]
    total_rounds = len(forgetting_curve)
    num_rounds = min(max_rounds, total_rounds) if max_rounds is not None else total_rounds

    x_vals = list(range(1, num_rounds + 1))
    forgetting_curve = forgetting_curve[:num_rounds]

    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300, facecolor='white' if not is_transparent else None)
    ax1.plot(x_vals, forgetting_curve, 'r-', label='Cumulative Forgetting')
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Forgetting (%)', color='r')
    ax1.tick_params(axis='y', labelcolor='r')

    ax2 = ax1.twinx()
    for idx in expert_ids:
        load_curve = stats[:, load_criteria_start + idx][:num_rounds]
        ax2.plot(x_vals, load_curve, label=f'Expert {idx} Load')

    ax2.set_ylabel('Expert Load', color='teal')
    ax2.tick_params(axis='y', labelcolor='teal')

    fig.suptitle(title_notes or 'Cumulative Forgetting vs Expert Load')
    fig.legend(loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
    fig.tight_layout()
    plt.grid(True)
    plt.show()


def plot_forgetting_vs_all_expert_loads_with_accuracy(
    exp_header='20250720_exp4',
    forget_criteria=20,
    load_criteria_start=15,
    acc_criteria=10,
    title_notes="",
    is_transparent=False,
    max_rounds=None,
    load_scale=5  # << New: Scale load curve visibility
):
    stats_path = f"{PATH_EXPLOG_PREFIX}/tmoecl/log/{exp_header}/stats_run_0.npz"
    stats = np.load(stats_path)['theory_stats']

    num_columns = stats.shape[1]
    num_experts = num_columns - load_criteria_start
    if num_experts <= 0:
        raise ValueError(f"No expert load columns found starting at index {load_criteria_start}.")

    forgetting_curve = stats[:, forget_criteria]
    accuracy_curve = stats[:, acc_criteria]
    total_rounds = len(forgetting_curve)
    num_rounds = min(max_rounds or total_rounds, total_rounds)

    x_vals = list(range(1, num_rounds + 1))
    forgetting_curve = forgetting_curve[:num_rounds]
    accuracy_curve = accuracy_curve[:num_rounds]

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    fig, ax1 = plt.subplots(figsize=(14, 7), dpi=300, facecolor='white' if not is_transparent else None)

    ax1.plot(x_vals, forgetting_curve, 'r-', linewidth=2, label='Cumulative Forgetting')
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Forgetting (%)', color='red')
    ax1.tick_params(axis='y', labelcolor='red')

    ax2 = ax1.twinx()
    ax2.set_ylabel(f'Expert Load ×{load_scale}', color='teal')
    ax2.tick_params(axis='y', labelcolor='teal')
    color_map = cm.get_cmap('tab20', num_experts)

    for i in range(num_experts):
        col_idx = load_criteria_start + i
        load_curve = stats[:, col_idx][:num_rounds] * load_scale
        ax2.plot(x_vals, load_curve, linestyle='--', linewidth=1.2,
                 color=color_map(i), alpha=0.9, label=f'Expert {i} Load')

    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    ax3.plot(x_vals, accuracy_curve, 'b-', linewidth=2, label='Test Accuracy')
    ax3.set_ylabel('Accuracy (%)', color='blue')
    ax3.tick_params(axis='y', labelcolor='blue')

    fig.suptitle(title_notes or f"Forgetting vs Expert Load vs Accuracy\n({exp_header})")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()

    fig.legend(
        lines1 + lines2 + lines3,
        labels1 + labels2 + labels3,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.12),
        ncol=4,
        fontsize=9
    )

    fig.tight_layout()
    plt.grid(True)
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

def plot_forgetting_vs_load_stat(
    exp_header,
    num_experts=12,
    forget_criteria=20,
    load_criteria_start=30,
    stat_type='entropy',  # 'entropy' or 'std'
    title_notes="",
    is_transparent=False,
    max_rounds=None
):
    """
    Plot cumulative forgetting vs. diversity of expert loads (across all experts).
    Supports entropy or standard deviation.
    """
    # Load theory stats
    stats = np.load(f'{PATH_EXPLOG_PREFIX}/tmoecl/log/{exp_header}/stats_run_0.npz')['theory_stats']
    forgetting_curve = stats[:, forget_criteria]
    total_rounds = len(forgetting_curve)
    num_rounds = min(max_rounds, total_rounds) if max_rounds is not None else total_rounds

    x_vals = list(range(1, num_rounds + 1))
    forgetting_curve = forgetting_curve[:num_rounds]

    # Slice the load matrix for all experts
    expert_loads = stats[:num_rounds, load_criteria_start:load_criteria_start + num_experts]

    # Compute diversity metric across the 12 experts
    if stat_type == 'entropy':
        # Normalize to probabilities row-wise
        probs = expert_loads / (expert_loads.sum(axis=1, keepdims=True) + 1e-8)
        load_stat = np.array([entropy(p, base=2) for p in probs])
        stat_label = 'Load Entropy (bits)'
        stat_color = 'darkgreen'
    elif stat_type == 'std':
        load_stat = np.std(expert_loads, axis=1)
        stat_label = 'Load Std. Dev.'
        stat_color = 'purple'
    else:
        raise ValueError("stat_type must be 'entropy' or 'std'")

    # Plot
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300, facecolor='white' if not is_transparent else None)
    ax1.plot(x_vals, forgetting_curve, 'r-', label='Cumulative Forgetting')
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Forgetting (%)', color='red')
    ax1.tick_params(axis='y', labelcolor='red')

    ax2 = ax1.twinx()
    ax2.plot(x_vals, load_stat, color=stat_color, linewidth=2, label=stat_label)
    ax2.set_ylabel(stat_label, color=stat_color)
    ax2.tick_params(axis='y', labelcolor=stat_color)

    fig.suptitle(title_notes or f'Forgetting vs {stat_label}')
    fig.legend(loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
    fig.tight_layout()
    plt.grid(True)
    plt.show()


