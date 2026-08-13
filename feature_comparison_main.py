import json
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from matplotlib import pyplot as plt
from matplotlib import ticker as ticker
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

from feature_data_extractor import clean_feature_data, feat_to_idx 
from behavior_data_extractor import clean_behavior_data, behavior_to_idx 

control_feat, mutant_feat = clean_feature_data()
control_beh, mutant_beh = clean_behavior_data()
feature_list = np.load('features.npy')

group_dict = {
    'Control': (control_feat, control_beh),
    'Mutant': (mutant_feat, mutant_beh)
}

dpi=100

# plt.rcParams['axes.labelsize'] = 26 # Axis titles (X and Y)
# plt.rcParams['xtick.labelsize'] = 26  # X-tick dimensions
# plt.rcParams['ytick.labelsize'] = 26   # Y-tick dimensions
# plt.rcParams['axes.titlesize'] = 26

# ==========================================
# 1. HEATMAP RASTER PLOTTING UTILITY
# ==========================================
def plot_bouts(target_behavior, group_label='Control', 
               back_window=60, forward_window=90, align='start', 
               min_duration=None, max_duration=None, context_filter=None, save_fig=None):
    """
    Generates a compact behavioral raster plot showing bout occurrences over time.

    Inputs:
    - target_behavior (str): The behavior to plot (e.g., 'attack').
    - align (str): 'start' or 'end' alignment for the 0-anchor on the plot.
    - save_fig (str): Optional. Set to 'png' or 'svg' to save the generated plot.
    
    Returns:
    - Displays (and optionally saves) a matplotlib figure.
    """
    if align not in ['start', 'end']:
        raise ValueError("The 'align' parameter must be set to either 'start' or 'end'.")

    behaviors = ['None', 'Attack', 'Close Investigation', 'Mount', 'Other']
    color_hexes = ['#ffffff', '#e74c3c', '#3498db', "#ff9900", '#94a3b8'] 
    cmap = ListedColormap(color_hexes)
    
    try:
        target_idx = behavior_to_idx(target_behavior)
    except NameError:
        raise NameError("Please ensure the 'behavior_to_idx' function is active in your environment.")
        
    core_mapping = ['attack', 'closeinvestigation', 'mount']
    idx_to_plot_val = {}
    for i, b_name in enumerate(core_mapping, start=1):
        idx_to_plot_val[behavior_to_idx(b_name)] = i
        
    hist_idx = None
    require_presence = False
    if context_filter is not None:
        hist_name = context_filter[:-1]
        hist_sign = context_filter[-1]
        hist_idx = behavior_to_idx(hist_name)
        require_presence = (hist_sign == '+')

    if group_label not in group_dict:
        raise KeyError(f"Group '{group_label}' not found in your group_dict setup.")
    _, behavior_list = group_dict[group_label]

    kept_bout_windows = []
    y_labels = []
    divider_positions = [] 
    
    total_bouts_detected = 0
    cut_bouts_count = 0
    boundary_skip_count = 0
    
    for mouse_idx in range(len(behavior_list)):
        behavior_data = behavior_list[mouse_idx]
        total_frames = len(behavior_data)
        
        condition = (behavior_data == target_idx)
        padded = np.zeros(len(condition) + 2, dtype=bool)
        padded[1:-1] = condition
        diffs = np.diff(padded.astype(int))
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]
        
        mouse_windows = []
        
        for b_idx, start in enumerate(starts):
            end = ends[b_idx]
            duration = end - start
            total_bouts_detected += 1
            
            if align == 'start':
                slice_start = start - back_window
                slice_end = start + forward_window
            else:
                slice_start = end - back_window
                slice_end = end + forward_window
            
            if slice_start < 0 or slice_end > total_frames:
                boundary_skip_count += 1
                continue
                
            is_cut = False
            if min_duration is not None and duration < min_duration:
                is_cut = True
            if max_duration is not None and duration > max_duration:
                is_cut = True
                
            if context_filter is not None and not is_cut:
                if align == 'start':
                    check_window = behavior_data[slice_start:start]
                else:
                    check_window = behavior_data[end:slice_end]
                    
                has_hist_beh = np.any(check_window == hist_idx)
                if require_presence and not has_hist_beh:
                    is_cut = True
                elif not require_presence and has_hist_beh:
                    is_cut = True
                    
            if is_cut:
                cut_bouts_count += 1
                continue 
                
            raw_window = behavior_data[slice_start:slice_end]
            mapped_window = np.zeros_like(raw_window)
            for frame_idx, raw_val in enumerate(raw_window):
                if raw_val == 0:
                    mapped_window[frame_idx] = 0 
                else:
                    mapped_window[frame_idx] = idx_to_plot_val.get(raw_val, 6) 
                
            mouse_windows.append(mapped_window)
            
        if mouse_windows:
            if kept_bout_windows:
                divider_positions.append(len(kept_bout_windows) - 0.5)
                
            kept_bout_windows.extend(mouse_windows)
            
            n_bouts = len(mouse_windows)
            for i in range(n_bouts):
                bout_num = i + 1
                is_top = (i == n_bouts - 1)
                is_fifth = (bout_num % 5 == 0)
                
                if is_top:
                    y_labels.append(f"Mouse {mouse_idx} ({n_bouts} total)")
                elif is_fifth:
                    y_labels.append(f"{bout_num}")
                else:
                    y_labels.append("")
            
    if not kept_bout_windows:
        print(f"[{group_label}] All Mice: Skipped: Completely empty after filtering criteria applied.")
        return
        
    matrix = np.array(kept_bout_windows)
    fig, ax = plt.subplots(figsize=(14, max(2.5, len(matrix) * 0.14)), dpi=dpi)
    
    ax.imshow(matrix, aspect='auto', cmap=cmap, vmin=0, vmax=4,
              extent=[-back_window, forward_window, 0, len(matrix) - 0.5],
              origin='lower', zorder=1)
              
    for i in range(len(matrix) - 1):
        ax.axhline(i + 0.5, color='#cbd5e1', linestyle='-', linewidth=0.4, alpha=0.6, zorder=2)
              
    ax.grid(True, axis='x', color='#94a3b8', linestyle=':', linewidth=0.6, alpha=0.7, zorder=5)
              
    for pos in divider_positions:
        ax.axhline(pos, color='#7f8c8d', linestyle='--', linewidth=1.0, zorder=3)
              
    ax.axvline(0, color='#2d3436', linestyle='-', linewidth=2.5, zorder=4)
    
    secax = ax.secondary_xaxis('top')
    secax.set_xticks(np.arange(-back_window, forward_window + 1, step=max(10, (back_window+forward_window)//8)))
    secax.tick_params(axis='x', labelsize=8.5)
    
    active_filters = []
    if context_filter is not None: active_filters.append(f"{context_filter}")
    if min_duration is not None: active_filters.append(f"min {min_duration/30} seconds")
    if max_duration is not None: active_filters.append(f"max {max_duration/30} seconds")
    filter_string = ", ".join(active_filters) if active_filters else "None"
    
    total_evaluable = total_bouts_detected - boundary_skip_count
    percent_cut = (cut_bouts_count / total_evaluable * 100) if total_evaluable > 0 else 0
    stats_string = f"Total Bouts: {total_bouts_detected}  |  Cut: {cut_bouts_count} ({percent_cut:.1f}% dropped)"
                    
    ax.set_title(f" {target_behavior.upper()} Bout Timing ({align.upper()}-Aligned) \n{group_label} | Filters: {filter_string}\n {stats_string}", 
                 fontweight='bold', pad=26)
            
    ax.set_xlabel(f"Seconds Relative to Bout {align} (0)", fontweight='semibold')
    ax.set_xticks(np.arange(-back_window, forward_window + 1, step=max(10, (back_window+forward_window)//8)))
    ax.set_yticks(np.arange(len(matrix)))
    ax.set_yticklabels(y_labels, fontweight='semibold')
    ax.tick_params(axis='both', which='major', labelsize=8)
    
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#cbd5e1')
    
    legend_patches = [mpatches.Patch(color=color_hexes[i], label=behaviors[i]) for i in range(1, len(behaviors))]
    ax.legend(handles=legend_patches, bbox_to_anchor=(1.02, 1), loc='upper left', 
              frameon=True, facecolor='white', edgecolor='none')
    
    plt.tight_layout()
    
    if save_fig in ['png', 'svg']:
        safe_beh = target_behavior.replace(" ", "_")
        plt.savefig(f"raster_{safe_beh}_{group_label}_{align}.{save_fig}", format=save_fig, bbox_inches='tight', dpi=dpi)
        
    plt.show()


# ==========================================
# 2. CORE TIME-SERIES SIGNAL EXTRACTOR
# ==========================================
def extract_bout_triggered_features(feat_data, beh_data, target_idx, mouse_idx, group_label="Control", 
                                    back_window=60, forward_window=90, min_duration=None, max_duration=None, 
                                    context_filter=None, align='start', hide_warnings=False):
    """
    Extracts strictly fixed-length feature arrays centered around a behavioral event.

    Inputs:
    - feat_data (np.array): Feature data for a single mouse.
    - beh_data (np.array): Behavior classifications for a single mouse.
    - back_window, forward_window (int): Defines the exact bounds of the uniform output array.

    Returns:
    - np.array: A 2D array of shape (n_bouts, back_window + forward_window).
    
    Tricky Details:
    - The returned array is always a perfect matrix. Bout events will always sit exactly at index `back_window`.
    """
    if align not in ['start', 'end']:
        raise ValueError("The 'align' parameter must be set to either 'start' or 'end'.")

    total_frames = len(feat_data)
    condition = (beh_data == target_idx)
    padded = np.zeros(len(condition) + 2, dtype=bool)
    padded[1:-1] = condition
    
    diffs = np.diff(padded.astype(int))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]

    total_bouts = len(starts)
    if total_bouts == 0:
        if not hide_warnings:
            print(f"[{group_label}] Mouse {mouse_idx}: Skipped: No behavior instances found.")
        return np.array([])

    context_idx = None
    require_presence = True
    if context_filter is not None:
        if context_filter.endswith('+'):
            require_presence = True
            context_behavior_name = context_filter[:-1]
        elif context_filter.endswith('-'):
            require_presence = False
            context_behavior_name = context_filter[:-1]
        else:
            raise ValueError("context_filter must end with '+' or '-'")
        
        context_idx = behavior_to_idx(context_behavior_name)

    valid_bouts = []
    duration_skip_count = 0
    boundary_skip_count = 0
    context_skip_count = 0

    for i, start in enumerate(starts):
        bout_end = ends[i]
        actual_duration = bout_end - start
        
        if align == 'start':
            slice_start = start - back_window
            slice_end = start + forward_window
        else:
            slice_start = bout_end - back_window
            slice_end = bout_end + forward_window
            
        if slice_start < 0 or slice_end > total_frames:
            boundary_skip_count += 1
            continue

        if context_idx is not None:
            if align == 'start':
                check_window = beh_data[slice_start:start]
            else:
                check_window = beh_data[bout_end:slice_end]
                
            has_behavior = np.any(check_window == context_idx)
            if require_presence and not has_behavior:
                context_skip_count += 1
                continue
            elif not require_presence and has_behavior:
                context_skip_count += 1
                continue

        if min_duration is not None and actual_duration < min_duration:
            duration_skip_count += 1
            continue
        if max_duration is not None and actual_duration > max_duration:
            duration_skip_count += 1
            continue
            
        valid_bouts.append(feat_data[slice_start:slice_end])
        
    if len(valid_bouts) == 0 and not hide_warnings:
        reasons = []
        if boundary_skip_count > 0: reasons.append(f"{boundary_skip_count} boundary clips")
        if context_skip_count > 0:  reasons.append(f"{context_skip_count} failed context filter")
        if duration_skip_count > 0: reasons.append(f"{duration_skip_count} failed duration cut")
        print(f"[{group_label}] Mouse {mouse_idx}: Skipped: All {total_bouts} bouts dropped ({', '.join(reasons)}).")
        
    return np.array(valid_bouts)


# ==========================================
# 2b. VARIABLE LENGTH SIGNAL EXTRACTOR (NEW)
# ==========================================
def extract_variable_bout_lengths(feat_data, beh_data, target_idx, mouse_idx, group_label="Control", 
                                  back_window=0, forward_window=0, min_duration=None, max_duration=None, 
                                  context_filter=None, align='start', hide_warnings=False):
    """
    Extracts raw feature arrays corresponding to naturally varying bout lengths.

    Inputs:
    - back_window (int): Amount of padding to include prior to the START of the bout.
    - forward_window (int or None): Amount of padding to include after the END of the bout. 
                                    If None, defaults to 0 (no extra padding).
    - align (str): Dictates whether the `context_filter` looks in the back_window ('start') 
                   or the forward_window ('end').

    Returns:
    - list: A Python list containing arrays of varying lengths.
    """
    if align not in ['start', 'end']:
        raise ValueError("The 'align' parameter must be set to either 'start' or 'end'.")

    # Guard against None values crashing the integer math
    fw = 0 if forward_window is None else forward_window
    bw = 0 if back_window is None else back_window

    total_frames = len(feat_data)
    condition = (beh_data == target_idx)
    padded = np.zeros(len(condition) + 2, dtype=bool)
    padded[1:-1] = condition
    
    diffs = np.diff(padded.astype(int))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]

    if len(starts) == 0:
        if not hide_warnings:
            print(f"[{group_label}] Mouse {mouse_idx}: Skipped: No behavior instances found.")
        return []

    context_idx = None
    require_presence = True
    if context_filter is not None:
        require_presence = context_filter.endswith('+')
        context_idx = behavior_to_idx(context_filter[:-1])

    valid_bouts = []
    
    for i, start in enumerate(starts):
        bout_end = ends[i]
        actual_duration = bout_end - start
        
        # Apply padding symmetrically to grab the whole bout + context
        slice_start = max(0, start - bw)
        slice_end = min(total_frames, bout_end + fw)
            
        if slice_start < 0 or slice_end > total_frames:
            continue

        # Use alignment strictly to determine WHERE to look for the context behavior
        if context_idx is not None:
            if align == 'start':
                check_window = beh_data[slice_start:start]
            else:
                check_window = beh_data[bout_end:slice_end]
                
            has_behavior = np.any(check_window == context_idx)
            if require_presence != has_behavior:
                continue

        # Check durations against the actual bout length (ignoring padding)
        if (min_duration and actual_duration < min_duration) or (max_duration and actual_duration > max_duration):
            continue
            
        valid_bouts.append(feat_data[slice_start:slice_end])
        
    return valid_bouts

# ==========================================
# 3. SINGLE SUBJECT TRACE VISUALIZATION
# ==========================================
def plot_single_mouse_bouts(mouse_idx, feature_name, behavior_name, group_label='Control',
                             back_window=60, forward_window=90, min_duration=None, max_duration=None, 
                             context_filter=None, align='start', save_fig=None):
    """
    Plots individual event-triggered traces for a single mouse to visualize variance.
    """
    feature_data, behavior_data = group_dict[group_label]
    feat_idx = feat_to_idx(feature_name)
    beh_idx = behavior_to_idx(behavior_name)
    x_range = np.arange(-back_window, forward_window)
    
    feat_dat = feature_data[mouse_idx][feat_idx]
    beh_dat = behavior_data[mouse_idx]
    
    bouts = extract_bout_triggered_features(
        feat_dat, beh_dat, beh_idx, group_label=group_label, mouse_idx=mouse_idx, 
        back_window=back_window, forward_window=forward_window, 
        min_duration=min_duration, max_duration=max_duration, 
        context_filter=context_filter, align=align
    )    
    
    if len(bouts) == 0:
        print(f"[{group_label}] Mouse {mouse_idx}: Skipped: No valid inside-bounds bouts.")
        return

    plt.figure(figsize=(9, 5), dpi=dpi)
    for i, bout in enumerate(bouts):
        plt.plot(x_range/30, bout, alpha=0.25, lw=0.8, label=f'Bout {i+1}' if len(bouts) <= 10 else "")
        
    mouse_average = np.mean(bouts, axis=0)
    plt.plot(x_range/30, mouse_average, lw=3, color='black', label='Mouse Average')
    
    plt.axvline(x=0, linestyle='--', color='#7f8c8d', lw=1.5)
    plt.title(f"Individual Bouts: {group_label} Mouse {mouse_idx}\nBehavior: {behavior_name.upper()} ({len(bouts)} bouts) | Feature: {feature_name}\nFilters: min={min_duration}, max={max_duration}, Context={context_filter}",
              fontsize=11, fontweight='bold', pad=12)
    plt.xlabel(f'seconds relative to Bout {align} (0)', fontsize=11)
    plt.ylabel(feature_name, fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(frameon=True, facecolor='white', edgecolor='none')
    plt.tight_layout()
    
    if save_fig in ['png', 'svg']:
        safe_beh = behavior_name.replace(" ", "_")
        safe_feat = feature_name.replace(" ", "_")
        plt.savefig(f"single_mouse_{mouse_idx}_{safe_beh}_{safe_feat}.{save_fig}", format=save_fig, bbox_inches='tight', dpi=dpi)
        
    plt.show()


# ==========================================
# 4. SINGLE COHORT AGGREGATION PLOTTER
# ==========================================
def plot_bout_triggered_averages(feature_name, behavior_name, group_label="Control", 
                                 back_window=60, forward_window=90, min_duration=None, max_duration=None, 
                                 color=None, context_filter=None, align='start', save_fig=None):
    """
    Plots the grand average of a single experimental cohort across all their valid bouts.
    """
    feature_data, behavior_list = group_dict[group_label]
    feat_idx = feat_to_idx(feature_name)
    beh_idx = behavior_to_idx(behavior_name)
    x_range = np.arange(-back_window, forward_window)
    
    group_colors = {'Control': '#7f8c8d', 'Mutant': '#e74c3c'}
    main_color = group_colors.get(group_label, color if color is not None else '#7f8c8d')
    
    cohort_averages = []
    plt.figure(figsize=(9, 5), dpi=dpi)
    
    for mouse_idx in range(len(behavior_list)):
        feat_dat = feature_data[mouse_idx][feat_idx]
        beh_dat = behavior_list[mouse_idx]
        
        bouts = extract_bout_triggered_features(
            feat_dat, beh_dat, beh_idx, group_label=group_label, mouse_idx=mouse_idx, 
            back_window=back_window, forward_window=forward_window, 
            min_duration=min_duration, max_duration=max_duration, 
            context_filter=context_filter, align=align
        )
        
        if len(bouts) == 0:
            continue
            
        mouse_avg = np.mean(bouts, axis=0)
        cohort_averages.append(mouse_avg)
        plt.plot(x_range/30, mouse_avg, alpha=0.20, lw=1, color=main_color, label=f'Mouse {mouse_idx}' if len(behavior_list) <= 6 else "")

    if cohort_averages:
        grand_average = np.mean(cohort_averages, axis=0)
        plt.plot(x_range/30, grand_average, lw=3, color=main_color, alpha=1.0, label='Grand Average')
        plt.axvline(x=0, linestyle='--', color='#2c3e50', lw=1.2)
        
        plt.title(f"Bout-Triggered Feature Average: {group_label} Cohort\nBehavior: {behavior_name.upper()} | Feature: {feature_name}\nFilters: min={min_duration}, max={max_duration}, Context={context_filter}", 
                  fontsize=11, fontweight='bold', pad=12)
        plt.xlabel(f'Seconds relative to bout {align} (0)', fontsize=11)
        plt.ylabel(f'Mean {feature_name}', fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.legend(frameon=True, facecolor='white', edgecolor='none')
        plt.tight_layout()
        
        if save_fig in ['png', 'svg']:
            safe_beh = behavior_name.replace(" ", "_")
            safe_feat = feature_name.replace(" ", "_")
            plt.savefig(f"cohort_avg_{group_label}_{safe_beh}_{safe_feat}.{save_fig}", format=save_fig, bbox_inches='tight', dpi=dpi)
            
        plt.show()
    else:
        plt.close()
        print(f"[{group_label}] All Mice: Skipped: Aggregation failed due to lack of valid data.")


# ==========================================
# 5. MULTI-COHORT OVERLAY COMPARISON ENGINE
# ==========================================
def group_comparison(feature_name, behavior_name, back_window=60, forward_window=90, 
                     min_duration=None, max_duration=None, error_style='envelope', 
                     context_filter=None, align='start', plot=True, stats=False, hide_warnings=False, save_fig=None):
    """
    Overlays Control vs. Mutant averages onto a single plot and returns data for AUC processing.
    Inputs:
    -save_fig: saves a plot (can input 'svg', 'png' or None)
    """
    if align not in ['start', 'end']:
        raise ValueError("The 'align' parameter must be set to either 'start' or 'end'.")

    feat_idx = feat_to_idx(feature_name)
    beh_idx = behavior_to_idx(behavior_name)
    x_range = np.arange(-back_window, forward_window)   
    colors = {'Control': '#7f8c8d', 'Mutant': '#e74c3c'} 
    
    if plot: plt.figure(figsize=(9, 7), dpi=dpi)

    group_event_triggered_lines = []
    for group_name, (feature_data, behavior_data) in group_dict.items():    
        cohort_averages = []
        
        for mouse_idx in range(len(behavior_data)): 
            feat_dat = feature_data[mouse_idx][feat_idx]
            beh_dat = behavior_data[mouse_idx]
            
            bouts = extract_bout_triggered_features(
                feat_dat, beh_dat, beh_idx, group_label=group_name, mouse_idx=mouse_idx,
                back_window=back_window, forward_window=forward_window, 
                min_duration=min_duration, max_duration=max_duration,
                context_filter=context_filter, align=align, hide_warnings=hide_warnings 
            )
            if len(bouts) > 0:
                cohort_averages.append(np.mean(bouts, axis=0))  
                
        if not cohort_averages:
            if not hide_warnings:
                print(f"[{group_name}] All Mice: Skipped: No valid data found. Group excluded from overlay.")
            group_event_triggered_lines.append(np.array([]))
            continue
        
        cohort_averages = np.array(cohort_averages)
        grand_average = np.mean(cohort_averages, axis=0)

        if plot:
            color = colors.get(group_name, plt.cm.tab10(len(plt.gca().lines)))
            if error_style == 'envelope':
                grand_std = np.std(cohort_averages, axis=0)
                plt.fill_between(x_range/30, grand_average - grand_std, grand_average + grand_std, color=color, alpha=0.15, zorder=1)
            elif error_style == 'lines':
                for mouse_avg in cohort_averages:
                    plt.plot(x_range/30, mouse_avg, color=color, alpha=0.2, lw=0.8, zorder=1)

            plt.plot(x_range/30, grand_average, label=f'{group_name} (n={len(cohort_averages)})', color=color, lw=4, zorder=2)

        group_event_triggered_lines.append(cohort_averages)  

    if plot: 
        plt.axvline(x=0, linestyle='--', color='#2c3e50', lw=1.2, zorder=0)
        plt.title(f"Behavior: {behavior_name.upper()} ({align.upper()}-Aligned) | Feature: {feature_name}\nFilters: min={min_duration}, max={max_duration}, Context={context_filter}", 
                  fontsize=11, fontweight='bold', pad=12)

        plt.legend(frameon=True, facecolor='white', edgecolor='none', fontsize='x-large')
        plt.legend(
            loc='upper left',           # Pin the top-left of the legend...
            bbox_to_anchor=(0.0, -0.15),# ...to 18% below the bottom-left corner of the plot
            ncols=5,                    # Flattens entries into a single horizontal row (adjust to match your total items)
            frameon=False,              # Removes the box border completely                # Legend text size
            columnspacing=1.5,          # Space between horizontal legend items
            handletextpad=0.5           # Space between color symbol and text label
        )

        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.tight_layout()
        
        if save_fig in ['png', 'svg']:
            safe_beh = behavior_name.replace(" ", "_")
            safe_feat = feature_name.replace(" ", "_")
            plt.savefig(f"group_compare_{safe_beh}_{safe_feat}.{save_fig}", format=save_fig, bbox_inches='tight', dpi=dpi)
            
        plt.show()

    if stats: return group_event_triggered_lines


# ==========================================
# 6. AREA UNDER THE CURVE (AUC) STATISTICS
# ==========================================
def compare_bouts_auc(mouse_lines, start_stats, stop_stats, back_window=60, forward_window=90,
                      group_names=['Control', 'Mutant'], align='start', plot=True, 
                      get_stats=True, feat_name="Feature", hide_warnings=False, save_fig=None):  
    """
    Calculates Area Under the Curve and computes statistical significance.

    Inputs:
    - start_stats, stop_stats (int): Relative index bounds where 0 is the event itself.
      (e.g., -30 to 0 analyzes the 30 frames prior to the event).
    - save_fig: Can be "None" or "png" or "svg"

    Returns:
    - p_value (float): The Mann-Whitney U test p-value.
    - group_aucs (list): The raw AUC calculations for both groups.
    
    Tricky Details:
    - Because the feature extractor places the event perfectly at `back_window`, this math relies 
      on `anchor_idx = back_window` to ensure calculations are identical across alignment types.
    """
    if not mouse_lines or len(mouse_lines) < 2:
        if not hide_warnings:
            print(f"[{feat_name}] Skipped: Input matrix 'mouse_lines' is missing necessary sample cohorts.")
        return (None, []) if get_stats else None

    # The Anchor Principle: Time 0 is always at index `back_window`
    anchor_idx = back_window
    slice_start = anchor_idx + start_stats
    slice_end = anchor_idx + stop_stats
    
    # Boundary checks to prevent array index out-of-bounds errors
    slice_start = max(0, slice_start)
    slice_end = min(back_window + forward_window, slice_end)

    group_aucs = []
    for group in mouse_lines:
        if len(group) == 0:
            group_aucs.append(np.array([]))
            continue
        window_data = group[:, slice_start:slice_end]
        auc_per_mouse = np.trapezoid(window_data, axis=1)
        group_aucs.append(auc_per_mouse)
        
    control_aucs = group_aucs[0]
    mutant_aucs = group_aucs[1]
    n_control, n_mutant = len(control_aucs), len(mutant_aucs)
    
    # print(f"\n--- AUC Analysis [{feat_name}] ---")
    # print(f"Sample Windows: {group_names[0]} (n={n_control}) | {group_names[1]} (n={n_mutant})")

    if plot:
        x_positions = [1, 2]
        means = [np.mean(control_aucs) if n_control > 0 else 0, np.mean(mutant_aucs) if n_mutant > 0 else 0]
        stds = [np.std(control_aucs) if n_control > 0 else 0, np.std(mutant_aucs) if n_mutant > 0 else 0]

        fig, ax = plt.subplots(figsize=(4.5, 5.5), dpi=dpi) 
        colors = ['#7f8c8d', '#e74c3c']
        ax.bar(x_positions, means, align='center', width=0.4, color=colors, alpha=0.4, edgecolor=colors, linewidth=2, zorder=1)

        if n_control > 0:
            jitter1 = np.random.normal(1, 0.04, size=n_control)
            ax.scatter(jitter1, control_aucs, color=colors[0], edgecolor='white', linewidth=0.8, s=45, zorder=2)
        if n_mutant > 0:
            jitter2 = np.random.normal(2, 0.04, size=n_mutant)
            ax.scatter(jitter2, mutant_aucs, color=colors[1], edgecolor='white', linewidth=0.8, s=45, zorder=2)

        ax.errorbar(x_positions, means, yerr=stds, fmt='none', ecolor='#2c3e50', elinewidth=1.2, capsize=12, capthick=1.2, zorder=3)
        ax.set_ylabel(f'AUC ({start_stats/30} to {stop_stats/30} seconds relative to {align})', fontsize=11, fontweight='semibold', labelpad=10)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([f'{group_names[0]}\n(n={n_control})', f'{group_names[1]}\n(n={n_mutant})'], fontsize=11, fontweight='semibold')
        ax.set_title(f'{feat_name} AUC Comparison', fontsize=13, fontweight='bold', pad=22)
        
        for spine in ['top', 'right']: ax.spines[spine].set_visible(False)
        for spine in ['left', 'bottom']: ax.spines[spine].set_color('#bdc3c7')
        ax.yaxis.grid(True, linestyle='--', alpha=0.4, color='#bdc3c7', zorder=0)
        plt.tight_layout()
        
        if save_fig in ['png', 'svg']:
            safe_feat = feat_name.replace(" ", "_")
            plt.savefig(f"auc_{safe_feat}.{save_fig}", format=save_fig, bbox_inches='tight', dpi=dpi)
            
        plt.show()

    if get_stats:
        if n_control == 0 or n_mutant == 0:
            return None, group_aucs
        try:
            res = scipy_stats.mannwhitneyu(control_aucs, mutant_aucs, alternative='two-sided', method='exact')
        except ValueError:
            res = scipy_stats.mannwhitneyu(control_aucs, mutant_aucs, alternative='two-sided')

        # print(f"Mann-Whitney U p-value: {res.pvalue:.5f}\n" + "-"*35)

        return res.pvalue, group_aucs


# ==========================================
# 7. AUTOMATED MASS FEATURE RANKING LEADERBOARD
# ==========================================
def rank_features_by_pvalue(feature_list=feature_list, behavior_name='attack',
                            back_window=60, forward_window=90, start_stats=10, stop_stats=40,
                            min_duration=None, max_duration=None, context_filter=None, align='start', 
                            error_style='lines', group_names=['Control', 'Mutant'], hide_warnings=True, save_rankings=False):
    """
    Iterates through all features to identify the most statistically significant differences.

    Inputs:
    - feature_list (list): Array of all tracking feature string names.
    - hide_warnings (bool): When True, silences data-drop logs to keep the console clean during mass execution.
    - save_rankings(bool): Creates a json outfile in this directory with name feature_rankings.json

    Returns:
    - dict: A dictionary of {feature_name: p_value} sorted ascending from most to least significant.
    """
    feature_to_p = {}

    for feature in feature_list:
        try:
            mouse_lines = group_comparison(feature_name=feature, behavior_name=behavior_name,
                error_style=error_style, back_window=back_window, forward_window=forward_window,
                min_duration=min_duration, max_duration=max_duration, context_filter=context_filter, 
                align=align, plot=False, stats=True, hide_warnings=True
            )

            if not mouse_lines or len(mouse_lines) < 2 or len(mouse_lines[0]) == 0 or len(mouse_lines[1]) == 0:
                continue

            p_value, _ = compare_bouts_auc(
                mouse_lines=mouse_lines, start_stats=start_stats, stop_stats=stop_stats,
                back_window=back_window, forward_window=forward_window, group_names=group_names,
                align=align, plot=False, get_stats=True, feat_name=feature, hide_warnings=hide_warnings
            )
            
            if p_value is not None:
                feature_to_p[str(feature)] = float(p_value)
                
        except Exception as e:
            if not hide_warnings:
                print(f"[{feature}] Skipped: Exception during processing: {e}")
            continue

    sorted_rankings = dict(sorted(feature_to_p.items(), key=lambda x: x[1]))

    # print(f"\n--- Feature Significance Rankings (Aligned: {align.upper()}) ---")
    # print(json.dumps(sorted_rankings, indent=4))

    if save_rankings:
        out_filename = 'feature_rankings.json'
        with open(out_filename, 'w') as outfile:
            json.dump(sorted_rankings, outfile, indent=4)


    return sorted_rankings
