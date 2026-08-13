import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
import matplotlib.cm as cm
from pathlib import Path
from feature_comparison_main import extract_variable_bout_lengths
from feature_data_extractor import clean_feature_data, feat_to_idx
from behavior_data_extractor import clean_behavior_data, behavior_to_idx
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse
from sklearn.preprocessing import StandardScaler
import os
import json  # Added to load the json feature configuration
from hyppo.ksample import Energy

feature_list = np.load('features.npy')


control_path = Path('/Users/josieallred/SURF/data/RIM/control')
mutant_path = Path('/Users/josieallred/SURF/data/RIM/mutant')
control_feats, mutant_feats = clean_feature_data(control_path, mutant_path)
control_behs, mutant_behs = clean_behavior_data(control_path, mutant_path)

# for weird bug in bout triggered features data
control_feats = [control_feat.T for control_feat in control_feats.copy()]
mutant_feats = [mutant_feat.T for mutant_feat in mutant_feats.copy()]

CONTROL_COLOR = "gray"
MUTANT_COLOR = "#E93339"

dpi=300
point_size=8

plt.rcParams['axes.labelsize'] = 16  # Axis titles (X and Y)
plt.rcParams['xtick.labelsize'] = 16  # X-tick dimensions
plt.rcParams['ytick.labelsize'] = 16   # Y-tick dimensions
plt.rcParams['axes.titlesize'] = 25

# ==========================================
# DATA PROCESSING HELPERS
# ==========================================
def process_extracted_bouts(bouts, start_cut, n_features):
    """
    Processes extracted bouts (either a 3D numpy array or a list of 2D arrays)
    by slicing from start_cut and averaging over the time/frame axis.
    Returns one row per bout: shape (n_bouts, n_features).
    """
    if len(bouts) == 0:
        return np.empty((0, n_features))

    if isinstance(bouts, np.ndarray) and bouts.ndim == 3:
        # Homogeneous fixed-length 3D array case: (n_bouts, n_frames, n_features)
        sliced = bouts[:, start_cut:, :]
        if sliced.shape[1] == 0:
            return np.empty((bouts.shape[0], n_features))
        return np.mean(sliced, axis=1)
    else:
        # List of variable-length 2D arrays case (when end_cut is None)
        bout_means = []
        for bout in bouts:
            if bout.shape[0] > start_cut:
                sliced = bout[start_cut:, :]
                bout_means.append(np.mean(sliced, axis=0))
        if len(bout_means) > 0:
            return np.vstack(bout_means)
        else:
            return np.empty((0, n_features))


def scale_and_project_to_pca(control_arr, mutant_arr):
    """
    Standardizes control and mutant feature arrays together (fit on the pooled
    data), fits a 2-component PCA on the pooled, scaled data, and projects both
    groups into that PCA space.
    """
    scaler = StandardScaler()
    pooled_raw = np.concatenate((control_arr, mutant_arr))
    scaler.fit(pooled_raw)

    control_scaled = scaler.transform(control_arr)
    mutant_scaled = scaler.transform(mutant_arr)

    pca = PCA(n_components=2)
    pooled_scaled = np.concatenate((control_scaled, mutant_scaled))
    pca.fit(pooled_scaled)

    control_points = pca.transform(control_scaled)
    mutant_points = pca.transform(mutant_scaled)


    return control_points, mutant_points, pca


def get_pca_components_dict(pca, feature_names=None, top_n=5):
    """
    Returns a dictionary detailing the explained variance and the top 
    contributing features (loadings) for each principal component.
    """
    pca_info = {}
    for i, ratio in enumerate(pca.explained_variance_ratio_):
        pc_name = f"PC{i+1}"
        pc_dict = {
            "explained_variance_percent": round(ratio * 100, 2),
            "top_features": {}
        }
        
        component = pca.components_[i]
        top_indices = np.argsort(np.abs(component))[::-1][:top_n]
        
        for idx in top_indices:
            weight = component[idx]
            name = feature_list[idx] if feature_names and idx < len(feature_names) else f"Feature_{idx}"
            pc_dict["top_features"][name] = round(weight, 4)
            
        pca_info[pc_name] = pc_dict
        
    return pca_info


# ==========================================
# DRAW ELLIPSE
# ==========================================
def draw_layered_covariance_ellipse(points, ax, n_stds=[1, 2, 3], n=1, **kwargs):
    """Draws nested ellipses representing multiple levels of standard deviation."""
    cov = np.cov(points, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    order = eigenvalues.argsort()[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    mean_pos = np.mean(points, axis=0)

    for layer_idx, n_std in enumerate(sorted(n_stds, reverse=True)):
        width, height = 2 * n_std * np.sqrt(eigenvalues)

        current_kwargs = kwargs.copy()
        if layer_idx > 0 and "label" in current_kwargs:
            current_kwargs.pop("label")

        ellipse = Ellipse(xy=mean_pos, width=width / np.sqrt(n), height=height / np.sqrt(n),
                           angle=angle, **current_kwargs)
        ax.add_patch(ellipse)


# ==========================================
# PLOTTING HELPER
# ==========================================
def render_pca_scatter(control_points, mutant_points, pca, title,
                        point_size, ellipse_stds, save_path, format):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=dpi)

    ax.scatter(control_points[:, 0], control_points[:, 1], alpha=0.8, color=CONTROL_COLOR,
               s=point_size, zorder=3, label="Control")
    ax.scatter(mutant_points[:, 0], mutant_points[:, 1], alpha=0.8, color=MUTANT_COLOR,
               s=point_size, zorder=3, label="Mutant")

    draw_layered_covariance_ellipse(control_points, ax, n_stds=ellipse_stds,
                                     edgecolor=CONTROL_COLOR, facecolor=CONTROL_COLOR, alpha=0.15, lw=2)
    draw_layered_covariance_ellipse(mutant_points, ax, n_stds=ellipse_stds,
                                     edgecolor=MUTANT_COLOR, facecolor=MUTANT_COLOR, alpha=0.15, lw=2)

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)", fontsize=16, labelpad=12)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)", fontsize=16, labelpad=12)
    # ax.set_title(title)
    # plt.legend(
    #     loc="upper center",
    #     bbox_to_anchor=(0.5, -0.15),
    #     ncol=3,            # Arranges items into horizontal columns
    #     frameon=True       # Keeps the outer box visible
    # )
    # ax.set_xlim(-22, 62)
    ax.set_xticks([])
    # ax.set_ylim(-42, 24)
    ax.set_yticks([])
    # plt.title('Mount', pad=20)
    plt.tight_layout(rect=[0, 0.07, 1, 1])

    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    if save_path is not None:
        plt.savefig(save_path, format=format)
    plt.show()


# ==========================================
# MAIN ANALYSIS PIPELINE
# ==========================================
def main_pca(start_cut, forward_window, min_duration, back_window, max_duration, context_filter,
             align, behaviors, hide_warnings, format, save_dir, save_files, isolate_features=None, plot=True, out=True, print_components=False):

    # 1. Handle Feature Isolation if specified
    active_control_feats = control_feats
    active_mutant_feats = mutant_feats
    
    feature_title_suffix = ""
    all_feature_names = []
    
    if isolate_features is not None:
        if isinstance(isolate_features, str):
            isolate_features = [isolate_features]
            
        json_path = Path(__file__).parent / 'feature_groups.json' if '__file__' in locals() else Path('feature_groups.json')
        with open(json_path, 'r') as f:
            feature_groups = json.load(f)
            
        selected_feature_names = []
        all_valid_individual_features = set()
        
        for group_list in feature_groups.values():
            all_valid_individual_features.update(group_list)

        for group in isolate_features:
            if group in feature_groups:
                selected_feature_names.extend(feature_groups[group])
            elif group in all_valid_individual_features:
                selected_feature_names.append(group)
            else:
                raise ValueError(f"'{group}' is not a recognized feature group or individual feature name in feature_groups.json!")
                
        selected_feature_names = list(dict.fromkeys(selected_feature_names))
        selected_indices = []
        
        for feat_name in selected_feature_names:
            try:
                idx = feat_to_idx(feat_name)
                selected_indices.append(idx)
            except (ValueError, KeyError, IndexError):
                if not hide_warnings:
                    print(f"Warning: feature '{feat_name}' could not be resolved to an index.")
                    
        if selected_indices:
            active_control_feats = [cf[:, selected_indices] for cf in control_feats]
            active_mutant_feats = [mf[:, selected_indices] for mf in mutant_feats]
            feature_title_suffix = f" ({', '.join(isolate_features)})"
            all_feature_names = selected_feature_names
        else:
            raise ValueError("None of the specified isolated features or feature groups were resolved.")

    n_features = active_control_feats[0].shape[-1]
    
    # If not isolating, create dummy names for the dictionary
    if not all_feature_names:
        all_feature_names = [f"Feature_{i}" for i in range(n_features)]

    title_str = ", ".join([b.replace("_", " ").title() for b in behaviors])
    filename_str = "_".join(behaviors)
    
    if isolate_features is not None:
        filename_str += "_" + "_".join([g.replace(" ", "_").lower() for g in isolate_features])

    if save_files:
        os.makedirs(save_dir, exist_ok=True)

    # Note: Ensured this matches your provided logic. Make sure behavior_to_idx expects 3 arguments!
    target_behavior_idxs = [behavior_to_idx(behavior, control_path, mutant_path) for behavior in behaviors]

    n_control_mice = len(active_control_feats)
    n_mutant_mice = len(active_mutant_feats)
    n_mice = max(n_control_mice, n_mutant_mice)

    pooled_control_bouts = []
    pooled_mutant_bouts = []
    control_bouts_by_mouse = []
    mutant_bouts_by_mouse = []

    for mouse_idx in range(n_mice):
        this_mouse_control_bouts = []
        this_mouse_mutant_bouts = []

        for behavior_idx in target_behavior_idxs:
            if mouse_idx < n_control_mice:
                control_bouts = extract_variable_bout_lengths(
                    active_control_feats[mouse_idx], control_behs[mouse_idx], behavior_idx, mouse_idx, "Control",
                    back_window, forward_window, min_duration, max_duration, context_filter, align, hide_warnings
                )
                control_bout_means = process_extracted_bouts(control_bouts, start_cut, n_features)
                if len(control_bout_means) > 0:
                    this_mouse_control_bouts.append(control_bout_means)
                    pooled_control_bouts.append(control_bout_means)

            if mouse_idx < n_mutant_mice:
                mutant_bouts = extract_variable_bout_lengths(
                    active_mutant_feats[mouse_idx], mutant_behs[mouse_idx], behavior_idx, mouse_idx, "Mutant",
                    back_window, forward_window, min_duration, max_duration, context_filter, align, hide_warnings
                )
                mutant_bout_means = process_extracted_bouts(mutant_bouts, start_cut, n_features)
                if len(mutant_bout_means) > 0:
                    this_mouse_mutant_bouts.append(mutant_bout_means)
                    pooled_mutant_bouts.append(mutant_bout_means)

        if mouse_idx < n_control_mice:
            if this_mouse_control_bouts:
                control_bouts_by_mouse.append(np.vstack(this_mouse_control_bouts))
            else:
                control_bouts_by_mouse.append(np.empty((0, n_features)))

        if mouse_idx < n_mutant_mice:
            if this_mouse_mutant_bouts:
                mutant_bouts_by_mouse.append(np.vstack(this_mouse_mutant_bouts))
            else:
                mutant_bouts_by_mouse.append(np.empty((0, n_features)))

    # ==========================================
    # BOUT-BY-BOUT PCA
    # ==========================================
    # Added safe stacking to prevent ValueErrors if lists are empty
    pooled_control_bouts = np.vstack(pooled_control_bouts) if pooled_control_bouts else np.empty((0, n_features))
    pooled_mutant_bouts = np.vstack(pooled_mutant_bouts) if pooled_mutant_bouts else np.empty((0, n_features))

    if len(pooled_control_bouts) > 0 and len(pooled_mutant_bouts) > 0:
        control_points, mutant_points, pca_by_bout = scale_and_project_to_pca(
            pooled_control_bouts, pooled_mutant_bouts
        )
        
        if print_components:
            bout_pca_dict = get_pca_components_dict(pca_by_bout, feature_names=all_feature_names, top_n=5)
            print("\n--- BOUT-BY-BOUT PCA COMPONENTS ---")
            print(json.dumps(bout_pca_dict, indent=4))

        bout_path = None
        if save_files:
            if forward_window is not None:
                bout_path = os.path.join(save_dir, f"pca_{filename_str}_by_bout_{round(forward_window / 30, 2):g}s.{format}")
            else:
                bout_path = os.path.join(save_dir, f"pca_{filename_str}_by_bout_no_cut.{format}")
        if plot: 
            render_pca_scatter(
                control_points, mutant_points, pca_by_bout,
                title=f"PCA for {title_str} {feature_title_suffix} (by bout)",
                point_size=point_size, ellipse_stds=[2], save_path=bout_path, format=format)
    else:
        if not hide_warnings:
            print("Warning: Insufficient bouts to perform bout-by-bout PCA.")
            
    # # ==========================================
    # # MOUSE-BY-MOUSE PCA
    # # ==========================================
    # control_mouse_means = np.array([np.mean(mouse, axis=0) for mouse in control_bouts_by_mouse if len(mouse) > 0])
    # mutant_mouse_means = np.array([np.mean(mouse, axis=0) for mouse in mutant_bouts_by_mouse if len(mouse) > 0])

    # # Added safe-guard here to prevent scaler crash on empty arrays
    # if len(control_mouse_means) > 0 and len(mutant_mouse_means) > 0:
    #     control_points_m, mutant_points_m, pca_by_mouse = scale_and_project_to_pca(
    #         control_mouse_means, mutant_mouse_means
    #     )

    #     if print_components:
    #         mouse_pca_dict = get_pca_components_dict(pca_by_mouse, feature_names=all_feature_names, top_n=5)
    #         print("\n--- MOUSE-BY-MOUSE PCA COMPONENTS ---")
    #         print(json.dumps(mouse_pca_dict, indent=4))

    #     mouse_path = None
    #     if save_files:
    #         # Replaced end_cut with forward_window
    #         if forward_window is not None:
    #             mouse_path = os.path.join(save_dir, f"pca_{filename_str}_by_mouse_{round(forward_window / 30, 2):g}s.{format}")
    #         else:
    #             mouse_path = os.path.join(save_dir, f"pca_{filename_str}_by_mouse_no_cut.{format}")

    #     if plot: 
    #         render_pca_scatter(
    #             control_points_m, mutant_points_m, pca_by_mouse,
    #             title=f"PCA for {title_str} Features{feature_title_suffix} (by mouse)",
    #             point_size=point_size, ellipse_stds=[1, 2], save_path=mouse_path, format=format
    #         )
    # else:
    #     if not hide_warnings:
    #         print("Warning: Insufficient data to perform mouse-by-mouse PCA.")

    # return pooled_control_bouts, pooled_mutant_bouts, control_mouse_means, mutant_mouse_means
    return pooled_control_bouts, pooled_mutant_bouts


def calculate_energy(control_data, mutant_data):
    ''' A metric to indicate the difference between two distributions'''
    scaler = StandardScaler()
    scaler.fit(np.concatenate((control_data, mutant_data)))
    cont_scaled = scaler.transform(control_data)
    mut_scaled = scaler.transform(mutant_data)

    pca = PCA(n_components=0.95)
    pca.fit(np.concatenate((cont_scaled, mut_scaled)))

    cont_reduced = pca.transform(cont_scaled)
    mut_reduced = pca.transform(mut_scaled)

    stat, pvalue = Energy().test(cont_reduced, mut_reduced)

    return stat, pvalue




def calculate_mahalanobis(control_data, mutant_data):
    ''' A metric to indicate the difference between the centroids of two distributions,
    with a Hotelling's T^2 -> F-test p-value. '''
    scaler = StandardScaler()
    scaler.fit(np.concatenate((control_data, mutant_data)))
    cont_scaled = scaler.transform(control_data)
    mut_scaled = scaler.transform(mutant_data)

    pca = PCA(n_components=0.95)
    pca.fit(np.concatenate((cont_scaled, mut_scaled)))

    cont_reduced = pca.transform(cont_scaled)
    mut_reduced = pca.transform(mut_scaled)

    # 1. Calculate the center (centroid) of each group across columns (axis=0)
    mean_cont = np.mean(cont_reduced, axis=0)
    mean_mut = np.mean(mut_reduced, axis=0)
    # The vector connecting the two centers
    diff = mean_cont - mean_mut

    # 2. Calculate the covariance of each group's features (rowvar=False)
    cov_cont = np.cov(cont_reduced, rowvar=False)
    cov_mut = np.cov(mut_reduced, rowvar=False)

    # 3. Calculate the Pooled Covariance Matrix
    n_cont = cont_reduced.shape[0]
    n_mut = mut_reduced.shape[0]
    cov_pooled = ((n_cont - 1) * cov_cont + (n_mut - 1) * cov_mut) / (n_cont + n_mut - 2)

    # 4. Calculate the (squared) Mahalanobis distance
    inv_cov = np.linalg.inv(cov_pooled)
    D2 = diff.T @ inv_cov @ diff
    stat = np.sqrt(D2)

    # 5. Convert to Hotelling's T^2, then to an F-statistic and p-value
    p = cont_reduced.shape[1]  # number of PCA dimensions retained
    n1, n2 = n_cont, n_mut

    T2 = (n1 * n2) / (n1 + n2) * D2
    F_stat = ((n1 + n2 - p - 1) / ((n1 + n2 - 2) * p)) * T2
    df1, df2 = p, n1 + n2 - p - 1
    pvalue = 1 - stats.f.cdf(F_stat, df1, df2)

    return stat, pvalue

def diffs_covs(control_data, mutant_data):
    ''' A metric to indicate the difference between the centroids of two distributions,
    with a Hotelling's T^2 -> F-test p-value. '''
    scaler = StandardScaler()
    scaler.fit(np.concatenate((control_data, mutant_data)))
    cont_scaled = scaler.transform(control_data)
    mut_scaled = scaler.transform(mutant_data)

    pca = PCA(n_components=0.95)
    pca.fit(np.concatenate((cont_scaled, mut_scaled)))
    total_variance = np.sum(pca.explained_variance_ratio_)

    cont_reduced = pca.transform(cont_scaled)
    mut_reduced = pca.transform(mut_scaled)

    # # 1. Calculate the center (centroid) of each group across columns (axis=0)
    # mean_cont = np.mean(cont_reduced, axis=0)
    # mean_mut = np.mean(mut_reduced, axis=0)

    # # The vector connecting the two centers
    # diff = mean_cont - mean_mut
    # diff_means_scaler = np.linalg.norm(diff)

    # 2. Calculate the covariance of each group's features (rowvar=False)
    cov_cont = np.cov(cont_reduced, rowvar=False)
    cov_cont_norm = np.linalg.norm(cov_cont, ord='fro')
    cov_mut = np.cov(mut_reduced, rowvar=False)
    cov_mut_norm = np.linalg.norm(cov_mut, ord='fro')

    difference = np.linalg.norm(cov_cont - cov_mut, ord='fro')

    return total_variance, cov_cont_norm, cov_mut_norm, difference




if __name__ == "__main__":
    start_cut = 0
    max_duration = None
    context_filter = None
    hide_warnings = True

    # isolate_features can be:
    # 1. None (uses all features)
    # 2. A string name of a category: "Position Features"
    # 3. A list of category strings: ["Position Features", "Locomotion Features"]

    save_files = False
    plot = True

    align = "start"
    back_window = 0
    forward_window = 15  # Updated from end_cut
    min_duration = None
    format = "png"       # png or svg

    isolate_features = ['Position Features']
    behaviors = ["attack"]
    target_folder = f"/Users/josieallred/SURF/output"

    # Note: added print_components=True here based on your request
    # currently not getting mouse by mouse data
    control_bouts, mutant_bouts = main_pca(
        start_cut, forward_window, min_duration, back_window, max_duration, context_filter,
        align, behaviors, hide_warnings, format, target_folder, save_files, 
        isolate_features=isolate_features, plot=plot, out=True, print_components=False
    )

    total_variance, cov_cont, cov_mut, diff = diffs_covs(control_bouts, mutant_bouts)

    print(f'covariances: control - {cov_cont}, mutant - {cov_mut}')
    print(f'difference: {diff}')
    print(f'total variance explained: {total_variance}')

    print(f'energy: {calculate_energy(control_bouts, mutant_bouts)}')
    print(f'mahalanobis: {calculate_mahalanobis(control_bouts, mutant_bouts)}')