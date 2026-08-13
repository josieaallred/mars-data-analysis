# use this function to get clean_behavior_data which can optionally take in a filepath
# note that behavior_to_idx also can optionally take in a filepath

import numpy as np
from pathlib import Path

control_path = Path('/Users/josieallred/SURF/data/RIM/control')
mutant_path = Path('/Users/josieallred/SURF/data/RIM/mutant')

def build_behavior_mapping(control_path=control_path, mutant_path=mutant_path):
    """
    Scans all .annot files to build a universal dictionary mapping behavior strings to unique integers.
    """
    unique_behaviors = set()
    all_annot_files = list(control_path.glob("*.annot")) + list(mutant_path.glob("*.annot"))
    
    for annot_file in all_annot_files:
        with open(annot_file, "r", encoding="latin1") as f:
            in_list = False
            for line in f:
                line = line.strip()
                if line == "List of annotations:":
                    in_list = True
                    continue
                if in_list:
                    if not line or line.startswith("Ch1") or line.startswith("List of channels:"):
                        break 
                    unique_behaviors.add(line)

    behavior_mapping = {'unlabeled': 0}

    for idx, behavior in enumerate(sorted(list(unique_behaviors)), start=1):
        behavior_mapping[behavior] = idx
    return behavior_mapping


def parse_trimmed_behavior_array(annot_filename, behavior_mapping, print_trimmed, fps=30):
    """
    Reads an annotation file, builds the integer array, and returns only the sequence 
    between the end of 'intruder_enter' and the start of 'intruder_out'.
    """
    max_frame = 0
    intervals_dict = {}
    in_ch1 = False
    current_behavior = None

    # 1. Parse the file into the intervals dictionary
    with open(annot_filename, "r", encoding="latin1") as f:
        for line in f:
            line = line.strip()

            if line.startswith("Annotation stop frame:"):
                max_frame = int(line.split(":")[1].strip())
                continue
            if line.startswith("Ch"):
                if "Ch1" in line: in_ch1 = True
                else: break  
                continue
            if not in_ch1: continue
            if line.startswith(">"):
                current_behavior = line[1:]
                intervals_dict.setdefault(current_behavior, [])
                continue
            if not line or line.startswith("Start"): continue

            if current_behavior is not None:
                parts = line.split()
                if len(parts) == 3:
                    try:
                        start_f = round(float(parts[0]) * fps)
                        stop_f = round(float(parts[1]) * fps)
                        intervals_dict[current_behavior].append((start_f, stop_f))
                    except ValueError:
                        pass

   # 2. Determine the precise trim points
    enter = 0
    out = max_frame  # Default to end of video if missing
    file_prefix = Path(annot_filename).name[:3]

    if 'intruder_enter' in intervals_dict and intervals_dict['intruder_enter']:
        enter = intervals_dict['intruder_enter'][0][1] 
    else:
        if print_trimmed: print(f"FLAG: {file_prefix} is missing 'intruder_enter'. Defaulting start trim to 0.")

    if 'intruder_out' in intervals_dict and intervals_dict['intruder_out']:
        out = intervals_dict['intruder_out'][0][0] 
    else:
        if print_trimmed: print(f"FLAG: {file_prefix} is missing 'intruder_out'. Defaulting stop trim to video end.")

    # --- HARD CUTOFF FOR 113 BEFORE SLICING ---
    if file_prefix == '113':
        raw_cutoff = int(np.floor(((60 * 20) + 38) * 30))
        out = raw_cutoff

    # 3. Build the full integer array
    behavior_array = np.zeros(max_frame, dtype=np.uint8)

    for behavior, intervals in intervals_dict.items():
        if behavior not in behavior_mapping:
            continue
            
        b_idx = behavior_mapping[behavior]
        for start, stop in intervals:
            stop = min(stop, max_frame) 
            behavior_array[start:stop] = b_idx

    # 4. Return just the sliced timeframe
    return behavior_array[enter:out]


def load_group_encoded_behaviors(group_path, behavior_mapping, print_trimmed):
    """
    Reads all .annot files in a directory directly and returns a list of trimmed integer arrays.
    """
    # Iterate directly over .annot files, entirely bypassing .npz
    annot_files = sorted(list(group_path.glob("*.annot")))
    group_behaviors = []

    for annot_file in annot_files:
        behavior_array = parse_trimmed_behavior_array(annot_file, behavior_mapping, print_trimmed, fps=30)
        group_behaviors.append(behavior_array)
    return group_behaviors


def clean_behavior_data(control_path=control_path, mutant_path=mutant_path, print_trimmed=False):
    """
    Main pipeline function.
    """    
    BEHAVIOR_MAP = build_behavior_mapping(control_path, mutant_path)
    control_behaviors = load_group_encoded_behaviors(control_path, BEHAVIOR_MAP, print_trimmed)
    mutant_behaviors = load_group_encoded_behaviors(mutant_path, BEHAVIOR_MAP, print_trimmed)

    return control_behaviors, mutant_behaviors

def behavior_to_idx(behavior, control_path=control_path, mutant_path=mutant_path):
    '''quick for consistent use later'''
    BEHAVIOR_MAP = build_behavior_mapping(control_path, mutant_path)
    return BEHAVIOR_MAP[behavior]
