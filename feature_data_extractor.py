# use this to get clean_feature_data
# note that feat_to_idx relies on the presence of features.npy in the file, 
# if it is not there, it must be an argument in that function

import re
import numpy as np
from pathlib import Path


control_path = Path('/Users/josieallred/SURF/data/RIM/control')
mutant_path = Path('/Users/josieallred/SURF/data/RIM/mutant')
MOUSE_NUM = 0 # can be 1 or 0 -- 0 is brown resident, 1 is white intruder

def parse_bento_annotation(annot_filename, fps=30):
    """
    Creates a dictionary mapping behaviors to frame intervals,
    reading only Ch1 data. Includes metadata such as 'annotation_stop_frame'.
    """
    behaviors = {}
    current_behavior = None
    in_ch1 = False
    
    # Initialize metadata entry before starting the loop
    behaviors['annotation_stop_frame'] = None

    with open(annot_filename, "r", encoding="latin1") as f:
        for line in f:
            line = line.strip()

            # --- PARSE METADATA HEADER (Before Ch1 restrictions) ---
            if line.startswith("Annotation stop frame:"):
                try:
                    behaviors['annotation_stop_frame'] = int(line.split(":")[-1].strip())
                except ValueError:
                    pass
                continue

            # Channel markers: enter Ch1, break on anything else
            if line.startswith("Ch"):
                if "Ch1" in line:
                    in_ch1 = True
                else:
                    break  # Ch2 or beyond — stop entirely
                continue

            # Only process content inside Ch1
            if not in_ch1:
                continue

            # New behavior section
            if line.startswith(">"):
                current_behavior = line[1:]
                behaviors[current_behavior] = []
                continue

            # Skip headers and blank lines
            if not line or line.startswith("Start"):
                continue

            # Parse interval rows
            if current_behavior is not None:
                parts = line.split()
                if len(parts) == 3:
                    try:
                        start_sec = float(parts[0])
                        stop_sec = float(parts[1])
                        behaviors[current_behavior].append(
                            (round(start_sec * fps), round(stop_sec * fps))
                        )
                    except ValueError:
                        pass

    return behaviors


def get_start_stop(annot_filename, fps=30, stop_frame_annot=True):
    ''' use annotation file to get entry and exit frame of the mouse'''

    annot_dict = parse_bento_annotation(annot_filename, fps=fps)
    file_prefix = Path(annot_filename).name[:3]

    if annot_dict.get('intruder_enter'):    # end of entry
        enter_frame = annot_dict['intruder_enter'][0][1]
    else:
        enter_frame = None

    # --- DYNAMIC EXIT FRAME SELECTION ---
    if stop_frame_annot and annot_dict.get('annotation_stop_frame') is not None:
        exit_frame = annot_dict['annotation_stop_frame']
    elif annot_dict.get('intruder_out'):  # beginning of exit
        exit_frame = annot_dict['intruder_out'][0][0]
    else:
        exit_frame = None  # None slices until the end of the array

    # --- ADD THIS LOGIC TO FIX 113 IN THE FEATURE EXTRACTOR ---
    # if file_prefix == '113':
    #     raw_cutoff = int(np.floor(((60 * 20) + 38) * fps))
    #     exit_frame = raw_cutoff
    #     print(f'FEATURE 113 trim: {enter_frame, exit_frame}')

    return enter_frame, exit_frame


def trim_features(np_filename, annot_filename, print_trimmed=False, stop_frame_annot=True):
    '''makes feature array for a give file exclude data before and after intruder entry
    (where noted in annotation file)'''

    data = np.load(np_filename)
    feature_array = data['data_smooth']
    enter_frame, exit_frame = get_start_stop(annot_filename, stop_frame_annot=stop_frame_annot)

    if enter_frame is not None: trim_start = True
    else: trim_start = False

    if exit_frame is not None: trim_stop = True
    else: trim_stop = False
    
    if print_trimmed: 
        # Only raise the flag if an annotation is missing
        if not trim_start or not trim_stop:
            # Safely extract just the first 3 letters of the filename from the Path
            file_prefix = np_filename.name[:3] if hasattr(np_filename, 'name') else str(np_filename)[:3]
            print(f"FLAG: {file_prefix} -- start trim: {trim_start}, stop trim: {trim_stop}")
    
    return feature_array[:, enter_frame: exit_frame, :]



def clean_feature_data(control_path=control_path, mutant_path=mutant_path, print_trimmed=False, stop_frame_annot=True):
    '''
    returns a list of feature data for each mouse in the two control groups
    list of 158xframes arrays
    '''

    # trim to start stop where this is given in annotation file
    control_datafiles = sorted(list(control_path.glob("*.npz")))

    # match with annotation files based on starting prefix
    control_annotfiles = []
    for npz_file in control_datafiles:
        # Extract the first 3 characters (e.g., '001')
        mouse_id = npz_file.name[:3]  
        # Search for an .annot file starting with that prefix
        matching_annots = list(control_path.glob(f"{mouse_id}*.annot"))
        if len(matching_annots) == 0:
            raise FileNotFoundError(f"Could not find a matching .annot file for prefix: {mouse_id}")
            
        # Pick the first match found
        control_annotfiles.append(matching_annots[0])

    # Pass down stop_frame_annot switch
    control_data = [
        trim_features(control_datafiles[i], control_annotfiles[i], print_trimmed, stop_frame_annot)[MOUSE_NUM].T
        for i in range(len(control_datafiles))
    ]  

    # do the same for mutants
    mutant_datafiles = sorted(list(mutant_path.glob("*.npz")))
    mutant_annotfiles = []

    for npz_file in mutant_datafiles:
        mouse_id = npz_file.name[:3]
        matching_annots = list(mutant_path.glob(f"{mouse_id}*.annot"))
        if len(matching_annots) == 0:
            raise FileNotFoundError(f"Could not find a matching .annot file for prefix: {mouse_id}")
            
        mutant_annotfiles.append(matching_annots[0])

    # Pass down stop_frame_annot switch
    mutant_data = [
        trim_features(mutant_datafiles[i], mutant_annotfiles[i], print_trimmed, stop_frame_annot)[MOUSE_NUM].T
        for i in range(len(mutant_datafiles))
    ]
    
    return control_data, mutant_data


def feat_to_idx(feature_name, feature_file='features.npy'):
    '''
    easily get the index associated with a given feature name
    '''

    feature_list = np.load(feature_file)
    feat_to_idx = {feature: i for i, feature in enumerate(feature_list)}
    feat_idx = feat_to_idx[feature_name]

    return feat_idx




