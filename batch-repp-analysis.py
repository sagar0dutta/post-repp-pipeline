##################################################################################
# Script to batch process REPP analysis for multiple participants in a directory
##################################################################################

# If you encounter any memory error, simply rerun the script for the remaining participants.
# Resume from the last successfully processed participant.


import os
import pickle
import pandas as pd

from post_repp_pipeline import (
    setup_participant_directories,
    process_participant_audio_files,
    run_repp_analysis_for_participant,
    )

from custom_config import sms_tapping     # see custom_config.py
# from repp.config import sms_tapping     # default repp config
from repp.config import ConfigUpdater





# configure paths
# base_dir = r"D:\pyspace\Djembe\psynet\data_2025\Group-1\November-2025\italy-group1-final"       # Set base directory here
# base_dir = r"D:\pyspace\Djembe\psynet\data_2025\Group-1\November-2025\mali-group1-final"      # Set base directory here
base_dir = r"D:\pyspace\Djembe\psynet\data_2025\Group-1\November-2025\us-group-nov9"      # Set base directory here -----------------------------------
output_dir = r"output"


TapTrialMusic_path = os.path.join(base_dir, "data", "TapTrialMusic.csv")
TapTrialMusic_df = pd.read_csv(TapTrialMusic_path)

participant_ids = TapTrialMusic_df['participant_id'].unique()

print("Sub-directories of assets", os.listdir(os.path.join(base_dir, "assets")))
print("Participant Ids:", participant_ids)


choose_sub_dir = "Task 2"       # specify the sub-directory in assets -----------------------------

# config_params = sms_tapping
config_params= ConfigUpdater.create_config(
    sms_tapping,    # see custom_config.py
    {
        'EXTRACT_THRESH': [0.225, 0.12],        # [0.225, 0.12] -- default
        'EXTRACT_COMPRESS_FACTOR': 1.1,
        'EXTRACT_FIRST_WINDOW': [18, 18],           # [18, 18]
        'EXTRACT_SECOND_WINDOW': [26, 120],         # [26, 120]
        'MARKERS_MATCHING_WINDOW': 35.0,
        'TAPPING_RANGE': [200, 400], 
        'MARKERS_MAX_ERROR': 30,
        ## TODO: add a parameter that extend the MARKER ERROR THRESHOLD to 20.

        }
    )

for choose_participant_id in participant_ids:

    # Setup participant directories
    participant_dir, output_participant_dir, participant_audio_fnames = setup_participant_directories(
        base_dir, choose_sub_dir, choose_participant_id, output_dir
        )

    # Skip if source participant directory doesn't exist
    if participant_dir is None:
        print(f"Skipping participant {choose_participant_id}: directory not found")
        continue

    # Skip if already processed
    if os.path.exists(output_participant_dir):
        # print("Skipping already processed participant:", choose_participant_id)
        continue
    
    os.makedirs(output_participant_dir, exist_ok=True)
    # Process all audio files: convert audio and extract stimulus info
    audio_stim_pairs = process_participant_audio_files(
        choose_participant_id,
        participant_dir, 
        output_participant_dir,         # output_participant_dir
        TapTrialMusic_df,
        overwrite=False  # Set to True to reprocess existing files
    )



    # Run REPP analysis for all recordings
    results = run_repp_analysis_for_participant(
        audio_stim_pairs,
        output_participant_dir,
        config_params,
        title_plot= audio_stim_pairs[0],   # audio file name
        display_plots=False,
        figsize=(14, 12)
    )


    save_reslt_path = os.path.join(output_participant_dir, f'participant_{choose_participant_id}_repp_analysis_results.pkl')
    with open(save_reslt_path, 'wb') as f:
        pickle.dump(results, f)
    