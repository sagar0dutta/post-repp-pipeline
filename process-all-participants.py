##################################################################################
# Script to batch process REPP analysis for multiple participants in a directory
##################################################################################

# If you encounter any memory error, simply rerun the script for the remaining participants.
# Resume from the last successfully processed participant.


import os
import pickle
import pandas as pd

from post_repp_utils import (
    setup_participant_directories,
    process_participant_audio_files,
    run_repp_analysis_for_participant,
    )

from custom_config import sms_tapping, get_config_for_recording     # see custom_config.py
# from repp.config import sms_tapping     # default repp config
from repp.config import ConfigUpdater


# configure paths

############## OSLO EXPERIMENT ###################
# work_dir = r"D:\pyspace\Djembe\psynet\data_collection_clapping_experiment\Pilot_May2025_vito"
# exp_dir = r"oslo-pilot1\regular"     
# exp_dir = r"oslo-pilot2\regular"     
# exp_dir = r"oslo-pilot3\regular"     

########### NOVEMBER EXPERIMENT ################
work_dir = r"D:\pyspace\Djembe\psynet\data_2025\Group-1\November-2025"      # Set base directory here
# exp_dir = r"mali-group1-aws"
exp_dir = r"italy-group1-aws" 
# exp_dir = r"us-group-aws" 

########### OCTOBER EXPERIMENT ################
# work_dir = r"D:\pyspace\Djembe\psynet\data_2025\Group-1\October-2025"
# exp_dir = r"mali-group1-mpi\regular"
# exp_dir = r"italy-group1-mpi\regular"

#################################################

base_dir = os.path.join(work_dir, exp_dir)
output_dir = os.path.join("output", exp_dir)


TapTrialMusic_path = os.path.join(base_dir, "data", "TapTrialMusic.csv")
TapTrialMusic_df = pd.read_csv(TapTrialMusic_path)

participant_ids = TapTrialMusic_df['participant_id'].unique()

print("Sub-directories of assets", os.listdir(os.path.join(base_dir, "assets")))
print("Participant Ids:", participant_ids)


# choose_sub_dir = "Task 1"       # specify the sub-directory in assets -----------------------------

# config_params = sms_tapping
# config_params= ConfigUpdater.create_config(
#     sms_tapping,    # see custom_config.py
#     {
#         'EXTRACT_THRESH': [0.12, 0.18],        # [0.12, 0.12] -- for future
#         'TAPPING_RANGE': [200, 400], 
#         'MARKERS_MAX_ERROR': 33,           # last used: 30 ms ,current: 33
#         ## TODO: add a parameter that extend the MARKER ERROR THRESHOLD to 20.
#         }
#     )


config_params = get_config_for_recording    # see custom_config.py

for choose_sub_dir in ["Task 1", "Task 2"]:    # Loop over both sub-directories
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
            participant_audio_fnames,
            participant_dir, 
            output_participant_dir,         # output_participant_dir
            TapTrialMusic_df,
            overwrite=False  # Set to True to reprocess existing files
        )

        # Run REPP analysis for all recordings
        results = run_repp_analysis_for_participant(
            audio_stim_pairs,
            output_participant_dir,
            choose_sub_dir,
            config_params,
            title_plot= audio_stim_pairs[0],   # audio file name
            display_plots=False,
            figsize=(14, 12)
        )


        save_reslt_path = os.path.join(output_participant_dir, f'participant_{choose_participant_id}_repp_analysis_results.pkl')
        with open(save_reslt_path, 'wb') as f:
            pickle.dump(results, f)
    