##################################################################################
# Script to batch process REPP analysis for multiple participants in a directory
##################################################################################

# If you encounter any memory error, simply rerun the script for the remaining participants.
# Resume from the last successfully processed participant.


import os
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
base_dir = r"D:\pyspace\Djembe\psynet\data_2025\Group-1\November-2025\us-group-nov9"      # Set base directory here
output_dir = r"output"


TapTrialMusic_path = os.path.join(base_dir, "data", "TapTrialMusic.csv")
TapTrialMusic_df = pd.read_csv(TapTrialMusic_path)

print("Sub-directories of assets", os.listdir(os.path.join(base_dir, "assets")))
print("Participant Ids:", TapTrialMusic_df['participant_id'].unique())

participant_ids = TapTrialMusic_df['participant_id'].unique()

choose_sub_dir = "Task 2"       # specify the sub-directory in assets

config_params = sms_tapping
# config_params= ConfigUpdater.create_config(
#     sms_tapping,    # see custom_config.py
#     {
#         'EXTRACT_THRESH': [0.225, 0.12],
#         'EXTRACT_COMPRESS_FACTOR': 1,
#         'EXTRACT_FIRST_WINDOW': [18, 18],
#         'EXTRACT_SECOND_WINDOW': [26, 120],
#         ## TODO: add a parameter that extend the MARKER ERROR THRESHOLD to 20.

#     }
# )

for choose_participant_id in participant_ids:


    participant_dir = os.path.join(
            base_dir, "assets", choose_sub_dir, "participants", 
            f"participant_{choose_participant_id}"
        )

    if not os.path.exists(participant_dir):
        continue


    # Setup participant directories
    participant_dir, output_participant_dir, participant_audio_fnames = setup_participant_directories(
        base_dir, choose_sub_dir, choose_participant_id, output_dir
    )

    
    
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


    