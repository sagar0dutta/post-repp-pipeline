
import os
import json
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from typing import List, Tuple, Dict, Optional

import matplotlib.image as mpimg
from matplotlib import pyplot as plt
from repp_beatfinding.beat_detection import do_beat_detection_analysis


def parse_repp_analysis(analysis_str):
    """
    Parse nested JSON strings inside a REPP/psynet analysis field.
    Returns fully expanded dictionaries.
    """

    # ---- 1. Handle NaN / empty ----
    if pd.isna(analysis_str) or analysis_str == "":
        return {}

    # ---- 2. Try parsing outer JSON ----
    try:
        data = json.loads(analysis_str)
    except Exception as e:
        print("❌ Outer json.loads failed:", e)
        return {"_raw": analysis_str}

    # ---- 3. Recursively parse any nested JSON strings ----
    def recursive_parse(value):
        """
        Recursively try json.loads on string fields.
        """
        # Case A: nested dict → traverse
        if isinstance(value, dict):
            return {k: recursive_parse(v) for k, v in value.items()}

        # Case B: list → traverse
        if isinstance(value, list):
            return [recursive_parse(x) for x in value]

        # Case C: string → maybe JSON?
        if isinstance(value, str):
            try:
                # Try to decode as JSON
                nested = json.loads(value)
                return recursive_parse(nested)
            except Exception:
                return value  # keep original string

        # Case D: any other type
        return value

    return recursive_parse(data)



def load_stim_info_from_csv(trial_id:int, df: pd.DataFrame) -> dict:

    row = df[df['id'] == trial_id]
    if row.empty:
        raise ValueError(f"No trial found in CSV for ID: {trial_id}")
    row = row.iloc[0]

    stim_duration = float(row['duration_sec'])

    try:
        analysis_parsed = parse_repp_analysis(row['vars'])
    except Exception as e:
        raise RuntimeError(f"Could not parse: {e}")

    stim_info = {
        "stim_duration": stim_duration,
        "stim_onsets": [],
        "stim_shifted_onsets": [],
        "onset_is_played": [],
        "markers_onsets": analysis_parsed['analysis']['output']['markers_onsets_input'],
        "stim_name": analysis_parsed['analysis']['stim_name'],
    }

    return stim_info


def extract_trial_id_from_filename(audio_fname: str) -> int:
    """
    Extract trial ID from audio filename.
    
    Expected format: "node_10__trial_7__trial_main_page.wav"
    
    Args:
        audio_fname: Audio filename
        
    Returns:
        trial_id: Integer trial ID
    """
    parts = audio_fname.split("__")
    trial_id = int(parts[1].split("_")[1])
    return trial_id


def convert_and_save_audio(
    audio_path: str, 
    output_path: str, 
    target_sr: int = 44100,
    overwrite: bool = False
) -> Tuple[np.ndarray, int]:
    """
    Convert audio file to mono and resample to target sample rate.
    
    Args:
        audio_path: Path to input audio file
        output_path: Path to save converted audio
        target_sr: Target sample rate (default: 44100)
        overwrite: Whether to overwrite existing file (default: False)
        
    Returns:
        Tuple of (audio_data, sample_rate)
    """
    if not overwrite and os.path.exists(output_path):
        # Load existing file instead of converting
        data, fs = sf.read(output_path)
        return data, fs
    
    # Read audio file
    data, fs = sf.read(audio_path)
    
    # Convert to mono if stereo
    if len(data.shape) == 2:
        data = np.mean(data, axis=1)
    
    # Resample if needed
    if fs != target_sr:
        data = librosa.resample(data, orig_sr=fs, target_sr=target_sr)
        fs = target_sr
    
    # Save converted audio
    sf.write(output_path, data, fs, subtype='PCM_16')
    # print(f"WAV converted and saved to {os.path.dirname(output_path)}")
    
    return data, fs


def setup_participant_directories(
    base_dir: str,
    choose_sub_dir: str,
    choose_participant_id: int,
    output_dir: str
) -> Tuple[str, str, List[str]]:
    """
    Set up and validate participant directories.
    
    Args:
        base_dir: Base directory containing assets
        choose_sub_dir: Subdirectory name (e.g., "Task 1")
        choose_participant_id: Participant ID
        output_dir: Output directory for processed files
        
    Returns:
        Tuple of (participant_dir, output_participant_dir, participant_audio_fnames)
        
    Raises:
        ValueError: If participant directory does not exist
    """
    participant_dir = os.path.join(
        base_dir, "assets", choose_sub_dir, "participants", 
        f"participant_{choose_participant_id}"
    )
    output_participant_dir = os.path.join(output_dir, choose_sub_dir, f"participant_{choose_participant_id}")
    
    if not os.path.exists(participant_dir):
        raise ValueError(
            f"Participant directory does not exist: {participant_dir}. "
            f"Choose another participant id."
        )
    
    participant_audio_fnames = [f for f in os.listdir(participant_dir) if f.endswith('.wav')]
    os.makedirs(output_participant_dir, exist_ok=True)
    
    return participant_dir, output_participant_dir, participant_audio_fnames


def process_participant_audio_files(
    participant_id: int,
    participant_dir: str,
    output_participant_dir: str,
    TapTrialMusic_df: pd.DataFrame,
    overwrite: bool = False
) -> List[Tuple[str, str, str]]:
    """
    Process all audio files for a participant: convert audio and extract stimulus info.
    
    Args:
        participant_dir: Directory containing participant audio files
        output_participant_dir: Directory to save processed files
        TapTrialMusic_df: DataFrame containing trial metadata
        overwrite: Whether to overwrite existing files (default: False)
        
    Returns:
        List of tuples: (audio_basename, audio_fname, stim_info_fname)
    """
    participant_audio_fnames = [f for f in os.listdir(participant_dir) if f.endswith('.wav')]
    audio_stim_pairs = []
    
    for audio_fname in participant_audio_fnames:
        # Remove .wav extension to get basename
        audio_basename = os.path.splitext(audio_fname)[0]
        trial_id = extract_trial_id_from_filename(audio_fname)
        
        audio_path = os.path.join(participant_dir, audio_fname)
        output_audio_path = os.path.join(output_participant_dir, f"participant_{participant_id}__{audio_fname}")
        stim_info_json_path = os.path.join(output_participant_dir, f"participant_{participant_id}__{audio_basename}_stim_info.json")
        plot_path = os.path.join(output_participant_dir, f"participant_{participant_id}__{audio_basename}.png")
        
        # check if stim_info and output_audio_path file already exists
        if not overwrite and os.path.exists(stim_info_json_path) and os.path.exists(output_audio_path) and os.path.exists(plot_path):
            continue
        
        
        # Convert and save WAV file
        convert_and_save_audio(audio_path, output_audio_path, overwrite=overwrite)
        
        # Extract and save stimulus info
        stim_info = load_stim_info_from_csv(trial_id, TapTrialMusic_df)
        
        if overwrite or not os.path.exists(stim_info_json_path):
            with open(stim_info_json_path, 'w') as f:
                json.dump(stim_info, f, indent=4)
            # print(f"stim_info saved: {audio_basename}_stim_info.json")
        
        audio_stim_tup = (f"participant_{participant_id}__{audio_basename}", f"participant_{participant_id}__{audio_fname}", f"participant_{participant_id}__{audio_basename}_stim_info.json")
        audio_stim_pairs.append(audio_stim_tup)
    
    print(f"WAV files converted and saved to {output_participant_dir}")
    print(f"Stim_info files saved to {output_participant_dir}")
    
    return audio_stim_pairs


def run_repp_analysis_for_participant(
    audio_stim_pairs: List[Tuple[str, str, str]],
    output_participant_dir: str,
    config,
    title_plot: str = 'Beat Finding Analysis',
    display_plots: bool = True,
    figsize: Tuple[int, int] = (14, 12)
) -> List[Dict]:
    """
    Run REPP beat detection analysis for all participant recordings.
    
    Args:
        audio_stim_pairs: List of tuples (basename, audio_fname, stim_info_fname)
        output_participant_dir: Directory containing processed files
        config: REPP configuration object
        title_plot: Title for plots
        display_plots: Whether to display plots inline (default: True)
        figsize: Figure size for plots (default: (14, 12))
        
    Returns:
        List of analysis results dictionaries
    """
    
    
    results = []
    
    for recording_basename, recording_fname, stim_info_fname in audio_stim_pairs:
        # Define filenames for outputs
        plot_filename = f'{recording_basename}.png'
        recording_path = os.path.join(output_participant_dir, recording_fname)
        plot_path = os.path.join(output_participant_dir, plot_filename)
        stim_info_path = os.path.join(output_participant_dir, stim_info_fname)
        
        # Load stimulus info
        with open(stim_info_path, 'r') as f:
            stim_info = json.load(f)
        
        print("-------------------------------------------------\n Running REPP\n")
        
        # Run REPP analysis
        output, extracted_onsets, stats = do_beat_detection_analysis(
            recording_path,
            title_plot,
            plot_path,
            stim_info=stim_info,
            config=config
        )
        
        # print("extracted onsets:-----------------------------\n")
        # print(extracted_onsets)
        # print("-------------------------------------------------\n")
        
        # Display plot if requested
        if display_plots:
            display_analysis_plot(plot_path, figsize=figsize)
        
        results.append({
            'recording_basename': recording_basename,
            'extracted_onsets': extracted_onsets,
            'stats': stats,
            'output': output
        })
    
    return results


def display_analysis_plot(plot_path: str, figsize: Tuple[int, int] = (14, 12)):
    """
    Display analysis plot from saved image file.
    
    Args:
        plot_path: Path to saved plot image
        figsize: Figure size (default: (14, 12))
    """
    
    plt.clf()
    plt.figure(figsize=figsize)
    img = mpimg.imread(plot_path)
    imgplot = plt.imshow(img)
    plt.axis('off')
    plt.tight_layout()