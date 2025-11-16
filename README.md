# Post-hoc REPP Analysis Pipeline

Post-hoc analysis pipeline for processing tapping data from psynet experiments using REPP (Rhythm Extraction and Pulse Prediction) beat detection algorithm.

## Overview

This pipeline processes participant audio recordings from tapping experiments and performs beat detection analysis to extract rhythmic patterns and onsets. It handles data preprocessing, stimulus information extraction, and generates visualization plots for each recording.

## Features

- **Audio Processing**: Converts stereo recordings to mono and resamples to 44.1 kHz
- **Stimulus Information Extraction**: Extracts trial metadata from CSV files
- **REPP Beat Detection**: Runs REPP analysis with configurable parameters
- **Visualization**: Generates beat detection plots for each recording
- **Batch Processing**: Processes all recordings for a participant automatically

## Installation

### Prerequisites

- Python 3.8+
- Required Python packages (see Dependencies below)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd notebook
```

2. Install dependencies:
```bash
pip install pandas numpy soundfile librosa matplotlib repp
```

## Usage

### Basic Workflow

1. **Configure paths and load data**:
   - Set `base_dir` to your data directory containing `assets/` and `data/TapTrialMusic.csv`
   - The pipeline will automatically discover participant directories

2. **Select participant**:
   - Choose the subdirectory (e.g., "Task 1", "Task 2")
   - Select participant ID

3. **Run the pipeline**:
   - Execute the notebook cells in order
   - The pipeline will:
     - Convert and save audio files
     - Extract stimulus information
     - Run REPP beat detection analysis
     - Generate visualization plots


## Project Structure

```
notebook/
├── post_repp_pipeline.ipynb      # Main analysis notebook
├── post_repp_pipeline.py         # Core pipeline functions
├── custom_config.py              # REPP configuration parameters
├── repp_beatfinding/             # Beat detection analysis module
│   ├── beat_detection.py
│   └── enhanced_tapping_analysis.py
└── output/                       # Generated outputs (per participant)
    └── participant_X/
        ├── *.wav                 # Converted audio files
        ├── *_stim_info.json      # Stimulus metadata
        └── *.png                 # Beat detection plots
```


## Output

For each participant recording, the pipeline generates:

- **Audio files** (`.wav`): Converted and processed audio recordings
- **Stimulus info** (`.json`): Trial metadata including
- **Analysis plots** (`.png`): Beat detection visualizations



## Data Format

### Expected Directory Structure

```
base_dir/
├── data/
│   └── TapTrialMusic.csv        # Trial metadata
└── assets/
    └── Task X/
        └── participants/
            └── participant_Y/
                └── *.wav        # Audio recordings
```

### Audio Filename Format

Expected format: `node_X__trial_Y__trial_main_page.wav`

- Extracts trial ID from filename for metadata lookup


