from repp.config import ConfigUpdater

# Global parameters
NUM_PARTICIPANTS = 5000
DURATION_ESTIMATED_TRIAL = 40
EXPECTED_TRIALS_PER_PARTICIPANT = 6       # 10
MAX_TRIALS_PER_BLOCK = 2        #5
REPETITION_TASK_TRIAL = ['repeat_A','repeat_B', 'repeat_C']
MAX_TRIALS_PER_PARTICIPANT = 6         # 10
EXTRA_TRIALS_ACCOUNTING_FOR_ERRORS= 2
PRACTICE_MAX_TRIALS_PER_PARTICIPANT = 2     # 5

parameters = {  # Global parameters for sms experiments
    'LABEL': 'sms_tapping',
    'FS': 48000,    # sampling rate
    'FS0': 24000,       # downsampled
    # Stimulus preparation step
    'STIM_RANGE': [30, 1600],
    'STIM_AMPLITUDE': 0.12,
    'MARKERS_RANGE': [200, 340],
    'TEST_RANGE': [100, 170],
    'MARKERS_AMPLITUDE': 0.9,
    'MARKERS_ATTACK': 2,
    'MARKERS_DURATION': 15,
    'MARKERS_IOI': [0, 280, 230],
    'MARKERS_BEGINNING': 1500.0,
    'STIM_BEGINNING': 3000.0,       # changed
    'MARKERS_END': 3000.0,          # changed from 2000 to 3000 on 28 feb
    'MARKERS_END_SLACK': 6000.0,    # changed from 5000 to 6000 on 28 feb
    # failing criteria
    'MIN_RAW_TAPS': 5,              # changed
    'MAX_RAW_TAPS': 250,            # changed
    'MARKERS_MAX_ERROR': 15,
    'MIN_NUM_ASYNC': -1,     # 2
    'MIN_SD_ASYNC': 0,     # 10
    # metronome sound
    'CLICK_FILENAME': 'click01.wav',
    'USE_CLICK_FILENAME': False,
    'CLICK_DURATION': 50,
    'CLICK_FREQUENCY': 1000,
    'CLICK_ATTACK': 5,
    # Onset extraction step
    'TAPPING_RANGE': [80, 300],     
    'EXTRACT_THRESH': [0.225, 0.12],    # 0.225, 0.16       [0.225, 0.12]
    'EXTRACT_FIRST_WINDOW': [18, 18],
    'EXTRACT_SECOND_WINDOW': [26, 120],
    'EXTRACT_COMPRESS_FACTOR': 1.1,
    'EXTRACT_FADE_IN': 500,
    # Cleaning procedure
    'CLEAN_BIN_WINDOW': 100,
    'CLEAN_MAX_RATIO': 10,
    'CLEAN_LOCATION_RATIO': [0.333, 0.66],
    'CLEAN_NORMALIZE_FACTOR': 0.05,
    # Onset alignment step
    'ONSET_MATCHING_WINDOW_MS': 1999.0,  # if you want to use only phase set it to 1999 (2 sec)
    'ONSET_MATCHING_WINDOW_PHASE': [-1, 1],  # for relative phase (if you want to use only ms set it to [-1 1])
    'MARKERS_MATCHING_WINDOW': 35.0,
    # Plotting
    'DISPLAY_PLOTS': True, # TODO BRING THIS BACK
    'PLOTS_TO_DISPLAY': [3, 4]
    }

sms_tapping = ConfigUpdater(parameters)