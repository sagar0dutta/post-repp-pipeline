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
    'STIM_BEGINNING': 3000.0,       
    'MARKERS_END': 3000.0,          
    'MARKERS_END_SLACK': 6000.0,    
    # failing criteria
    'MIN_RAW_TAPS': 5,              
    'MAX_RAW_TAPS': 250,            
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
    'DISPLAY_PLOTS': True,
    'PLOTS_TO_DISPLAY': [3, 4]
    }

sms_tapping = ConfigUpdater(parameters)


def get_config_for_recording(participant_id, trial_id, choose_sub_dir=None):
    """
    Return config parameters based on participant and recordings.
    """
    base_config = sms_tapping
    
    ### Task 1
    participant_ids_06 = [12, 26]
    
    trial_ids_04 = [237]
    trial_ids_06 = [89,91,90,93,151,152,154,153,155,215,235,236,239,
                    240,238,450,491,492,490,489,487,510,511,509,508,
                    513,512]
    trial_ids_08 = [488]
    
    ### Task 2
    participant_ids_05_t2 = [11]
    participant_ids_08_t2 = [8]
    participant_ids_06_t2 = [12, 26, 24, 29]
    trial_ids_06_t2 = [306, 413, 416, 415]
    
    
    ### Task 1
    if participant_id in participant_ids_06 and choose_sub_dir == "Task 1":
        overrides = {
            'EXTRACT_THRESH': [0.12, 0.06],
            'TAPPING_RANGE': [200, 450],
            'MARKERS_MAX_ERROR': 35,
        }
    
    elif trial_id in trial_ids_04 and choose_sub_dir == "Task 1":
        overrides = {
            'EXTRACT_THRESH': [0.12, 0.04],
            'TAPPING_RANGE': [200, 450],
            'MARKERS_MAX_ERROR': 35,
        }
    
    elif trial_id in trial_ids_06 and choose_sub_dir == "Task 1":
        overrides = {
            'EXTRACT_THRESH': [0.12, 0.06],
            'TAPPING_RANGE': [200, 450],
            'MARKERS_MAX_ERROR': 35,
        }
    
    
    elif trial_id in trial_ids_08 and choose_sub_dir == "Task 1":
        overrides = {
            'EXTRACT_THRESH': [0.12, 0.08],
            'TAPPING_RANGE': [200, 450],
            'MARKERS_MAX_ERROR': 35,
        }
        
    ### Task 2   
    elif participant_id in participant_ids_05_t2 and choose_sub_dir == "Task 2":
        overrides = {
            'EXTRACT_THRESH': [0.12, 0.05],
            'TAPPING_RANGE': [200, 450],
            'MARKERS_MAX_ERROR': 35,
        }    
    
    elif participant_id in participant_ids_06_t2 and choose_sub_dir == "Task 2":
        overrides = {
            'EXTRACT_THRESH': [0.12, 0.06],
            'TAPPING_RANGE': [200, 450],
            'MARKERS_MAX_ERROR': 35,
        }
    
    elif participant_id in participant_ids_08_t2 and choose_sub_dir == "Task 2":
        overrides = {
            'EXTRACT_THRESH': [0.12, 0.08],
            'TAPPING_RANGE': [200, 450],
            'MARKERS_MAX_ERROR': 35,
        }
    
    elif trial_id in trial_ids_06_t2 and choose_sub_dir == "Task 2":
        overrides = {
            'EXTRACT_THRESH': [0.12, 0.06],
            'TAPPING_RANGE': [200, 450],
            'MARKERS_MAX_ERROR': 35,
        }
    
    
    
    else:  #  default
        overrides = {
            'EXTRACT_THRESH': [0.12, 0.03],
            'TAPPING_RANGE': [200, 400],
            'MARKERS_MAX_ERROR': 33,
        }
    
    return ConfigUpdater.create_config(base_config, overrides)