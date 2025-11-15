
import json
import pandas as pd




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