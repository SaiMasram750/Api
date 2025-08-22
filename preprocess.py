import pandas as pd
import mne
import tempfile
import io

def preprocess_csv(contents):
    df = pd.read_csv(io.BytesIO(contents), header=None)
    return df.values.flatten()

def preprocess_edf(contents):
    with tempfile.NamedTemporaryFile(delete=True, suffix=".edf") as tmp:
        tmp.write(contents)
        tmp.flush()
        raw = mne.io.read_raw_edf(tmp.name, preload=True)
        raw.pick_types(eeg=True)
        raw.crop(tmin=0, tmax=1)
        return raw.get_data()[0]
