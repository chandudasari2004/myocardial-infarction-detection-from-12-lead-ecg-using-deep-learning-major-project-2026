import numpy as np
import wfdb

window_size = 5000
n_leads = 12

def get_label_from_header(header_path):
    MI_CODES = ["164865005", "Myocardial infarction"]
    try:
        with open(header_path, "r", encoding="utf-8") as f:
            header = f.read()
    except UnicodeDecodeError:
        with open(header_path, "r", encoding="latin1") as f:
            header = f.read()
    return 1 if any(code in header for code in MI_CODES) else 0

def load_and_preprocess(record_path):
    record = wfdb.rdrecord(record_path)
    signal = record.p_signal[:, :n_leads]

    mean = np.mean(signal, axis=0)
    std = np.std(signal, axis=0)
    norm_signal = (signal - mean) / std

    segments = []
    for start in range(0, norm_signal.shape[0] - window_size + 1, window_size):
        segments.append(norm_signal[start:start + window_size, :])

    return np.array(segments)
