import os
import numpy as np
from data_preprocessing import get_label_from_header, load_and_preprocess

ptbdb_path = '/content/ptbdb/physionet.org/files/ptbdb/1.0.0/'

def patient_generator(patient_folders, batch_size=32):
    X_batch, y_batch = [], []

    for patient_folder in patient_folders:
        patient_dir = os.path.join(ptbdb_path, patient_folder)

        if os.path.isdir(patient_dir):
            for file in os.listdir(patient_dir):
                if file.endswith('.hea'):
                    record_prefix = file[:-4]
                    record_path = os.path.join(patient_dir, record_prefix)

                    label = get_label_from_header(record_path + '.hea')
                    segments = load_and_preprocess(record_path)

                    for segment in segments:
                        X_batch.append(segment)
                        y_batch.append(label)

                        if len(X_batch) == batch_size:
                            yield np.array(X_batch), np.array(y_batch)
                            X_batch, y_batch = [], []

    if X_batch:
        yield np.array(X_batch), np.array(y_batch)


def count_batches(patient_folders, batch_size):
    count = 0
    for _ in patient_generator(patient_folders, batch_size):
        count += 1
    return count
