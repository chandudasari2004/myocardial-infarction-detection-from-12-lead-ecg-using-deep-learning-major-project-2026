import os
from sklearn.model_selection import train_test_split
from data_generator import patient_generator, count_batches
from model import build_model

ptbdb_path = '/content/ptbdb/physionet.org/files/ptbdb/1.0.0/'

# Load patients
all_patients = sorted([
    f for f in os.listdir(ptbdb_path)
    if os.path.isdir(os.path.join(ptbdb_path, f))
])

# Simple split
train_patients, test_patients = train_test_split(all_patients, test_size=0.2, random_state=42)

# Model
model = build_model(5000, 12)
model.summary()

# Steps
train_steps = count_batches(train_patients, 32)
test_steps = count_batches(test_patients, 32)

# Training
history = model.fit(
    patient_generator(train_patients, batch_size=32),
    validation_data=patient_generator(test_patients, batch_size=32),
    steps_per_epoch=train_steps,
    validation_steps=test_steps,
    epochs=10
)

# Save model
model.save('../models/mi_model.h5')
print("Model saved!")
