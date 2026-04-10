import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from tensorflow.keras.models import load_model
from data_generator import patient_generator

model = load_model('../models/mi_model.h5')

y_true = []
y_pred_probs = []

for X_batch, y_batch in patient_generator([], batch_size=32):
    preds = model.predict(X_batch)
    y_true.extend(y_batch)
    y_pred_probs.extend(preds.flatten())

y_true = np.array(y_true)
y_pred_probs = np.array(y_pred_probs)
y_pred = (y_pred_probs > 0.5).astype(int)

print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
print("Classification Report:\n", classification_report(y_true, y_pred))
print("ROC AUC:", roc_auc_score(y_true, y_pred_probs))
