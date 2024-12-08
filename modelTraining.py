from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

def train_and_evaluate_model(X, y, create_model_func, class_weights=None, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = []
    trained_model = None

    fold = 1
    for train_idx, test_idx in tscv.split(X):
        print(f"Training Fold {fold}...")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = create_model_func(X_train.shape[1:])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_test, y_test),
            class_weight=class_weights,
            callbacks=[early_stop],
            verbose=1
        )

        # Predictions
        y_pred_prob = model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int).reshape(-1)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        print(f"Fold {fold} Results - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1 Score: {f1:.4f}")
        results.append((acc, prec, rec, f1))

        # Save the last trained model
        trained_model = model

        fold += 1

    return results, trained_model

def save_model(model, scaler, label_encoder, output_dir='artifacts'):
    import os
    import joblib

    os.makedirs(output_dir, exist_ok=True)

    # Save the model in the Keras
    model.save(os.path.join(output_dir, 'model.keras'))
    print(f"Model saved to {os.path.join(output_dir, 'model.keras')}")

    # Save the scaler and label encoder using joblib
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.joblib'))
    joblib.dump(label_encoder, os.path.join(output_dir, 'label_encoder.joblib'))
    print(f"Scaler and label encoder saved to {output_dir}")

def save_metrics(results, output_dir='artifacts'):
    import os
    import json

    os.makedirs(output_dir, exist_ok=True)

    metrics = {
        'folds': [],
        'average': {}
    }

    acc_list = []
    prec_list = []
    rec_list = []
    f1_list = []

    for idx, (acc, prec, rec, f1) in enumerate(results, start=1):
        metrics['folds'].append({
            'fold': idx,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1
        })
        acc_list.append(acc)
        prec_list.append(prec)
        rec_list.append(rec)
        f1_list.append(f1)

    metrics['average'] = {
        'accuracy': np.mean(acc_list),
        'precision': np.mean(prec_list),
        'recall': np.mean(rec_list),
        'f1_score': np.mean(f1_list)
    }

    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {os.path.join(output_dir, 'metrics.json')}")

