# Time Series and Machine Learning Analysis of ECG Heartbeat Signals

This project studies ECG heartbeat signals using both time series ARIMA modeling and machine learning classification methods.

Repository files:
| Path | Purpose |
|---|---|
| `arima.ipynb` | TimeSeries and ARIMA modeling |
| `forest_ml.ipynb` | Random Forest Machine Learning Classification |
| `writeup.pdf` | Entire report, findings and code |
| `final.ipynb` | Code used for report |


## UPDATE Sep 25th:

Built a 1D CNN in PyTorch to classify ECG heartbeats into all five classes. I started with a baseline CNN and experimented with dropout, weight decay, normalization, learning rate, model architecture, and other training configurations to improve performance.

The final model achieved approximately 98% test accuracy and 0.90 macro F1 score. I used MLflow to track experiments, parameters, metrics, and model performance throughout development.

The final PyTorch model was exported to ONNX and deployed with ONNX Runtime Web, allowing inference to run directly in the browser.

Live Demo:  
https://harrisonfunk2.github.io/ecg_heartbeat_time_series_and_machine_learning/