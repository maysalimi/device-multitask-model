# device-multitask-model
This project builds a machine learning model to predict the number of running tasks on a smartphone using passive sensing data. Features include app usage patterns, time-based information (e.g., hour, weekday), and background process activity.

What the model does?
> Takes input features like app usage metrics and timestamp-derived features.
> Predicts how many apps or processes are running in the background.
> Uses a Random Forest Regressor with feature scaling, hyperparameter tuning, and SHAP interpretability.

Contents?
Data loading and preprocessing from the StudentLife dataset.
Feature engineering (e.g., timestamp to hour, weekday).
Random Forest training, evaluation, and optimization.
Model saving with joblib.
Visualization: Correlation heatmap, actual vs predicted plots.

Performance?
R² score (test): ~0.83 – The model explains ~83% of the variance in the number of running tasks.
MSE (test): ~0.14 – On average, predictions are off by a small squared error.

By estimating device multitasking behavior, this model can support applications in:
Cognitive load monitoring
Mental health assessment
Behavioral analysis through passive sensing
