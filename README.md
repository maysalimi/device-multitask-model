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

Approach:
To predict the number of running tasks on a phone, I used machine learning. Here's what I did:

Data Cleaning: The data had some missing values, so I filled those gaps using the median (for numbers) and mode (for categories).

Feature Engineering: I turned raw data (like time stamps) into useful features like time of day and which day of the week it is, because people tend to use their phones differently at different times.

Modeling: I chose a Random Forest model, which is great for this kind of task because it can handle lots of different types of data and relationships between features.

Causal Method:
Instead of looking for direct cause-and-effect relationships, I used SHAP (a method to explain machine learning models) to understand which features (like time of day or specific apps) were most important for predicting multitasking.


Performance?
R² score (test): ~0.83  This means that this model could explain 83% of the variation in how many tasks were running. In simpler terms, the model is pretty good at predicting multitasking.
MSE (test): ~0.14  This tells us how close the predictions were to the real values. A lower number is better, so this shows that our model made good predictions overall.

Interpretation:
The model was able to predict multitasking behavior pretty accurately. The R² score of 0.83 is quite good, meaning our model is doing a great job explaining multitasking patterns. However, there is still some room for improvement since 17% of the variation was not explained by the model. The MSE value being low suggests the model's predictions were close to the actual number of tasks running, but there is still a bit of error.

By estimating device multitasking behavior, this model can support applications in:
Cognitive load monitoring
Mental health monitoring: By combining the phone usage data with self-reported mental health surveys (like the PHQ-9), we could use this model to predict mental health issues, like stress or depression, based on smartphone behavior.
Time-Series Predictions: Since phone usage changes over time, it might be better to use time-series models (like LSTM) to predict multitasking patterns more accurately over time.
Real-Time Monitoring: Eventually, we could create a real-time app that uses this model to track multitasking behavior and give feedback to users. For example, it could alert users if they’re multitasking too much and suggest breaks.
