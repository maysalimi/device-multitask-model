    "RUNNING_TASKS_topActivity_mClass",
    "RUNNING_TASKS_topActivity_mPackage",
]

app_usage_df = app_usage_df.drop(
    columns=[column for column in columns_to_drop if column in app_usage_df.columns]
)


# Standardize numeric features
numeric_columns = app_usage_df.select_dtypes(include="number").columns
scaler = StandardScaler()
app_usage_df[numeric_columns] = scaler.fit_transform(app_usage_df[numeric_columns])


# Explore feature correlations
correlation_matrix = app_usage_df.corr(numeric_only=True)

plt.figure(figsize=(12, 8))
sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5,
)
plt.tight_layout()
plt.show()


# Define features and target
target_column = "RUNNING_TASKS_numRunning"
X = app_usage_df.drop(columns=[target_column])
y = app_usage_df[target_column]


# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)


# Train a Random Forest regression model
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)


# Evaluate with cross-validation
cv_scores = cross_val_score(
    rf_model,
    X_train,
    y_train,
    cv=5,
    scoring="neg_mean_squared_error",
)

mse_scores = -cv_scores
print("MSE across folds:", mse_scores)
print("Average MSE:", np.mean(mse_scores))


# Evaluate on the test set
y_pred = rf_model.predict(X_test)

mse_test = mean_squared_error(y_test, y_pred)
r2_test = r2_score(y_test, y_pred)

print("Test MSE:", mse_test)
print("Test R²:", r2_test)


# Display feature importance
feature_importances = rf_model.feature_importances_
sorted_indices = np.argsort(feature_importances)[::-1]

for index in sorted_indices:
    print(f"{X_train.columns[index]}: {feature_importances[index]}")


# Tune Random Forest hyperparameters
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

grid_search = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_grid=param_grid,
    cv=3,
    n_jobs=-1,
    verbose=2,
    scoring="neg_mean_squared_error",
)

grid_search.fit(X_train, y_train)

best_rf_model = grid_search.best_estimator_
y_pred_best = best_rf_model.predict(X_test)

print("Best parameters:", grid_search.best_params_)
print("Best model test MSE:", mean_squared_error(y_test, y_pred_best))
print("Best model test R²:", r2_score(y_test, y_pred_best))


# Save the trained model
joblib.dump(best_rf_model, "random_forest_model.pkl")


# Compare actual and predicted values
plt.figure(figsize=(7, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "k--",
    linewidth=2,
)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs. Predicted Values")
plt.tight_layout()
plt.show()
