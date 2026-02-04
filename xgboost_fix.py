# Fixed XGBoost training cell - Copy and paste this into your notebook

print("="*60)
print("Training Model 3: XGBoost (GPU)")
print("="*60)

# Train model with GPU acceleration
xgb_config = MODEL_CONFIG['xgboost'].copy()

# Check if GPU is available, otherwise use CPU
if len(tf.config.list_physical_devices('GPU')) == 0:
    print("GPU not detected, using CPU (hist)")
    xgb_config['tree_method'] = 'hist'
    xgb_config.pop('gpu_id', None)

# Add early_stopping_rounds to constructor for newer XGBoost versions
xgb_config['early_stopping_rounds'] = 20

xgb_model = xgb.XGBRegressor(**xgb_config, verbosity=1)
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=True
)

# Predictions
y_pred_xgb = xgb_model.predict(X_test)

# Evaluate
xgb_results = evaluate_model(y_test, y_pred_xgb, "XGBoost")
results.append(xgb_results)

# Visualize
plot_predictions(y_test.values, y_pred_xgb, "XGBoost")
