import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    # Packages
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from sklearn.model_selection import KFold
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import mean_absolute_error
    return GroupKFold, mean_absolute_error, np, pd, tf


@app.cell
def _(pd):
    print("Step 1: Load train, test and sample submission data")
    # Load Train Data
    train = pd.read_csv("data/osic-pulmonary-fibrosis-progression/train.csv")
    train.drop_duplicates(keep=False, inplace=True, subset=['Patient', 'Weeks'])

    # Load Test Data
    test = pd.read_csv("data/osic-pulmonary-fibrosis-progression/test.csv")

    # Load Sample Submission Data
    sample_submission = pd.read_csv("data/osic-pulmonary-fibrosis-progression/sample_submission.csv")

    # Dims of dfs
    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    print(f"Submission shape: {sample_submission.shape}")
    return sample_submission, test, train


@app.cell
def _(pd, sample_submission, test, train):
    print("Step 2: Concat all of the data into a single df for consistent feature engineering")
    # Combine all sample data
    sample_submission['Patient'] = sample_submission['Patient_Week'].str.split('_').str[0]
    sample_submission['Weeks'] = sample_submission['Patient_Week'].str.split('_').str[1].astype(int)
    submission_data = sample_submission.merge(test.drop('Weeks', axis=1), on='Patient', how='left')
    # Mark data sources
    train['data_source'] = 'train'
    test['data_source'] = 'test' 
    submission_data['data_source'] = 'submission'
    # Combine all data into one df
    all_data = pd.concat([
        train[['Patient', 'Weeks', 'FVC', 'Age', 'Sex', 'SmokingStatus', 'Percent', 'data_source']],
        test[['Patient', 'Weeks', 'FVC', 'Age', 'Sex', 'SmokingStatus', 'Percent', 'data_source']],
        submission_data[['Patient', 'Weeks', 'Age', 'Sex', 'SmokingStatus', 'Percent', 'data_source']]
    ], ignore_index=True)
    print(f"Combined data shape: {all_data.shape}")
    return (all_data,)


@app.cell
def _(all_data):
    print("Step 3: Feature Engineering")
    # First measurement of each regressor for each patient 
    baseline_data = all_data[all_data['data_source'] != 'submission'].groupby('Patient').agg({
        'Weeks': 'min',
        'FVC': 'first',
        'Age': 'first',
        'Percent': 'first'
    }).reset_index()
    baseline_data.columns = ['Patient', 'baseline_week', 'baseline_fvc', 'baseline_age', 'baseline_percent']
    # Merge baseline data
    all_data_baseline = all_data.merge(baseline_data, on='Patient', how='left')
    # Create temporal features
    all_data_baseline['weeks_from_baseline'] = all_data_baseline['Weeks'] - all_data_baseline['baseline_week']
    # Normalise numerical features
    all_data_baseline['age_norm'] = (all_data_baseline['Age'] - 50) / 20
    all_data_baseline['weeks_norm'] = all_data_baseline['weeks_from_baseline'] / 100
    all_data_baseline['baseline_fvc_norm'] = (all_data_baseline['baseline_fvc'] - 2000) / 1000
    all_data_baseline['percent_norm'] = (all_data_baseline['Percent'] - 70) / 30
    # One-hot encoding for Qualitative Regressors
    all_data_baseline["is_male"] = (all_data_baseline['Sex'] == 'Male').astype(int)
    all_data_baseline["never_smoked"] = (all_data_baseline['SmokingStatus'] == 'Never smoked').astype(int)
    all_data_baseline["ex_smoker"] = (all_data_baseline['SmokingStatus'] == 'Ex-smoker').astype(int)
    all_data_baseline["current_smoker"] = (all_data_baseline['SmokingStatus'] == 'Currently smokes').astype(int)
    smoking_order = {'Never smoked': 0, "Ex-smoker": 1, "Currently smokes": 2}
    all_data_baseline['smoking_level'] = all_data_baseline['SmokingStatus'].map(smoking_order).fillna(0) / 2
    # Interaction features
    all_data_baseline['age_time'] = all_data_baseline['age_norm'] * all_data_baseline['weeks_norm']
    all_data_baseline['baseline_time'] = all_data_baseline['baseline_fvc_norm'] * all_data_baseline['weeks_norm']
    all_data_baseline['male_smoker'] = all_data_baseline['is_male'] * all_data_baseline['current_smoker']
    all_data_baseline['age_smoking'] = all_data_baseline['age_norm'] * all_data_baseline['smoking_level']
    # Final feature list
    feature_columns = [
        'age_norm', 'weeks_norm', 'baseline_fvc_norm', 'percent_norm',
        'age_time', 'baseline_time',
        'is_male', 'smoking_level', 'never_smoked', 'ex_smoker', 'current_smoker',
        'male_smoker', 'age_smoking'
    ]
    print(f"Created {len(feature_columns)} features")
    print("Features:", feature_columns)
    # Check for missing values
    missing_check = all_data_baseline[feature_columns].isnull().sum()
    if missing_check.sum() > 0:
        print("Missing values found:")
        print(missing_check[missing_check > 0])
    else:
        print("\nNo missing values in features")
    return all_data_baseline, feature_columns


@app.cell
def _(all_data_baseline, feature_columns):
    # Split data back
    train_data = all_data_baseline[all_data_baseline['data_source'] == 'train'].copy()
    submission_pred_data = all_data_baseline[all_data_baseline['data_source'] == 'submission'].copy()
    # Training arrays
    X = train_data[feature_columns].values
    y = train_data['FVC'].values
    # Test arrays
    X_test = submission_pred_data[feature_columns].values
    print(f"Training data: X={X.shape}, y={y.shape}")
    print(f"Test data: X_test={X_test.shape}")
    print(f"Target range: [{y.min():.1f}, {y.max():.1f}]")
    return X, X_test, train_data, y


@app.cell
def _(competition_metric, tf):
    # Config
    QUANTILES = [0.2, 0.5, 0.8]
    N_FOLDS = 5
    EPOCHS = 300
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    # Loss FUnctions
    def pinball_loss(y_true, y_pred):
        losses = []
        for i, q in enumerate(QUANTILES):
            error = y_true - y_pred[:, i:i+1]
            loss_q = tf.maximum(q * error, (q - 1)* error)
            losses.append(loss_q)
        return tf.reduce_mean(tf.concat(losses, axis=1))
    def kaggle_metric(y_true, y_pred):
        y_true = tf.reshape(tf.cast(y_true, tf.float32), (-1,))
        median = y_pred[:, 1]
        sigma = y_pred[:, 2] - y_pred[:, 0]
        sigma = tf.maximum(sigma, 70.0)
        delta = tf.abs(y_true - median)
        delta = tf.minimum(delta, 1000.0)
        sqrt2 = tf.sqrt(tf.constant(2.0, tf.float32))
        return -tf.reduce_mean((delta / sigma) * sqrt2 + tf.math.log(sigma * sqrt2))
    # Combined Loss (80% 20 % split)
    def combined_loss(y_true, y_pred):
        return 0.8 * pinball_loss(y_true, y_pred) + 0.2 * competition_metric(y_true, y_pred)
    return (
        BATCH_SIZE,
        EPOCHS,
        LEARNING_RATE,
        N_FOLDS,
        QUANTILES,
        kaggle_metric,
        pinball_loss,
    )


@app.cell
def _(
    BATCH_SIZE,
    EPOCHS,
    GroupKFold,
    LEARNING_RATE,
    N_FOLDS,
    QUANTILES,
    X,
    X_test,
    kaggle_metric,
    mean_absolute_error,
    np,
    pinball_loss,
    tf,
    train_data,
    y,
):
    # Initialize cross-validation
    groups = train_data['Patient'].values
    kfold = GroupKFold(n_splits=N_FOLDS)
    oof_predictions = np.zeros((X.shape[0], len(QUANTILES)))
    test_predictions = np.zeros((X_test.shape[0], len(QUANTILES)))
    fold_scores_mae = []
    fold_scores_competition = []  # Add competition metric tracking

    # Training loop
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y, groups=groups)):
        print(f"\n" + "="*50)
        print(f"FOLD {fold + 1}/{N_FOLDS}")
        print("="*50)
    
        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
    
        # Build model
        inputs = tf.keras.layers.Input(shape=(X.shape[1],))
    
        # Input processing
        x = tf.keras.layers.BatchNormalization()(inputs)
        x = tf.keras.layers.GaussianNoise(0.1)(x)
    
        # Hidden layers
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
    
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
    
        x = tf.keras.layers.Dense(32, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
    
        # Quantile outputs
        base_pred = tf.keras.layers.Dense(len(QUANTILES), activation='linear')(x)
        positive_inc = tf.keras.layers.Dense(len(QUANTILES), activation='softplus')(x)
    
        # Ensure monotonic ordering
        ordered_quantiles = tf.keras.layers.Lambda(
            lambda x: x[0] + tf.math.cumsum(x[1], axis=1)
        )([base_pred, positive_inc])
    
        model = tf.keras.Model(inputs, ordered_quantiles)
    
        # Compile model
        optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        model.compile(optimizer=optimizer, loss=pinball_loss, metrics=[kaggle_metric])
    
        # Callbacks
        ckpt_path = f"qr_fold_{fold+1}.keras"
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
           ckpt_path, monitor='val_kaggle_metric', mode='max',
           save_best_only=True, save_weights_only=False, verbose=1
       )
        early_stop = tf.keras.callbacks.EarlyStopping(
           monitor='val_kaggle_metric', mode='max', patience=30,
           restore_best_weights=True, verbose=1
       )
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
           monitor='val_kaggle_metric', mode='max', factor=0.5,
           patience=15, min_lr=1e-7, verbose=1
       ) 
    
        # Train
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[checkpoint, early_stop, reduce_lr],
            verbose=1
        )
    
        # Predict
        val_pred = model.predict(X_val, verbose=0)
        test_pred = model.predict(X_test, verbose=0)
    
        # Store predictions
        oof_predictions[val_idx] = val_pred
        test_predictions += test_pred / N_FOLDS
    
        # Calculate BOTH metrics for comparison
        fold_mae = mean_absolute_error(y_val, val_pred[:, 1])
    
        # Calculate competition metric (numpy version)
        sigma = val_pred[:, 2] - val_pred[:, 0]
        sigma_clipped = np.maximum(sigma, 70.0)
        delta = np.abs(y_val - val_pred[:, 1])
        delta_clipped = np.minimum(delta, 1000.0)
        sqrt_2 = np.sqrt(2.0)
        fold_comp = np.mean((delta_clipped / sigma_clipped) * sqrt_2 + np.log(sigma_clipped * sqrt_2))
    
        fold_scores_mae.append(fold_mae)
        fold_scores_competition.append(fold_comp)
    
        print(f"Fold {fold + 1} MAE: {fold_mae:.4f}")
        print(f"Fold {fold + 1} Competition Metric: {fold_comp:.4f}")
    
        # Clean up
        del model
        tf.keras.backend.clear_session()

    # Calculate overall scores
    overall_mae = mean_absolute_error(y, oof_predictions[:, 1])

    sigma_oof = oof_predictions[:, 2] - oof_predictions[:, 0]
    sigma_clipped_oof = np.maximum(sigma_oof, 70.0)
    delta_oof = np.abs(y - oof_predictions[:, 1])
    delta_clipped_oof = np.minimum(delta_oof, 1000.0)
    overall_competition = np.mean((delta_clipped_oof / sigma_clipped_oof) * np.sqrt(2.0) + np.log(sigma_clipped_oof * np.sqrt(2.0)))

    print(f"\n" + "="*50)
    print("CROSS-VALIDATION RESULTS")
    print("="*50)

    print(f"\nMAE Scores (for reference):")
    print(f"Fold scores: {[f'{s:.4f}' for s in fold_scores_mae]}")
    print(f"Mean: {np.mean(fold_scores_mae):.4f} ± {np.std(fold_scores_mae):.4f}")
    print(f"Overall OOF: {overall_mae:.4f}")

    print(f"\n COMPETITION METRIC:")
    print(f"Fold scores: {[f'{s:.4f}' for s in fold_scores_competition]}")
    print(f"Mean: {np.mean(fold_scores_competition):.4f} ± {np.std(fold_scores_competition):.4f}")
    print(f"Overall OOF: {overall_competition:.4f}")

    print(f"\nExpected Leaderboard Score: ~{overall_competition:.4f}")
    return


if __name__ == "__main__":
    app.run()
