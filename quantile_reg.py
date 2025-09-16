import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    # Packages
    import pandas as pd
    import tensorflow as tf
    return pd, tf


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

    return


@app.cell
def _(tf):
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

    def competition_metric(y_true, y_pred):
        median_pred = y_pred[:, 1] 
        confidence_width = y_pred[:, 2] - y_pred[:, 0]

        confidence_clipped = tf.maximum(confidence_width, 70.0)
        error = tf.abs(y_true[:, 0] - median_pred)
        error_clipped = tf.minimum(error, 1000.0)

        sqrt_2 = tf.sqrt(2.0)
        metric = (error_clipped / confidence_clipped) * sqrt_2 + tf.math.log(confidence_clipped * sqrt_2)
        return tf.reduce_mean(metric)
    # Combined Loss (80% 20 % split)
    def combined_loss(y_true, y_pred):
        return 0.8 * pinball_loss(y_true, y_pred) + 0.2 * competition_metric(y_true, y_pred)
    
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
