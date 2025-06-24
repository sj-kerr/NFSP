# XGB best model with adding in other seals that do not have video cameras equipped #

import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight


## merge all pups ##
# dfA = pd.read_csv('A_f.csv')
# dfB = pd.read_csv('B_f.csv')
# dfC = pd.read_csv('C_f.csv')
# dfD = pd.read_csv('D_f.csv')
# dfF = pd.read_csv('F_f.csv')
# dfG = pd.read_csv('G_f.csv')
# dfJ = pd.read_csv('J_f.csv')
# dfK = pd.read_csv('K_f.csv')
# dfL = pd.read_csv('L_f.csv')
# dfM = pd.read_csv('M_f.csv')
# dfN = pd.read_csv('N_f.csv')
# dfO = pd.read_csv('O_f.csv')
# dfP = pd.read_csv('P_f.csv')
# dfR = pd.read_csv('R_f.csv')
# dfS = pd.read_csv('S_f.csv')
# dfT = pd.read_csv('T_f.csv')
# dfU = pd.read_csv('U_f.csv')
# dfV = pd.read_csv('V_f.csv')
# dfW = pd.read_csv('W_f.csv')
# dfX = pd.read_csv('X_f.csv')
# dfY = pd.read_csv('Y_f.csv')
# dfZ = pd.read_csv('Z_f.csv')
# dfAA = pd.read_csv('AA_f.csv')
# dfBB = pd.read_csv('BB_f.csv')
# dfCC = pd.read_csv('CC_f.csv')
# dfDD = pd.read_csv('DD_f.csv')
# dfEE = pd.read_csv('EE_f.csv')
# dfFF = pd.read_csv('FF_f.csv')
# dfGG = pd.read_csv('GG_f.csv')
# dfJJ = pd.read_csv('JJ_f.csv')
# dfKK = pd.read_csv('KK_f.csv')
# dfLL = pd.read_csv('LL_f.csv')
# dfMM = pd.read_csv('MM_f.csv')
# dfNN = pd.read_csv('NN_f.csv')
# dfOO = pd.read_csv('OO_f.csv')
# dfPP = pd.read_csv('PP_f.csv')
# dfRR = pd.read_csv('RR_f.csv')
# dfSS = pd.read_csv('SS_f.csv')
# dfTT = pd.read_csv('TT_f.csv')
# dfUU = pd.read_csv('UU_f.csv')
# dfVV = pd.read_csv('VV_f.csv')
# dfWW = pd.read_csv('WW_f.csv')
# dfXX = pd.read_csv('XX_f.csv')
# dfYY = pd.read_csv('YY_f.csv')
# dfZZ = pd.read_csv('ZZ_f.csv')
# dfSF = pd.read_csv('SF_f.csv')
# dfLB = pd.read_csv('LB_f.csv')
# dfDS = pd.read_csv('DS_f.csv')

# dfstd = [dfA, dfB, dfC, dfD, dfF, dfG, dfJ, dfK, dfL, dfM, dfN, dfO, dfP, dfR, dfS, dfT, dfU, dfV, dfW, dfX, dfY, dfZ, 
#           dfAA, dfBB, dfCC, dfDD, dfEE, dfFF, dfGG, dfJJ, dfKK, dfLL, dfMM, dfNN, dfOO, dfPP, dfRR, dfSS, dfTT, 
#           dfUU, dfVV, dfWW, dfXX, dfYY, dfZZ, dfSF, dfLB, dfDS]
# dfs = pd.concat(dfstd, ignore_index=True)

dfs = pd.read_csv('Final_comp_fixed.csv') #change for not compressed or for adaptive

dfs = dfs.drop(['Timestamp'],axis=1)

print(dfs['Behavior'].value_counts().sort_index())

#save GMT to reinsert later
#original_time = dfs['Behavior_UTC'].copy()


# Replace values in 'Behavior' column - make better behavior categories
# Group behaviors 3, 5, and 7 into 2 (active)
dfs['Behavior'] = dfs['Behavior'].replace({3: 2, 5: 2, 7: 2})

# Remove behavior 6 (shaking)
dfs = dfs[dfs['Behavior'] != 6]

# Now remap 0, 1, 2, 4 into final labels
rename_map = {
    0: 0,  # sleeping
    1: 1,  # nursing
    2: 2,  # active (includes original 2, 3, 5, 7)
    4: 3   # inactive (originally 4 → now 3)
}

dfs['Behavior'] = dfs['Behavior'].map(rename_map)

# Verify the changes
print("\nBehavior counts after modification:")
print(dfs['Behavior'].value_counts().sort_index())


## Begin Model ##

# Assuming dfs is your DataFrame
X = dfs.drop('Behavior', axis=1)
y = dfs['Behavior']

# Split into train+validation and test sets
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Split train+validation into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val)

# Compute sample weights
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

# Best Params found from randomsearch in pup_xgb_randomsearch.py: 
# {'xgb__subsample': 0.9, 'xgb__n_estimators': 300, 'xgb__min_child_weight': 1, 'xgb__max_depth': 7, 'xgb__learning_rate': 0.1, 'xgb__colsample_bytree': 1.0}

param_grid = {
    'xgb__n_estimators': [300], #100
    'xgb__max_depth': [7], #7
    'xgb__learning_rate': [0.2], #0.2
    'xgb__subsample': [0.9], #0.6
    'xgb__colsample_bytree': [1.0], #0.6
    'xgb__min_child_weight': [1] #3
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def train_evaluate_model(X_train, y_train, X_val, y_val, X_test, y_test, param_grid, cv):
    pipeline = Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('xgb', XGBClassifier(objective='multi:softmax', num_class=len(np.unique(y)), random_state=42))
    ])
    
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring='f1_weighted', cv=cv, verbose=1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    
    # Evaluate on validation set
    y_val_pred = best_model.predict(X_val)
    print("\nValidation Set Performance:")
    print("Accuracy:", accuracy_score(y_val, y_val_pred))
    print("Macro F1-Score:", f1_score(y_val, y_val_pred, average='macro'))
    
    # Evaluate on test set
    y_test_pred = best_model.predict(X_test)
    y_test_pred_proba = best_model.predict_proba(X_test)
    
    print("\nTest Set Performance:")
    print("Accuracy:", accuracy_score(y_test, y_test_pred))
    print("Macro F1-Score:", f1_score(y_test, y_test_pred, average='macro'))
    print("Weighted F1-Score:", f1_score(y_test, y_test_pred, average='weighted'))
    print("ROC-AUC:", roc_auc_score(y_test, y_test_pred_proba, multi_class='ovr', average='macro'))
    print('Classification Report:\n', classification_report(y_test, y_test_pred))
    print('Confusion Matrix:\n', confusion_matrix(y_test, y_test_pred))
    
    feature_importance = pd.Series(best_model.named_steps['xgb'].feature_importances_, index=X_train.columns).sort_values(ascending=False)
    print("\nFeature Importance:")
    print(feature_importance)

    return best_model, feature_importance

# Train initial model
best_model, feature_importance = train_evaluate_model(X_train, y_train, X_val, y_val, X_test, y_test, param_grid, cv)

# Identify the bottom 50 features
bottom_50_features = feature_importance.tail(50).index #drop any amount of bottom feautures
#tried dropping 40 and 70 and 50 - 50 was best.
print(bottom_50_features)
# Select all features except the bottom 50
selected_features = feature_importance.drop(bottom_50_features).index
selected_features = list(selected_features)

# Train model with selected features
X_train_fs = X_train[selected_features]
X_val_fs = X_val[selected_features]
X_test_fs = X_test[selected_features]

best_model_fs, feature_importance_fs = train_evaluate_model(X_train_fs, y_train, X_val_fs, y_val, X_test_fs, y_test, param_grid, cv)


# Cross-validation with best parameters
# Function to perform cross-validation and print classification report
def perform_cv_and_classification_report(X_train, y_train, X_val, y_val, param_grid, cv):
    pipeline = Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('xgb', XGBClassifier(objective='multi:softmax', num_class=len(np.unique(y_train)), random_state=42))
    ])

    # Cross-validation scores
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='f1_weighted')
    print(f'Cross-validation scores: {scores}')
    print(f'Mean cross-validation score: {np.mean(scores)}')

    # Train on entire training set and predict on validation set
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_val)

    # Classification report
    report = classification_report(y_val, y_pred)
    print("\nClassification Report:")
    print(report)

print("\nXGBoost with SMOTE:")
perform_cv_and_classification_report(X_train, y_train, X_val, y_val, param_grid, cv)

print("\nXGBoost with SMOTE and reduced features:")
perform_cv_and_classification_report(X_train_fs, y_train, X_val_fs, y_val, param_grid, cv)

######################################################
# Function to train and evaluate a model without SMOTE
######################################################

def train_evaluate_model_no_smote(X_train, y_train, X_val, y_val, X_test, y_test, param_grid, cv):
    # Create XGBoost classifier without SMOTE
    xgb_params = {
        'n_estimators': param_grid['xgb__n_estimators'][0],
        'max_depth': param_grid['xgb__max_depth'][0],
        'learning_rate': param_grid['xgb__learning_rate'][0],
        'subsample': param_grid['xgb__subsample'][0],
        'colsample_bytree': param_grid['xgb__colsample_bytree'][0],
        'min_child_weight': param_grid['xgb__min_child_weight'][0],
        'objective': 'multi:softmax',
        'num_class': len(np.unique(y)),
        'random_state': 42
    }
    
    model = XGBClassifier(**xgb_params)
    
    # Fit model with sample weights
    model.fit(X_train, y_train, sample_weight=sample_weights)
    
    # Evaluate on validation set
    y_val_pred = model.predict(X_val)
    print("\nValidation Set Performance (Regular XGB):")
    print("Accuracy:", accuracy_score(y_val, y_val_pred))
    print("Macro F1-Score:", f1_score(y_val, y_val_pred, average='macro'))
    
    # Evaluate on test set
    y_test_pred = model.predict(X_test)
    y_test_pred_proba = model.predict_proba(X_test)
    
    print("\nTest Set Performance (Regular XGB):")
    print("Accuracy:", accuracy_score(y_test, y_test_pred))
    print("Macro F1-Score:", f1_score(y_test, y_test_pred, average='macro'))
    print("Weighted F1-Score:", f1_score(y_test, y_test_pred, average='weighted'))
    print("ROC-AUC:", roc_auc_score(y_test, y_test_pred_proba, multi_class='ovr', average='macro'))
    print('Classification Report:\n', classification_report(y_test, y_test_pred))
    print('Confusion Matrix:\n', confusion_matrix(y_test, y_test_pred))
    
    feature_importance = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    print("\nFeature Importance (Regular XGB):")
    print(feature_importance)

    return model, feature_importance

# Function for cross-validation without SMOTE
def perform_cv_and_classification_report_no_smote(X_train, y_train, X_val, y_val, param_grid, cv):
    # Create XGBoost classifier with the same parameters
    xgb_params = {
        'n_estimators': param_grid['xgb__n_estimators'][0],
        'max_depth': param_grid['xgb__max_depth'][0],
        'learning_rate': param_grid['xgb__learning_rate'][0],
        'subsample': param_grid['xgb__subsample'][0],
        'colsample_bytree': param_grid['xgb__colsample_bytree'][0],
        'min_child_weight': param_grid['xgb__min_child_weight'][0],
        'objective': 'multi:softmax',
        'num_class': len(np.unique(y_train)),
        'random_state': 42
    }
    
    model = XGBClassifier(**xgb_params)
    
    # Cross-validation scores
    cv_scores = []
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_train_cv, X_val_cv = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_cv, y_val_cv = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # Compute sample weights for this fold
        train_weights = compute_sample_weight(class_weight='balanced', y=y_train_cv)
        
        model.fit(X_train_cv, y_train_cv, sample_weight=train_weights)
        y_pred = model.predict(X_val_cv)
        score = f1_score(y_val_cv, y_pred, average='weighted')
        cv_scores.append(score)
    
    print(f'Cross-validation scores: {cv_scores}')
    print(f'Mean cross-validation score: {np.mean(cv_scores)}')

    # Train on entire training set and predict on validation set
    model.fit(X_train, y_train, sample_weight=sample_weights)
    y_pred = model.predict(X_val)

    # Classification report
    report = classification_report(y_val, y_pred)
    print("\nClassification Report (Regular XGB):")
    print(report)

# Run additional model variants for Regular XGBoost

# 3. Regular XGBoost without SMOTE
print("\n3. Regular XGBoost without SMOTE:")
best_model_no_smote, feature_importance_no_smote = train_evaluate_model_no_smote(X_train, y_train, X_val, y_val, X_test, y_test, param_grid, cv)

# 4. Regular XGBoost with reduced features but without SMOTE
print("\n4. Regular XGBoost with reduced features but without SMOTE:")
best_model_fs_no_smote, feature_importance_fs_no_smote = train_evaluate_model_no_smote(X_train_fs, y_train, X_val_fs, y_val, X_test_fs, y_test, param_grid, cv)

# Cross-validation for regular XGBoost variants
print("\nCross-validation results for Regular XGBoost without SMOTE:")
perform_cv_and_classification_report_no_smote(X_train, y_train, X_val, y_val, param_grid, cv)

print("\nCross-validation results for Regular XGBoost with reduced features but without SMOTE:")
perform_cv_and_classification_report_no_smote(X_train_fs, y_train, X_val_fs, y_val, param_grid, cv)

### SMOTE XGB without reduced features and 1 sec rolling means in 3 second windows with 4 classes yeilds best right now! ###
# note, did not do another grid search for this, used best features from before...need to run grid search again #

########################################################
### SAVE model to predict behaviors from raw dataframe LATER ###
########################################################

# First, train your model as you normally would
# model = run_model_pipeline(...)

# Import libraries for saving/loading
import pickle
import joblib
import os

# ----- SAVING THE MODEL -----

def save_model(model, feature_names, filename='xgboost_model.pkl', use_joblib=False):
    """
    Save the trained model along with selected feature names
    
    Parameters:
    -----------
    model : trained model
        The fitted model to save
    feature_names : list
        List of feature names used for training
    filename : str, default='xgboost_model_with_features.pkl'
        Name of the file to save the model to
    use_joblib : bool, default=False
        Whether to use joblib instead of pickle (better for large models)
    """

    # Create a dictionary with model and features
    model_data = {
        'model': model,
        'feature_names': feature_names
    }
    
    if use_joblib:
        joblib.dump(model_data, filename)
        print(f"Model and features saved to {filename} using joblib")
    else:
        with open(filename, 'wb') as file:
            pickle.dump(model_data, file)
        print(f"Model and features saved to {filename} using pickle")

# Example usage:
# Assuming selected_features contains your list of feature names
save_model(best_model_fs_no_smote, selected_features, filename='xgboost_model.pkl', use_joblib=False)
    
# If you're using the full pipeline (including SMOTE and fs), get just the XGB step
#xgb_model = best_model_fs.named_steps['xgb'] # If you choose a different model than best_model_fs, make sure to adjust your code accordingly.

#Model #	Model Variant	Feature Set Used
#1	best_model	All features (smote)
#2	best_model_fs	Reduced features (smote)
#3	best_model_no_smote	All features (no smote)
#4	best_model_fs_no_smote	Reduced features (no smote)


########################################################
### Use model to predict behaviors from raw dataframe###
######### PREPROCESS IN R SCRIPT FIRST ###########
########################################################
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
import pickle
import joblib
import os

## load all pups ##

dfA = pd.read_csv('A_0000_S1_final_comp.csv') 
dfC = pd.read_csv('C_0000_S2_final_comp.csv')
dfE = pd.read_csv('E_0000_S2_final_comp.csv')
dfF = pd.read_csv('F_0000_S2_final_comp.csv')
dfJ = pd.read_csv('J_0000_S1_final_comp.csv')
dfM = pd.read_csv('M_0000_S1_final_comp.csv')
dfN = pd.read_csv('N_0000_S1_final_comp.csv')
dfO = pd.read_csv('O_0000_S2_final_comp.csv')
dfP = pd.read_csv('P_0000_S2_final_comp.csv')
dfT = pd.read_csv('T_0000_S1_final_comp.csv')
dfV = pd.read_csv('V_0000_S2_final_comp.csv')
dfW = pd.read_csv('W_0000_S2_final_comp.csv')
dfY = pd.read_csv('Y_0000_S1_final_comp.csv')
dfZ = pd.read_csv('Z_0000_S1_final_comp.csv')
dfBB = pd.read_csv('BB_0000_S1_final_comp.csv')
dfCC = pd.read_csv('CC_0000_S1_final_comp.csv')
dfGG = pd.read_csv('GG_0000_S1_final_comp.csv')
dfJJ = pd.read_csv('JJ_0000_S1_final_comp.csv')
dfLL = pd.read_csv('LL_0000_S1_final_comp.csv')
dfRR = pd.read_csv('RR_0000_S1_final_comp.csv')
dfSS = pd.read_csv('SS_0000_S1_final_comp.csv')
dfTT = pd.read_csv('TT_0000_Short_final_comp.csv')
dfUU = pd.read_csv('UU_0000_S1_final_comp.csv')
dfUU2 = pd.read_csv('UU_0000_S2_final_comp.csv')
dfWW = pd.read_csv('WW_0000_S1_final_comp.csv')
dfXX = pd.read_csv('XX_0000_S1_final_comp.csv')
dfYY = pd.read_csv('YY_0000_S1_final_comp.csv')
dfZZ = pd.read_csv('ZZ_0000_S1_final_comp.csv')
dfSF = pd.read_csv('SF_0000_S1_final_comp.csv')
dfLB = pd.read_csv('LB_0000_S1_final_comp.csv')
dfDS = pd.read_csv('$_0000_S2_final_comp.csv')

dfstd = [dfA, dfC, dfE, dfF, dfJ, dfM, dfN, dfO, dfP, dfT, dfV, dfW, dfY, dfZ, 
           dfBB, dfCC, dfGG, dfJJ, dfLL, dfRR, dfSS, dfTT, 
             dfUU, dfUU2, dfWW, dfXX, dfYY, dfZZ, dfSF, dfLB, dfDS]
dfs = pd.concat(dfstd, ignore_index=True)

dfs.to_csv("Allpups_comp.csv", index=False)

# ----- LOADING THE MODEL -----
def load_model(filename='xgboost_model.pkl', use_joblib=False):
    """
    Load a trained model from a file
    
    Parameters:
    -----------
    filename : str, default='xgboost_model.pkl'
        Name of the file to load the model from
    use_joblib : bool, default=False
        Whether to use joblib instead of pickle
        
    Returns:
    --------
    model : trained model
        The loaded model
    """
    if use_joblib:
        model_data = joblib.load(filename)
        print(f"Model loaded from {filename} using joblib")
    else:
        with open(filename, 'rb') as file:
            model_data = pickle.load(file)
        print(f"Model loaded from {filename} using pickle")
    
    return model_data['model'], model_data['feature_names']


loaded_model, selected_features = load_model('xgboost_model.pkl')

# read in what F_comp was saved as : FinalBB_comp_fixed_raw.csv
#this is the unseen data

def predict_with_loaded_model(model, new_data, feature_names, return_proba=False):
    """
    Use a loaded model to make predictions on new data
    
    Parameters:
    -----------
    model : trained model
        The loaded model
    new_data : DataFrame
        New data to predict on
    feature_names : list, optional
        Feature names if needed for ordering columns
    return_proba : bool, default=False
        Whether to return class probabilities instead of labels
        
    Returns:
    --------
    predictions : array
        The predicted labels or probabilities
    """
    
    # Select and reorder columns to match training data
    new_data = new_data[feature_names]

    # Make predictions
    if return_proba:
        return model.predict_proba(new_data)
    else:
        return model.predict(new_data)

def apply_behavior_filters(df, behavior_column='Filtered_Predicted_Behavior'):
    """
    Apply behavior-specific post-processing:
    1. Apply swimming override first (based on ADC)
    2. Apply streak + optional gap-filling for other behaviors
    3. Map final labels
    """
    def smooth_behavior_gaps(binary_array, max_gap=2):
        arr = binary_array.values if isinstance(binary_array, pd.Series) else binary_array
        result = arr.copy()
        i = 0
        while i < len(arr):
            if arr[i] == 0:
                start = i
                while i < len(arr) and arr[i] == 0:
                    i += 1
                gap_length = i - start
                if start > 0 and i < len(arr) and gap_length <= max_gap:
                    result[start:i] = 1
            else:
                i += 1
        return result

    # === 1. Apply swimming override based on conductivity ===
    conductivity_threshold = 20
    swim_binary = (df['ADC..raw.'] < conductivity_threshold).astype(int)
    swim_smoothed = smooth_behavior_gaps(swim_binary, max_gap=5)
    df.loc[swim_smoothed == 1, behavior_column] = 4  # Assign Swimming

    # === 2. Apply streak+gap filtering to all NON-swimming classes ===
    behavior_rules = {
        0: {'name': 'Sleeping',  'streak': 6, 'gap': 3},
        1: {'name': 'Nursing',   'streak': 10, 'gap': 2},
        2: {'name': 'Active',    'streak': 2, 'gap': None},
        3: {'name': 'Inactive',  'streak': 1, 'gap': 3}
    }

    for class_id, params in behavior_rules.items():
        # Only filter rows that aren't already Swimming
        current_mask = (df[behavior_column] == class_id)
        binary = current_mask.astype(int)

        # Identify consecutive sequences
        groups = (binary != np.roll(binary, 1)).cumsum()
        group_sizes = pd.Series(groups).map(pd.Series(binary).groupby(groups).sum())
        valid_streak = (binary == 1) & (group_sizes >= params['streak'])

        if params.get('gap') is not None:
            smoothed = smooth_behavior_gaps(valid_streak.astype(int), max_gap=params['gap'])
        else:
            smoothed = valid_streak.astype(int)

        # Remove all existing predictions for this behavior (but leave Swimming untouched)
        df.loc[current_mask, behavior_column] = -1

        # Restore only valid predictions
        df.loc[smoothed == 1, behavior_column] = class_id

    # === 3. Map readable labels ===
    behavior_map = {
        0: 'Sleeping',
        1: 'Nursing',
        2: 'Active',
        3: 'Inactive',
        4: 'Swimming',
        -1: 'Uncertain'
    }
    df['Behavior_Label'] = df[behavior_column].map(behavior_map)

    return df

#Raw = dfP
Raw = pd.read_csv("JJ_0000_S1_final_comp.csv") #change to all accels merged!

#import polars as pl
#Raw = pl.read_csv("Allpups_comp.csv")

#Raw = pd.read_csv("Allpups_comp.csv") 

# Save the Timestamp and conductivity column
timestamps = Raw['Timestamp'].copy()  # clone is polars - copy is pandas
conductivity = Raw['ADC..raw.'].copy()
id = Raw['Tag.ID'].copy()
#conductivity_threshold = 20  # Adjust threshold for "wet" ~300 is dry and ~0-20 is wet.

# Get predicted class probabilities
probas = predict_with_loaded_model(loaded_model, Raw, selected_features, return_proba=True)

# To get the max probability associated with each predicted behavior:
max_proba = probas.max(axis=1)

# You can also get predicted classes explicitly:
predicted_classes = probas.argmax(axis=1)

# Combine everything
output = pd.DataFrame({
    'ADC..raw.': conductivity,
    'Timestamp': timestamps,
    'Predicted_Behavior': predicted_classes,
    'Confidence': max_proba,
    'Tag.ID': id
})

# Apply confidence threshold to all predictions

# Only keep behavior predictions if confidence > 0.8
threshold = 0.8
output['Confident_Prediction'] = np.where(output['Confidence'] > threshold, output['Predicted_Behavior'], -1)

# Initialize Filtered_Predicted_Behavior with all confident predictions
output['Filtered_Predicted_Behavior'] = output['Confident_Prediction']

output = apply_behavior_filters(output, behavior_column='Filtered_Predicted_Behavior')

# === Binary nursing column for plotting ===
output['Nursing_Flag'] = (output['Filtered_Predicted_Behavior'] == 1).astype(int)
output['Filtered_Predicted_Behavior'].value_counts().sort_index()
output['Nursing_Flag'].value_counts().sort_index()

#output.to_csv("RawF_predictions_with_probs.csv", index=False) 
output.to_csv("JJ_predictions_final.csv", index=False)
output.to_csv("allpups_predictions_final.csv", index=False)

# (Optional) Merge other columns back in if desired
#output2 = pd.concat([Raw.drop(columns=selected_features), output[['Predicted_Behavior']]], axis=1)

# Save to CSV
#output2.to_csv("RawBB_fulldata_predictions.csv", index=False) # change name


# --- Plot timeline for a single week or two (Nursing) ---

# Convert Timestamp column to datetime if it's not already
output['Timestamp'] = pd.to_datetime(output['Timestamp'])

# Define start and end times for the week you want 
## USE THIS FOR SINGLE PUPS - WOULDNT BE FOR WHOLE GROUP
start_time = pd.to_datetime('2024-10-01')  # change this to your desired start
end_time = start_time + pd.Timedelta(days=20)  

# Subset the output DataFrame to only that week
week_data = output[(output['Timestamp'] >= start_time) & (output['Timestamp'] < end_time)]


plt.figure(figsize=(20, 4))
nursing_binary_smoothed = (week_data['Nursing_Flag'] == 1).astype(int)
plt.fill_between(week_data['Timestamp'], nursing_binary_smoothed, step='post', alpha=1.0)
plt.ylim(0, 1.1)
plt.xlabel('Time')
plt.ylabel('Nursing')
plt.title('Filtered High-Confidence Nursing Over One Week')
plt.tight_layout()
plt.savefig("nursing_filtered_plot_JJ.png")
print("✅ Plot saved: nursing_filtered_plot_2 week.png")

## darker plot
plt.figure(figsize=(20, 4))
plt.fill_between(week_data['Timestamp'], nursing_binary_smoothed, step='post', alpha=1.0, color='navy')
plt.ylim(0, 1.1)
plt.xlabel('Time')
plt.ylabel('Nursing')
plt.title('Filtered High-Confidence Nursing Over Two Weeks')
plt.tight_layout()
plt.savefig("nursing_filtered_plot_week_dark_P.png")
print("✅ Darker plot saved: nursing_filtered_plot_week_dark.png")

## include maternal dates of arrival/departure ##

output['Timestamp'] = pd.to_datetime(output['Timestamp'])
start_time = pd.to_datetime('2024-09-20')
end_time = start_time + pd.Timedelta(days=30)

# Subset the output DataFrame to only that period
week_data = output[(output['Timestamp'] >= start_time) & (output['Timestamp'] < end_time)]

## include maternal dates of arrival/departure ##
plt.figure(figsize=(20, 4))

# Create a boolean mask for when mother is present
mother_present = np.zeros(len(week_data), dtype=bool)

# Define mother's presence periods based on your arrival/departure times
presence_periods = [
    ('2024-09-21 00:55:00', '2024-09-23 05:50:00'),  # First visit
    ('2024-10-01 15:55:00', '2024-10-03 04:05:00'),  # Second visit
    ('2024-10-11 21:05:00', '2024-10-13 03:00:00'),  # Third visit
    ('2024-10-21 05:55:00', '2024-10-23 23:59:59'),  # Assuming she stays through end of period
]

# Mark periods when mother is present
for arrival, departure in presence_periods:
    mask = (week_data['Timestamp'] >= arrival) & (week_data['Timestamp'] <= departure)
    mother_present |= mask

nursing_binary_smoothed = (week_data['Nursing_Flag'] == 1).astype(int)

# Plot nursing events when mother is ABSENT (light, low alpha)
absent_nursing = nursing_binary_smoothed.copy()
absent_nursing[mother_present] = 0  # Hide nursing events when present
plt.fill_between(week_data['Timestamp'], absent_nursing, step='post', 
                 alpha=0.4, color='lightblue')

# Plot nursing events when mother is PRESENT (dark, high alpha)
present_nursing = nursing_binary_smoothed.copy()
present_nursing[~mother_present] = 0  # Hide nursing events when absent
plt.fill_between(week_data['Timestamp'], present_nursing, step='post', 
                 alpha=1.0, color='steelblue')

# Your original arrival/departure data
mother_arrivals = [
    '2024-09-21 00:55:00',
    '2024-10-01 15:55:00',      
    '2024-10-11 21:05:00',
    '2024-10-21 05:55:00',
]

mother_departures = [
    '2024-09-23 05:50:00',     
    '2024-10-03 04:05:00',
    '2024-10-13 03:00:00',
]

# Convert to datetime
mother_arrivals = pd.to_datetime(mother_arrivals)
mother_departures = pd.to_datetime(mother_departures)

# Plot arrival lines (green)
for arrival in mother_arrivals:
    plt.axvline(x=arrival, color='green', linestyle='--', alpha=0.7, linewidth=2)

# Plot departure lines (red)
for departure in mother_departures:
    plt.axvline(x=departure, color='red', linestyle='--', alpha=0.7, linewidth=2)

plt.ylim(0, 1.1)
plt.xlabel('Time')
plt.ylabel('Nursing')
plt.title('Filtered High-Confidence Nursing Over 29 Days - Mother Presence')
plt.tight_layout()
plt.savefig("nursing_with_mother_presenceJJ.png", dpi=300)


### Calculate percent of false positives ##

# Use the same presence periods and mother_present mask from your plotting code
mother_present = np.zeros(len(week_data), dtype=bool)

presence_periods = [
    ('2024-09-28 00:55:00', '2024-09-30 05:50:00'),  # First visit
    ('2024-10-07 15:55:00', '2024-10-09 04:05:00'),  # Second visit
    ('2024-10-14 21:05:00', '2024-10-17 03:00:00'),  # Third visit
    ('2024-10-25 05:55:00', '2024-10-27 23:59:59'),  # Last visit
]

# Mark periods when mother is present
for arrival, departure in presence_periods:
    mask = (week_data['Timestamp'] >= arrival) & (week_data['Timestamp'] <= departure)
    mother_present |= mask

# Calculate nursing events
nursing_events = (week_data['Nursing_Flag'] == 1)

# True positives: nursing when mother is present
true_positives = nursing_events & mother_present

# False positives: nursing when mother is absent
false_positives = nursing_events & ~mother_present

# Calculate counts
total_nursing_events = nursing_events.sum()
true_positive_count = true_positives.sum()
false_positive_count = false_positives.sum()

# Calculate percentages
false_positive_percentage = (false_positive_count / total_nursing_events) * 100
true_positive_percentage = (true_positive_count / total_nursing_events) * 100

print(f"Total nursing events detected: {total_nursing_events}")
print(f"Nursing events during maternal presence: {true_positive_count} ({true_positive_percentage:.1f}%)")
print(f"False positive events (maternal absence): {false_positive_count} ({false_positive_percentage:.1f}%)")

######### trouble shooting JJ ############### for bad plot ############

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

output = pd.read_csv("P_predictions_final.csv")

output['Timestamp'] = pd.to_datetime(output['Timestamp'])

# First, let's debug your data step by step
print("=== DEBUGGING YOUR NURSING PLOT ===")

# Check your original data
print("\n1. Original data shape:", output.shape)
print("2. Timestamp column type:", output['Timestamp'].dtype)
print("3. Date range in full dataset:")
print("   Min:", output['Timestamp'].min())
print("   Max:", output['Timestamp'].max())

# Check your filtering
start_time = pd.to_datetime('2024-09-28')
end_time = start_time + pd.Timedelta(days=29)
print(f"\n4. Filtering for: {start_time} to {end_time}")

week_data = output[(output['Timestamp'] >= start_time) & (output['Timestamp'] < end_time)]
print(f"5. Filtered data shape: {week_data.shape}")

# Check nursing events
print("\n6. Nursing Flag distribution in full dataset:")
print(output['Nursing_Flag'].value_counts().sort_index())

print("\n7. Nursing Flag distribution in filtered data:")
print(week_data['Nursing_Flag'].value_counts().sort_index())

print("\n8. Filtered Predicted Behavior distribution:")
print(week_data['Filtered_Predicted_Behavior'].value_counts().sort_index())

# Check confidence levels
print(f"\n9. Confidence distribution (> 0.8): {(output['Confidence'] > 0.8).sum()} out of {len(output)}")

# Try a different date range if needed
if len(week_data) == 0 or week_data['Nursing_Flag'].sum() == 0:
    print("\n10. No data in specified range. Trying to find data...")
    # Find actual data range
    actual_start = output['Timestamp'].min()
    actual_end = output['Timestamp'].max()
    print(f"    Actual data range: {actual_start} to {actual_end}")
    
    # Use first 30 days of actual data
    start_time = actual_start
    end_time = start_time + pd.Timedelta(days=30)
    week_data = output[(output['Timestamp'] >= start_time) & (output['Timestamp'] < end_time)]
    print(f"    New filtered data shape: {week_data.shape}")
    print(f"    Nursing events in new range: {week_data['Nursing_Flag'].sum()}")

# Create the plot with debugging
plt.figure(figsize=(20, 6))

if len(week_data) > 0 and week_data['Nursing_Flag'].sum() > 0:
    nursing_binary = (week_data['Nursing_Flag'] == 1).astype(int)
    
    # Plot with better visibility
    plt.fill_between(week_data['Timestamp'], nursing_binary, step='post', 
                     alpha=0.7, color='steelblue', label=f'Nursing events (n={nursing_binary.sum()})')
    
    # Add scatter points for better visibility
    nursing_times = week_data[week_data['Nursing_Flag'] == 1]['Timestamp']
    nursing_values = np.ones(len(nursing_times))
    plt.scatter(nursing_times, nursing_values, color='red', s=10, alpha=0.8, zorder=5)
    
    # Format x-axis properly
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=12))
    
    plt.xticks(rotation=45)
    plt.legend()
    
    title = f'Nursing Events: {start_time.strftime("%Y-%m-%d")} to {end_time.strftime("%Y-%m-%d")}'
else:
    plt.text(0.5, 0.5, 'No nursing events found in date range', 
             transform=plt.gca().transAxes, ha='center', va='center', fontsize=16)
    title = 'No Nursing Data Found'

plt.ylim(0.0, 1.0)
plt.xlabel('Date')
plt.ylabel('Nursing Event')
plt.title(title)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save with better name
plt.savefig("nursing_P_plot.png", dpi=300, bbox_inches='tight')
plt.show()

print("\n=== SUMMARY ===")
print(f"Total data points: {len(output)}")
print(f"Date range: {output['Timestamp'].min()} to {output['Timestamp'].max()}")
print(f"Filtered data points: {len(week_data)}")
print(f"Nursing events in filtered data: {week_data['Nursing_Flag'].sum() if len(week_data) > 0 else 0}")

# Additional debugging - check what behaviors are actually present
if len(week_data) > 0:
    print(f"\nBehavior distribution in filtered data:")
    behavior_counts = week_data['Filtered_Predicted_Behavior'].value_counts().sort_index()
    for behavior, count in behavior_counts.items():
        print(f"  Behavior {behavior}: {count} events")
        
    print(f"\nConfidence distribution in filtered data:")
    print(f"  Mean confidence: {week_data['Confidence'].mean():.3f}")
    print(f"  Min confidence: {week_data['Confidence'].min():.3f}")
    print(f"  Max confidence: {week_data['Confidence'].max():.3f}")
    print(f"  High confidence (>0.8): {(week_data['Confidence'] > 0.8).sum()}")

## add maternal presence

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Check your filtering
start_time = pd.to_datetime('2024-09-28')
end_time = start_time + pd.Timedelta(days=29)


print(f"Plotting data from {start_time} to {end_time}")

# Filter the data
week_data = output[(output['Timestamp'] >= start_time) & (output['Timestamp'] < end_time)].copy()
print(f"Filtered data shape: {week_data.shape}")
print(f"Nursing events found: {week_data['Nursing_Flag'].sum()}")

# Define mother's presence periods 
# Adjust these dates to match your actual data range!
# presence_periods = [
#     ('2024-09-21 12:30:00', '2024-09-23 02:55:00'),  # First visit
#     ('2024-10-01 02:55:00', '2024-10-03 05:25:00'),  # Second visit  
#     ('2024-10-11 23:10:00', '2024-10-13 13:25:00'),  # Third visit
# ]
presence_periods = [
    ('2024-09-28 00:55:00', '2024-09-30 05:50:00'),  # First visit
    ('2024-10-07 15:55:00', '2024-10-09 04:05:00'),  # Second visit
    ('2024-10-14 21:05:00', '2024-10-17 03:00:00'),  # Third visit
    ('2024-10-25 05:55:00', '2024-10-27 23:59:59'),  # Last visit
]

# Convert presence periods to datetime and filter to actual data range
valid_presence_periods = []
for arrival_str, departure_str in presence_periods:
    arrival = pd.to_datetime(arrival_str)
    departure = pd.to_datetime(departure_str)
    # Only include periods that overlap with our data
    if arrival <= end_time and departure >= start_time:
        valid_presence_periods.append((arrival, departure))

print(f"Valid presence periods: {len(valid_presence_periods)}")

# Create boolean mask for mother presence
mother_present = np.zeros(len(week_data), dtype=bool)

for arrival, departure in valid_presence_periods:
    mask = (week_data['Timestamp'] >= arrival) & (week_data['Timestamp'] <= departure)
    mother_present |= mask

print(f"Time points when mother present: {mother_present.sum()}")

# Create the plot
plt.figure(figsize=(24, 6))

nursing_binary = (week_data['Nursing_Flag'] == 1).astype(int)

# Plot nursing events when mother is ABSENT (light blue)
absent_nursing = nursing_binary.copy()
absent_nursing[mother_present] = 0
plt.fill_between(week_data['Timestamp'], absent_nursing, step='post', 
                 alpha=0.4, color='lightblue')

# Plot nursing events when mother is PRESENT (dark blue)
present_nursing = nursing_binary.copy()
present_nursing[~mother_present] = 0
plt.fill_between(week_data['Timestamp'], present_nursing, step='post', 
                 alpha=0.8, color='steelblue')

# Format x-axis
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax.xaxis.set_minor_locator(mdates.DayLocator())

plt.xticks(rotation=45, fontsize = 18)
plt.yticks(fontsize = 18)

# Formatting
plt.ylim(0.0, 1.0)
plt.xlabel('Date (2024)', fontsize = 20)
plt.ylabel('Nursing Event', fontsize = 20)

# Plot mother arrival and departure lines
mother_arrivals = []
mother_departures = []

for arrival, departure in valid_presence_periods:
    mother_arrivals.append(arrival)
    mother_departures.append(departure)

# Plot arrival lines (green)
for arrival in mother_arrivals:
    plt.axvline(x=arrival, color='green', linestyle='--', alpha=0.7, linewidth=2)

# Plot departure lines (red)  
for departure in mother_departures:
    plt.axvline(x=departure, color='red', linestyle='--', alpha=0.7, linewidth=2)
plt.tight_layout()

# Save the plot
plt.savefig("nursing_with_mother_presence_P.png", dpi=300, bbox_inches='tight')
plt.show()

# Calculate statistics
total_nursing = nursing_binary.sum()
nursing_when_present = present_nursing.sum() 
nursing_when_absent = absent_nursing.sum()

print("\n=== STATISTICS ===")
print(f"Total nursing events: {total_nursing}")
print(f"Nursing when mother present: {nursing_when_present}")
print(f"Nursing when mother absent: {nursing_when_absent}")

if total_nursing > 0:
    absent_percentage = (nursing_when_absent / total_nursing) * 100
    present_percentage = (nursing_when_present / total_nursing) * 100
    print(f"Percentage when mother absent: {absent_percentage:.1f}%")
    print(f"Percentage when mother present: {present_percentage:.1f}%")
    
    if nursing_when_absent > 0:
        print(f"\n⚠️  {absent_percentage:.1f}% of nursing events occurred when mother was absent")
        print("   This could indicate false positives or other pups nursing")
else:
    print("No nursing events found in the data range")