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
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, 
    classification_report, confusion_matrix, 
    make_scorer
)
import pickle
import joblib
import os

dfs = pd.read_csv('Final_comp_fixed.csv') #change for not compressed or for adaptive

#print(dfs['Behavior'].value_counts().sort_index())

# Replace values in 'Behavior' column - make better behavior categories
dfs['Behavior'] = dfs['Behavior'].replace({7: 2}) #make interacting active.
dfs['Behavior'] = dfs['Behavior'].replace({5: 2}) #make vocalizing active.
#dfs['Behavior'] = dfs['Behavior'].replace({3: 2}) #make grooming active.
dfs = dfs[dfs['Behavior'] != 6] #delete shaking - so few values and not confident on timing.


# Renaming behavior classes
rename_map = {
    0: 0,  
    1: 1,
    2: 2,
    3:3,
    4: 4  
     # 5, 6 and 7 are gone
    
}

dfs['Behavior'] = dfs['Behavior'].map(rename_map)

# Verify the changes
#print("\nBehavior counts after modification:")
#print(dfs['Behavior'].value_counts().sort_index())


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
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
def run_model_pipeline(
    X_train, y_train, X_val, y_val, X_test, y_test,
    feature_names=None,
    use_smote=False,
    use_random_search=False,
    param_dist=None,
    cv=None,
    sample_weights=None,
    run_cross_validation=False
):
    # Use selected features or all
    X_train = X_train[feature_names] if feature_names is not None else X_train
    X_val = X_val[feature_names] if feature_names is not None else X_val
    X_test = X_test[feature_names] if feature_names is not None else X_test

    # Base XGBClassifier
    xgb = XGBClassifier(
        objective='multi:softmax',
        num_class=len(np.unique(y_train)),
        eval_metric='mlogloss',
        use_label_encoder=False,
        n_jobs=1,
        random_state=42
    )

    if use_smote:
        steps = [('smote', SMOTE(random_state=42)), ('xgb', xgb)]
        pipeline = Pipeline(steps)
        fit_params = {} # Sample weights should NOT be passed when using SMOTE
    else:
        pipeline = Pipeline([('xgb', xgb)])
        fit_params = {}
        if sample_weights is not None:
            fit_params['xgb__sample_weight'] = sample_weights # Create fit parameters with properly routed sample_weights

    if use_random_search:
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_dist,
            n_iter=30,
            scoring='f1_macro',
            cv=cv,
            verbose=1,
            n_jobs=2,
            random_state=42
        )
        search.fit(X_train, y_train, **fit_params)
        model = search.best_estimator_
        print("Best Params:", search.best_params_)
    else:
        pipeline.fit(X_train, y_train, **fit_params)
        model = pipeline

# Run cross-validation if requested
    if run_cross_validation:
        # Define multiple scoring metrics for cross_validate
        scoring = {
            'accuracy': 'accuracy',
            'f1_macro': 'f1_macro',
            'f1_weighted': 'f1_weighted',
            'roc_auc_ovr': make_scorer(roc_auc_score, needs_proba=True, multi_class='ovr', average='macro')
        }
        
        # Create a fresh pipeline for CV with the same parameters
        if use_smote:
            # For SMOTE with CV, we need to use a special pipeline
            from imblearn.pipeline import Pipeline as ImbPipeline
            cv_pipeline = ImbPipeline([('smote', SMOTE(random_state=42)), ('xgb', xgb)])
            # DON'T pass sample_weights with SMOTE
            cv_fit_params = {}
        else:
            cv_pipeline = Pipeline([('xgb', xgb)])
            # Only use sample_weights when NOT using SMOTE
            cv_fit_params = {}
            if sample_weights is not None:
                cv_fit_params['xgb__sample_weight'] = sample_weights

        # If we used RandomSearch, apply the best parameters to our CV pipeline
        if use_random_search:
            # Extract the parameters for xgb
            xgb_params = {k.replace("xgb__", ""): v for k, v in search.best_params_.items() 
                  if k.startswith("xgb__")}
            cv_pipeline.named_steps['xgb'].set_params(**xgb_params)
        
        # Run cross-validation for metrics
        cv_results = cross_validate(
            cv_pipeline, 
            X_train, 
            y_train, 
            cv=cv, 
            scoring=scoring,
            return_train_score=True,
            n_jobs=2,
            fit_params=cv_fit_params  # Pass fit_params to cross_validate
        )
        
          # Run manual cross-validation to collect predictions for classification report and confusion matrix
        fold_predictions = []
        fold_true_values = []
        fold_confusion_matrices = []
        fold_class_reports = []
        unique_classes = np.unique(y_train)
        
        print("\nCollecting detailed cross-validation metrics...")
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
            print(f"Processing fold {fold_idx+1}/{cv.n_splits}...")
            # Split data for this fold
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            # If using sample weights and not using SMOTE, prepare fold-specific weights
            fold_fit_params = {}
            if sample_weights is not None and not use_smote:
                fold_weights = sample_weights[train_idx]
                fold_fit_params['xgb__sample_weight'] = fold_weights
            
            # Create and fit a fresh model for this fold
            fold_model = cv_pipeline.fit(X_fold_train, y_fold_train, **fold_fit_params)
            
            # Predict on validation fold
            y_fold_pred = fold_model.predict(X_fold_val)
            
            # Store true values and predictions
            fold_true_values.append(y_fold_val)
            fold_predictions.append(y_fold_pred)
            
            # Calculate and store confusion matrix
            cm = confusion_matrix(y_fold_val, y_fold_pred, labels=unique_classes)
            fold_confusion_matrices.append(cm)
            
            # Calculate and store classification report as dictionary
            cr = classification_report(y_fold_val, y_fold_pred, output_dict=True)
            fold_class_reports.append(cr)

        # Rest of the function remains the same...
        # Calculate average confusion matrix across folds
        avg_conf_matrix = np.mean(fold_confusion_matrices, axis=0).astype(int)
        
        # Create an average classification report
        avg_report = {}
        # Initialize with the structure of the first report
        for class_label in fold_class_reports[0].keys():
            if class_label in ['accuracy', 'macro avg', 'weighted avg']:
                avg_report[class_label] = {}
                for metric in fold_class_reports[0][class_label].keys():
                    avg_report[class_label][metric] = np.mean([report[class_label][metric] for report in fold_class_reports])
            elif not class_label.isdigit():  # Skip numeric class labels which are handled separately
                continue
            else:
                avg_report[class_label] = {}
                for metric in fold_class_reports[0][class_label].keys():
                    avg_report[class_label][metric] = np.mean([report[class_label][metric] for report in fold_class_reports])
        
        # Print CV results
        print("\nCross-Validation Results:")
        print(f"CV Accuracy: {cv_results['test_accuracy'].mean():.4f} ± {cv_results['test_accuracy'].std():.4f}")
        print(f"CV F1 Macro: {cv_results['test_f1_macro'].mean():.4f} ± {cv_results['test_f1_macro'].std():.4f}")
        print(f"CV F1 Weighted: {cv_results['test_f1_weighted'].mean():.4f} ± {cv_results['test_f1_weighted'].std():.4f}")
        print(f"CV ROC-AUC: {cv_results['test_roc_auc_ovr'].mean():.4f} ± {cv_results['test_roc_auc_ovr'].std():.4f}")
        
        # Print average confusion matrix
        print("\nAverage Confusion Matrix across CV folds:")
        print(avg_conf_matrix)
        
        # Print average classification report
        print("\nAverage Classification Report across CV folds:")
        # Format the report similar to sklearn's classification_report output
        print(f"              precision    recall  f1-score   support")
        for class_label in sorted([k for k in avg_report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]):
            print(f"         {class_label:2s}       {avg_report[class_label]['precision']:.2f}      {avg_report[class_label]['recall']:.2f}      {avg_report[class_label]['f1-score']:.2f}      {int(avg_report[class_label]['support'])}")
        print("")
        for avg_type in ['macro avg', 'weighted avg']:
            if avg_type in avg_report:
                print(f"    {avg_type}       {avg_report[avg_type]['precision']:.2f}      {avg_report[avg_type]['recall']:.2f}      {avg_report[avg_type]['f1-score']:.2f}      {int(avg_report[avg_type]['support'])}")
        print(f"      accuracy                           {avg_report['accuracy']:.2f}      {int(np.sum([report['accuracy'] * report[list(report.keys())[0]]['support'] for report in fold_class_reports]) / np.sum([report[list(report.keys())[0]]['support'] for report in fold_class_reports]))}")
        
        # Compare CV vs non-CV results
        print("\nComparison: CV vs. Single Train/Test Split")
        y_val_pred = model.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        val_f1_macro = f1_score(y_val, y_val_pred, average='macro')
        print(f"Single Split Validation Accuracy: {val_accuracy:.4f} vs CV Accuracy: {cv_results['test_accuracy'].mean():.4f}")
        print(f"Single Split Validation F1 Macro: {val_f1_macro:.4f} vs CV F1 Macro: {cv_results['test_f1_macro'].mean():.4f}")
        
        print("\nConfusion Matrix Comparison:")
        print("Cross-Validation Average Confusion Matrix:")
        print(avg_conf_matrix)
        print("\nSingle Split Validation Confusion Matrix:")
        print(confusion_matrix(y_val, y_val_pred, labels=unique_classes))

    # Standard evaluation on validation set
    y_val_pred = model.predict(X_val)
    print("\nValidation Set Performance:")
    print("Accuracy:", accuracy_score(y_val, y_val_pred))
    print("Macro F1-Score:", f1_score(y_val, y_val_pred, average='macro'))

    # Evaluate on test
    y_test_pred = model.predict(X_test)
    y_test_pred_proba = model.predict_proba(X_test)

    print("\nStandard Test Set Performance:")
    print("Accuracy:", accuracy_score(y_test, y_test_pred))
    print("Macro F1-Score:", f1_score(y_test, y_test_pred, average='macro'))
    print("Weighted F1-Score:", f1_score(y_test, y_test_pred, average='weighted'))
    print("ROC-AUC:", roc_auc_score(y_test, y_test_pred_proba, multi_class='ovr', average='macro'))
    print("Classification Report:\n", classification_report(y_test, y_test_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))

    # Feature importances
    if hasattr(model, 'named_steps'):
        xgb_step = model.named_steps['xgb'] if 'xgb' in model.named_steps else model.steps[0][1]
    else:
        xgb_step = model.get_params()['steps'][1][1]  # For RandomizedSearchCV best_estimator_
    
    feature_importance = pd.Series(xgb_step.feature_importances_, 
                                  index=feature_names if feature_names is not None else X_train.columns).sort_values(ascending=False)
    print("\nFeature Importance:")
    print(feature_importance)

    return model, feature_importance

param_dist = {
    'xgb__n_estimators': [100, 200, 300],
    'xgb__max_depth': [3, 5, 7],
    'xgb__learning_rate': [0.01, 0.05, 0.1],
    'xgb__subsample': [0.7, 0.9, 1.0],
    'xgb__colsample_bytree': [0.7, 0.9, 1.0],
    'xgb__min_child_weight': [1, 3, 5]
}

# Train base model with full features, SMOTE, and random search
print("\n1. SMOTE + RandomSearch + Full Features + CV")
model1, feat1 = run_model_pipeline(X_train, y_train, X_val, y_val, X_test, y_test,
                                   use_smote=True, use_random_search=True,
                                   param_dist=param_dist, cv=cv,
                                   sample_weights=sample_weights, run_cross_validation=True)

# Get top features
selected_features = feat1.head(80).index  # Explicitly choose top number


# Train model with selected features
print("\n2. SMOTE + Reduced Features + Randomsearch + CV")
model2, feat2 = run_model_pipeline(X_train, y_train, X_val, y_val, X_test, y_test,
                                   feature_names=selected_features,
                                   use_smote=True, use_random_search=True,
                                   param_dist=param_dist, cv=cv,
                                   sample_weights=sample_weights, run_cross_validation=True)


# Train model without SMOTE
print("\n3. No SMOTE + Full Features + randomsearch +CV")
model3, feat3 = run_model_pipeline(X_train, y_train, X_val, y_val, X_test, y_test,
                                   use_smote=False, use_random_search=True,
                                   param_dist=param_dist, cv=cv,
                                   sample_weights=sample_weights, run_cross_validation=True)

# Train model without SMOTE, reduced features
print("\n4. No SMOTE + Reduced Features +randomsearch + CV")
model4, feat4 = run_model_pipeline(X_train, y_train, X_val, y_val, X_test, y_test,
                                   feature_names=selected_features,
                                   use_smote=False, use_random_search=True,
                                   param_dist=param_dist, cv=cv,
                                   sample_weights=sample_weights, run_cross_validation=True)


####### LEARNING CURVES ########## FOR BEST MODEL ##
def plot_learning_curves(model, X, y, cv=None, train_sizes=np.linspace(0.1, 1.0, 10)):
    """
    Plot learning curves for a trained model.
    
    Parameters:
    -----------
    model : estimator instance
        The model for which to generate learning curves
    X : array-like of shape (n_samples, n_features)
        Training data
    y : array-like of shape (n_samples,)
        Target values
    cv : int, cross-validation generator, or iterable
        Determines the cross-validation strategy
    train_sizes : array-like
        Relative or absolute numbers of training examples for learning curve points
    """
    plt.figure(figsize=(12, 6))
    
    train_sizes, train_scores, test_scores = learning_curve(
        model,
        X,
        y,
        cv=cv,
        train_sizes=train_sizes,
        scoring='f1_macro',
        n_jobs=2,
        random_state=42
    )
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    # Plot learning curve
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.title("Learning Curves", fontsize=14)
    plt.xlabel("Training examples", fontsize=12)
    plt.ylabel("F1 Macro Score", fontsize=12)
    plt.legend(loc="best")
    
    # Add a second plot with a different y-scale to better see small differences
    plt.figure(figsize=(12, 6))
    plt.grid()
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.title("Learning Curves (Zoomed)", fontsize=14)
    plt.xlabel("Training examples", fontsize=12)
    plt.ylabel("F1 Macro Score", fontsize=12)
    plt.legend(loc="best")
    
    # Set a reasonable y-axis range based on scores
    min_score = min(np.min(train_scores_mean), np.min(test_scores_mean))
    max_score = max(np.max(train_scores_mean), np.max(test_scores_mean))
    padding = (max_score - min_score) * 0.1  # 10% padding
    plt.ylim(max(0, min_score - padding), min(1.0, max_score + padding))
    
    plt.tight_layout()
    plt.savefig('xgboost_learning_curves.png', dpi=300)
    plt.show()

# After running all your models, call this function with your best model
# Find best model based on test F1 score
print("\nComparing all models on test F1 score:")
models = [model1, model2, model3, model4]
model_names = ["SMOTE + Full Features", "SMOTE + Reduced Features", 
               "No SMOTE + Full Features", "No SMOTE + Reduced Features"]

# Get test F1 scores for all models
test_f1_scores = []
for model in models:
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='macro')
    test_f1_scores.append(f1)

# Print comparison
for name, f1 in zip(model_names, test_f1_scores):
    print(f"{name}: F1 = {f1:.4f}")

# Get best model
best_idx = np.argmax(test_f1_scores)
best_model = models[best_idx]
print(f"\nBest model: {model_names[best_idx]} with F1 = {test_f1_scores[best_idx]:.4f}")

# Generate learning curves for best model
print("\nGenerating learning curves for best model...")
# If the best model uses reduced features, we need to use the reduced feature set
if best_idx % 2 == 1:  # Models 1 and 3 use reduced features
    plot_learning_curves(best_model, X_train[selected_features], y_train, cv=cv)
else:
    plot_learning_curves(best_model, X_train, y_train, cv=cv)

# Save the best model
print("\nSaving best model...")
joblib.dump(best_model, 'best_xgboost_model.joblib')
print("Best model saved as 'best_xgboost_model.joblib'")
