import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
import pandas as pd

# Your confusion matrix data
cm = np.array([[2729, 159, 248, 33],
               [118, 1335, 358, 43],
               [99, 289, 5879, 70],
               [86, 84, 145, 309]])

# Class labels (customize these based on your actual classes)
class_names = ['Resting (0)', 'Nursing (1)', 'High-activity (2)', 'Low-activity (3)']

# Create the plot
plt.figure(figsize=(14, 12))

# Create heatmap with annotations
sns.heatmap(cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Count'},
            annot_kws={'size': 18, 'weight': 'bold'})

# Make tick labels bigger
plt.xticks(fontsize=16)  # Adjust size as needed
plt.yticks(fontsize=16)  # Adjust size as needed

# Customize the plot
#plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Predicted Label', fontsize=18, fontweight='bold')
plt.ylabel('True Label', fontsize=18, fontweight='bold')

# Improve layout
plt.tight_layout()

# Add grid for better readability
plt.gca().set_aspect('equal')

# Save the figure
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')

plt.show()

# Calculate and display metrics
def calculate_metrics(cm):
    """Calculate precision, recall, F1-score, and accuracy from confusion matrix"""
    
    # Calculate metrics for each class
    n_classes = cm.shape[0]
    precision = np.zeros(n_classes)
    recall = np.zeros(n_classes)
    f1_score = np.zeros(n_classes)
    
    for i in range(n_classes):
        # True positives
        tp = cm[i, i]
        # False positives
        fp = np.sum(cm[:, i]) - tp
        # False negatives
        fn = np.sum(cm[i, :]) - tp
        
        # Calculate metrics
        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0
    
    # Overall accuracy
    accuracy = np.trace(cm) / np.sum(cm)
    
    return precision, recall, f1_score, accuracy

# Calculate metrics
precision, recall, f1_score, accuracy = calculate_metrics(cm)

# Create a summary table
metrics_df = pd.DataFrame({
    'Class': class_names,
    'Precision': [f'{p:.3f}' for p in precision],
    'Recall': [f'{r:.3f}' for r in recall],
    'F1-Score': [f'{f:.3f}' for f in f1_score],
    'Support': [np.sum(cm[i, :]) for i in range(len(class_names))]
})

print("Classification Metrics:")
print("=" * 50)
print(metrics_df.to_string(index=False))
print(f"\nOverall Accuracy: {accuracy:.3f}")

# Macro and weighted averages
macro_precision = np.mean(precision)
macro_recall = np.mean(recall)
macro_f1 = np.mean(f1_score)

support = [np.sum(cm[i, :]) for i in range(len(class_names))]
weighted_precision = np.average(precision, weights=support)
weighted_recall = np.average(recall, weights=support)
weighted_f1 = np.average(f1_score, weights=support)

print(f"\nMacro Average:")
print(f"  Precision: {macro_precision:.3f}")
print(f"  Recall: {macro_recall:.3f}")
print(f"  F1-Score: {macro_f1:.3f}")

print(f"\nWeighted Average:")
print(f"  Precision: {weighted_precision:.3f}")
print(f"  Recall: {weighted_recall:.3f}")
print(f"  F1-Score: {weighted_f1:.3f}")

# Alternative visualization with percentages
plt.figure(figsize=(12, 8))

# Calculate percentages
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

# Create subplot with both counts and percentages
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Plot with counts
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            ax=ax1, cbar_kws={'label': 'Count'})
ax1.set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Predicted Label', fontsize=12)
ax1.set_ylabel('True Label', fontsize=12)

# Plot with percentages
sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            ax=ax2, cbar_kws={'label': 'Percentage (%)'})
ax2.set_title('Confusion Matrix (Percentages)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Predicted Label', fontsize=12)
ax2.set_ylabel('True Label', fontsize=12)

plt.tight_layout()
plt.savefig('confusion_matrix_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
