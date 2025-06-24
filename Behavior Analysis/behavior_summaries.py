import pandas as pd
import numpy as np

# ===========================
# STEP 1: CREATE BEHAVIOR FLAGS
# ===========================

# Load your data
output = pd.read_csv("allpups_predictions_final.csv")

# Check the structure of your prediction data
print("=== DATA STRUCTURE CHECK ===")
print("Data shape:", output.shape)
print("\nColumn names:")
print(output.columns.tolist())
print("\nFirst few rows:")
print(output.head())

# Define the prediction column
PREDICTION_COLUMN = 'Filtered_Predicted_Behavior'  

if PREDICTION_COLUMN in output.columns:
    print(f"\n=== BEHAVIOR CODE ANALYSIS ===")
    print(f"Unique values in {PREDICTION_COLUMN}:")
    value_counts = output[PREDICTION_COLUMN].value_counts().sort_index()
    print(value_counts)
    
    # If there's an existing Nursing_Flag, check which code corresponds to it
    if 'Nursing_Flag' in output.columns:
        nursing_code_check = output.groupby('Filtered_Predicted_Behavior')['Nursing_Flag'].mean()
        print(f"\nNursing flag by prediction code (existing column):")
        print(nursing_code_check)
    
    # Check if there's a Behavior_Label column for reference
    if 'Behavior_Label' in output.columns:
        print(f"\nBehavior labels by prediction code:")
        behavior_by_code = output.groupby('Filtered_Predicted_Behavior')['Behavior_Label'].value_counts()
        print(behavior_by_code)
    
    print(f"\n=== CREATING BEHAVIOR FLAGS ===")
    print("Mapping codes to behaviors...")
    print("Note: You may need to adjust these mappings based on your specific data")
    
    # Common behavior mapping (adjust these based on your data inspection above)
    # -1 = Uncertain/Unknown
    # 0 = Resting 
    # 1 = Nursing
    # 2 = High Activity
    # 3 = Low Activity  
    # 4 = Swimming
    
    # Create behavior flags based on numeric codes
    output['Resting_Flag'] = (output[PREDICTION_COLUMN] == 0).astype(int)
    output['Nursing_Flag'] = (output[PREDICTION_COLUMN] == 1).astype(int)
    output['High_Activity_Flag'] = (output[PREDICTION_COLUMN] == 2).astype(int)
    output['Low_Activity_Flag'] = (output[PREDICTION_COLUMN] == 3).astype(int)
    output['Swimming_Flag'] = (output[PREDICTION_COLUMN] == 4).astype(int)
    output['Uncertain_Flag'] = (output[PREDICTION_COLUMN] == -1).astype(int)
    
    # Verify the flag creation
    print(f"\nBehavior flag summary:")
    flag_columns = ['Resting_Flag', 'Nursing_Flag', 'High_Activity_Flag', 'Low_Activity_Flag', 'Swimming_Flag', 'Uncertain_Flag']
    
    for flag in flag_columns:
        if flag in output.columns:
            count = output[flag].sum()
            pct = (count / len(output)) * 100
            print(f"{flag}: {count:,} observations ({pct:.1f}%)")
    
    # Check coverage
    certain_flags = ['Resting_Flag', 'Nursing_Flag', 'High_Activity_Flag', 'Low_Activity_Flag', 'Swimming_Flag']
    total_flagged = output[certain_flags].sum(axis=1).sum()
    total_obs = len(output)
    uncertain_obs = output['Uncertain_Flag'].sum()
    
    print(f"\nCoverage check:")
    print(f"Total certain behavior observations: {total_flagged:,}")
    print(f"Total uncertain observations: {uncertain_obs:,}")
    print(f"Total observations: {total_obs:,}")
    print(f"Certain behavior coverage: {(total_flagged/total_obs)*100:.1f}%")
    print(f"Uncertain coverage: {(uncertain_obs/total_obs)*100:.1f}%")
    
    # Save the dataset with flags
    output.to_csv("allpups_predictions_with_flags.csv", index=False)
    print(f"\nDataset with flags saved as 'allpups_predictions_with_flags.csv'")

else:
    print(f"\n⚠️ Column '{PREDICTION_COLUMN}' not found!")
    print("Please check your column names and update PREDICTION_COLUMN variable.")
    print("\nAvailable columns:")
    for i, col in enumerate(output.columns):
        print(f"  {i}: {col}")
    exit()

# ===========================
# STEP 2: BEHAVIOR ANALYSIS FUNCTIONS
# ===========================

def calculate_bout_durations_corrected(pup_data, behavior_column):
    """Calculate bout durations for a single pup's data - CORRECTED VERSION"""
    if len(pup_data) == 0 or pup_data[behavior_column].sum() == 0:
        return pd.Series(dtype=float)
    
    # Sort by timestamp
    pup_data = pup_data.copy().sort_values('Timestamp')
    
    # Create run-length encoding to identify continuous periods
    behavior_values = pup_data[behavior_column].values
    
    # Find where behavior changes (including start of data)
    change_points = np.concatenate(([True], behavior_values[1:] != behavior_values[:-1]))
    
    # Create bout groups
    bout_groups = change_points.cumsum()
    pup_data['Bout_Group'] = bout_groups
    
    # Filter for only the behavior of interest
    behavior_data = pup_data[pup_data[behavior_column] == 1].copy()
    
    if len(behavior_data) == 0:
        return pd.Series(dtype=float)
    
    # Calculate duration for each bout
    bout_durations = []
    for bout_id in behavior_data['Bout_Group'].unique():
        bout_data = behavior_data[behavior_data['Bout_Group'] == bout_id]
        
        # Calculate actual duration based on number of observations and sampling frequency
        duration_minutes = len(bout_data) * get_sampling_interval_minutes(pup_data)
        bout_durations.append(duration_minutes)
    
    return pd.Series(bout_durations)

def get_sampling_interval_minutes(pup_data):
    """Estimate the sampling interval in minutes"""
    # Calculate median time difference between consecutive observations
    time_diffs = pup_data['Timestamp'].diff().dropna()
    if len(time_diffs) > 0:
        median_interval = time_diffs.median().total_seconds() / 60
        return median_interval
    else:
        # Default assumption: 1 minute intervals
        return 1.0

def calculate_behavior_summary_corrected(pup_data, behavior_column):
    """Calculate comprehensive summary statistics - CORRECTED VERSION"""
    
    # Calculate daily behavior frequency (percentage of time in behavior per day)
    daily_freq = pup_data.groupby('Date')[behavior_column].mean() * 100
    avg_daily_freq = daily_freq.mean()
    
    # Calculate daily duration using simple counting approach
    sampling_interval = get_sampling_interval_minutes(pup_data)
    
    # Daily duration = number of behavior observations per day * sampling interval
    daily_duration = pup_data.groupby('Date')[behavior_column].sum() * sampling_interval
    avg_daily_duration = daily_duration.mean()
    
    # Calculate bout statistics using corrected method
    bout_durations = calculate_bout_durations_corrected(pup_data, behavior_column)
    
    if len(bout_durations) > 0:
        total_bouts = len(bout_durations)
        avg_bout_duration = bout_durations.mean()
        max_bout_duration = bout_durations.max()
        
        # Calculate daily bout count
        pup_data_sorted = pup_data.sort_values('Timestamp')
        behavior_values = pup_data_sorted[behavior_column].values
        change_points = np.concatenate(([True], behavior_values[1:] != behavior_values[:-1]))
        pup_data_sorted['Bout_Group'] = change_points.cumsum()
        
        behavior_bouts = pup_data_sorted[pup_data_sorted[behavior_column] == 1]
        if len(behavior_bouts) > 0:
            daily_bout_count = behavior_bouts.groupby('Date')['Bout_Group'].nunique()
            avg_daily_bouts = daily_bout_count.mean()
        else:
            avg_daily_bouts = 0
    else:
        total_bouts = 0
        avg_bout_duration = 0
        max_bout_duration = 0
        avg_daily_bouts = 0
    
    return {
        f'{behavior_column}_Avg_Daily_Frequency_Percent': round(avg_daily_freq, 2),
        f'{behavior_column}_Avg_Daily_Duration_Minutes': round(avg_daily_duration, 2),
        f'{behavior_column}_Total_Bouts': total_bouts,
        f'{behavior_column}_Avg_Bout_Duration_Minutes': round(avg_bout_duration, 2),
        f'{behavior_column}_Max_Bout_Duration_Minutes': round(max_bout_duration, 2),
        f'{behavior_column}_Avg_Daily_Bout_Count': round(avg_daily_bouts, 2),
        f'{behavior_column}_Sampling_Interval_Minutes': round(sampling_interval, 2)
    }

# ===========================
# STEP 3: CALCULATE BEHAVIOR SUMMARIES
# ===========================

print(f"\n=== CALCULATING BEHAVIOR SUMMARIES ===")

# Reload the data with flags
output = pd.read_csv('allpups_predictions_with_flags.csv')

# Convert Timestamp to datetime and create Date column
output['Timestamp'] = pd.to_datetime(output['Timestamp'])
output['Date'] = output['Timestamp'].dt.date

# Define behavior columns (excluding uncertain and low-activity for main analysis)
behavior_columns = ['Resting_Flag', 'Nursing_Flag', 'High_Activity_Flag', 'Swimming_Flag']

# Create comprehensive summary dataset by pup
pup_summary_list = []

print(f"Processing {output['Tag.ID'].nunique()} unique pups...")

for pup_id in output['Tag.ID'].unique():
    pup_data = output[output['Tag.ID'] == pup_id].copy()
    
    # Basic observation info
    first_obs = pup_data['Date'].min()
    last_obs = pup_data['Date'].max()
    total_days = len(pup_data['Date'].unique())
    total_observations = len(pup_data)
    
    # Initialize summary record
    pup_summary = {
        'Pup_ID': pup_id,
        'Total_Days_Observed': total_days,
        'Total_Observations': total_observations,
        'First_Observation': first_obs,
        'Last_Observation': last_obs
    }
    
    # Calculate summary for each behavior
    for behavior in behavior_columns:
        if behavior in pup_data.columns:
            behavior_stats = calculate_behavior_summary_corrected(pup_data, behavior)
            pup_summary.update(behavior_stats)
        else:
            print(f"Warning: {behavior} column not found for pup {pup_id}")
    
    # Add uncertain behavior percentage
    if 'Uncertain_Flag' in pup_data.columns:
        uncertain_pct = pup_data['Uncertain_Flag'].mean() * 100
        pup_summary['Uncertain_Avg_Daily_Frequency_Percent'] = round(uncertain_pct, 2)
    
    # Validation check: do percentages add up to ~100%?
    total_pct = sum([pup_summary.get(f'{b}_Avg_Daily_Frequency_Percent', 0) 
                     for b in behavior_columns if f'{b}_Avg_Daily_Frequency_Percent' in pup_summary])
    uncertain_pct = pup_summary.get('Uncertain_Avg_Daily_Frequency_Percent', 0)
    pup_summary['Total_Behavior_Percentage'] = round(total_pct + uncertain_pct, 2)
    pup_summary['Certain_Behavior_Percentage'] = round(total_pct, 2)
    
    pup_summary_list.append(pup_summary)

# Create the final comprehensive summary dataframe
pup_behavior_summary = pd.DataFrame(pup_summary_list)

# ===========================
# STEP 4: VALIDATION AND RESULTS
# ===========================

print(f"\n=== VALIDATION CHECKS ===")
print(f"Number of pups analyzed: {len(pup_behavior_summary)}")

# Check sampling intervals
sampling_cols = [col for col in pup_behavior_summary.columns if 'Sampling_Interval' in col]
if sampling_cols:
    print(f"\nSampling intervals detected (minutes):")
    print(pup_behavior_summary[sampling_cols[0]].describe())

# Check total percentages
print(f"\nTotal behavior percentages (should be close to 100%):")
if 'Total_Behavior_Percentage' in pup_behavior_summary.columns:
    print(pup_behavior_summary['Total_Behavior_Percentage'].describe())

print(f"\nCertain behavior percentages (excluding uncertain):")
if 'Certain_Behavior_Percentage' in pup_behavior_summary.columns:
    print(pup_behavior_summary['Certain_Behavior_Percentage'].describe())

# Check individual behavior statistics
print(f"\n=== BEHAVIOR STATISTICS ===")
for behavior in behavior_columns:
    freq_col = f'{behavior}_Avg_Daily_Frequency_Percent'
    dur_col = f'{behavior}_Avg_Daily_Duration_Minutes'
    
    if freq_col in pup_behavior_summary.columns:
        print(f"\n{behavior.replace('_Flag', '')} behavior:")
        print(f"  Frequency %: {pup_behavior_summary[freq_col].describe()}")
        if dur_col in pup_behavior_summary.columns:
            print(f"  Duration min/day: {pup_behavior_summary[dur_col].describe()}")

# Save the corrected summary dataset
pup_behavior_summary.to_csv('pup_behavior_summary.csv', index=False)
print(f"\n=== SUMMARY ===")
print("Behavior summary saved as 'pup_behavior_summary.csv'")
print(f"Dataset shape: {pup_behavior_summary.shape}")

# Display sample results
print(f"\n=== SAMPLE RESULTS ===")
if len(pup_behavior_summary) > 0:
    sample_pup = pup_behavior_summary.iloc[0]
    print(f"Sample results for Pup ID: {sample_pup['Pup_ID']}")
    print(f"Total observations: {sample_pup['Total_Observations']}")
    print(f"Days observed: {sample_pup['Total_Days_Observed']}")
    
    for behavior in behavior_columns:
        freq_col = f'{behavior}_Avg_Daily_Frequency_Percent'
        dur_col = f'{behavior}_Avg_Daily_Duration_Minutes'
        if freq_col in sample_pup and dur_col in sample_pup:
            behavior_name = behavior.replace('_Flag', '')
            print(f"{behavior_name}: {sample_pup[freq_col]:.1f}% of time, {sample_pup[dur_col]:.1f} min/day")
    
    if 'Uncertain_Avg_Daily_Frequency_Percent' in sample_pup:
        print(f"Uncertain: {sample_pup['Uncertain_Avg_Daily_Frequency_Percent']:.1f}% of time")

print(f"\n=== COMPLETE ===")
print("Both files created:")
print("1. allpups_predictions_with_flags.csv (original data + behavior flags)")
print("2. pup_behavior_summary.csv (per-pup behavior statistics)")

#####################################################################3

#with pup_behavior_summary.csv - Create ethogram plot

# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# import seaborn as sns

# # Load the behavior summary data
# pup_behavior_summary = pd.read_csv('pup_behavior_summary.csv')
# # Remove the specific pup
# pup_behavior_summary = pup_behavior_summary[pup_behavior_summary['Pup_ID'] != 'UU_0000_S2']

# # Define behavior columns and their display names
# behavior_freq_columns = [
#     'Swimming_Flag_Avg_Daily_Frequency_Percent',
#     'Nursing_Flag_Avg_Daily_Frequency_Percent', 
#     'High_Activity_Flag_Avg_Daily_Frequency_Percent',
#     'Resting_Flag_Avg_Daily_Frequency_Percent'
# ]

# behavior_labels = ['In Water', 'Nursing', 'High Activity', 'Resting']

# # Define colors for each behavior (you can customize these)
# behavior_colors = ['#2E86AB', "#36B155", '#F18F01', '#C73E1D']

# # Check which columns exist in the data
# available_columns = [col for col in behavior_freq_columns if col in pup_behavior_summary.columns]
# available_labels = [behavior_labels[i] for i, col in enumerate(behavior_freq_columns) if col in pup_behavior_summary.columns]
# available_colors = [behavior_colors[i] for i, col in enumerate(behavior_freq_columns) if col in pup_behavior_summary.columns]

# print(f"Available behavior columns: {len(available_columns)}")
# print(f"Columns found: {available_columns}")

# # Create the figure with subplots - SWAPPED ORDER: averages on top, individual on bottom
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), 
#                                gridspec_kw={'height_ratios': [1, 4], 'hspace': 0.25})

# # Calculate population averages for top subplot
# population_averages = []
# for col in available_columns:
#     avg_value = pup_behavior_summary[col].fillna(0).mean()
#     population_averages.append(avg_value)

# # Create population average bar (TOP subplot - ax1)
# bottom_pop = 0
# bar_width = 0.6

# for i, (avg, label, color) in enumerate(zip(population_averages, available_labels, available_colors)):
#     ax1.barh(0, avg, left=bottom_pop, color=color, alpha=0.8, height=bar_width)
    
#     # Add percentage labels on bars if they're large enough
#     if avg > 5:  # Only label if bar is wide enough
#         ax1.text(bottom_pop + avg/2, 0, f'{avg:.1f}%', 
#                 ha='center', va='center', fontweight='bold', fontsize=14)
    
#     bottom_pop += avg

# # Customize population average plot (TOP subplot)
# ax1.set_yticks([0])
# ax1.set_yticklabels(['Population\nAverage'], fontsize=16, fontweight='bold')
# #ax1.set_xlabel('Daily Behavior Percentage (%)', fontsize=16)
# #ax1.set_title('Population Average Behavior Frequencies', fontsize=18, fontweight='bold')
# ax1.legend(available_labels, loc='upper right', bbox_to_anchor=(1.15, 1), fontsize=14)
# ax1.grid(axis='x', alpha=0.3)
# ax1.set_xlim(0, 80)
# ax1.tick_params(axis='x', labelsize=14)

# # Add vertical lines for reference
# for x in [25, 50, 75]:
#     ax1.axvline(x, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)

# # Sort pups by total observations for better visualization
# pup_behavior_summary_sorted = pup_behavior_summary.sort_values('Total_Observations', ascending=True)

# # Get pup IDs and their positions
# pup_ids = pup_behavior_summary_sorted['Pup_ID'].values
# n_pups = len(pup_ids)
# y_positions = np.arange(n_pups)

# # Create individual pup ethogram (BOTTOM subplot - ax2)
# bottom_values = np.zeros(n_pups)  # For stacking bars

# for i, (col, label, color) in enumerate(zip(available_columns, available_labels, available_colors)):
#     values = pup_behavior_summary_sorted[col].fillna(0).values
    
#     # Create horizontal stacked bars
#     ax2.barh(y_positions, values, left=bottom_values, 
#              color=color, label=label, alpha=0.8, height=0.9)
    
#     bottom_values += values

# # Customize individual pup plot (BOTTOM subplot)
# ax2.set_yticks(y_positions)
# pup_labels = [pid.split('_')[0] for pid in pup_ids]
# ax2.set_yticklabels(pup_labels, fontsize=14, rotation=0, ha='right')
# # Adjust tick spacing for better readability
# ax2.tick_params(axis='y', pad=8)
# ax2.tick_params(axis='x', labelsize=14)
# ax2.set_xlabel('Daily Behavior Percentage (%)', fontsize=16)
# #ax2.set_title('Individual Pup Behavior Frequencies (Ethogram)', fontsize=18, fontweight='bold')
# ax2.grid(axis='x', alpha=0.3)
# ax2.set_xlim(0, 80)

# # Add vertical lines for reference
# for x in [25, 50, 75]:
#     ax2.axvline(x, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)

# # Add summary statistics as text (moved to bottom subplot)
# total_pups = len(pup_behavior_summary)
# total_obs = pup_behavior_summary['Total_Observations'].sum()
# avg_days = pup_behavior_summary['Total_Days_Observed'].mean()

# #summary_text = f'Dataset Summary:\n• {total_pups} pups analyzed\n• {total_obs:,} total observations\n• {avg_days:.1f} avg days per pup'
# #ax2.text(1.02, 0.02, summary_text, transform=ax2.transAxes, fontsize=12,
#  #        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8),
#  #        verticalalignment='bottom')

# plt.tight_layout()
# plt.show()

# # Print summary statistics
# print("\n" + "="*50)
# print("ETHOGRAM SUMMARY STATISTICS")
# print("="*50)

# print(f"\nDataset Overview:")
# print(f"• Total pups: {total_pups}")
# print(f"• Total observations: {total_obs:,}")
# print(f"• Average days per pup: {avg_days:.1f}")

# print(f"\nPopulation Average Behavior Frequencies:")
# for label, avg in zip(available_labels, population_averages):
#     print(f"• {label}: {avg:.1f}%")

# print(f"\nBehavior Frequency Ranges (Min - Max):")
# for col, label in zip(available_columns, available_labels):
#     min_val = pup_behavior_summary[col].fillna(0).min()
#     max_val = pup_behavior_summary[col].fillna(0).max()
#     print(f"• {label}: {min_val:.1f}% - {max_val:.1f}%")

# # Optional: Save the plot
# plt.savefig('pup_ethogram_behavior_frequencies.png', dpi=300, bbox_inches='tight')
# print(f"\nPlot saved as 'pup_ethogram_behavior_frequencies.png'")

# # Optional: Create a more detailed analysis
# print(f"\n" + "="*50)
# print("DETAILED BEHAVIORAL ANALYSIS")
# print("="*50)

# # Calculate correlations between behaviors
# behavior_data = pup_behavior_summary[available_columns].fillna(0)
# correlation_matrix = behavior_data.corr()

# print(f"\nBehavior Correlations:")
# for i in range(len(available_labels)):
#     for j in range(i+1, len(available_labels)):
#         corr_val = correlation_matrix.iloc[i, j]
#         print(f"• {available_labels[i]} vs {available_labels[j]}: {corr_val:.3f}")

# # Identify extreme pups
# print(f"\nPups with Extreme Behavior Patterns:")
# for col, label in zip(available_columns, available_labels):
#     max_pup = pup_behavior_summary.loc[pup_behavior_summary[col].fillna(0).idxmax(), 'Pup_ID']
#     max_val = pup_behavior_summary[col].fillna(0).max()
#     print(f"• Highest {label}: Pup {max_pup} ({max_val:.1f}%)")
