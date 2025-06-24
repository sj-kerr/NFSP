import pickle
import joblib
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta


def analyze_nursing_by_mother_presence(output_df, mother_presence_data=None, by_pup=False):
    """
    Analyze nursing time based on mother presence status
    
    Parameters:
    -----------
    output_df : DataFrame
        Your predictions dataframe with Timestamp, Nursing_Flag, Pup_ID, etc.
    mother_presence_data : DataFrame, optional
        DataFrame with columns: ['Date', 'Tag.ID', 'Mother_status'] where Mother_status is 
        'Present', 'Partial', or 'Absent'
        If None, will create example data structure
    by_pup : bool, default=False
        Whether to analyze by individual pup or combine all pups
    
    Returns:
    --------
    summary_df : DataFrame
        Summary of nursing time by mother presence (and pup if by_pup=True)
    """
    
    # Convert timestamp to datetime if not already
    output_df['Timestamp'] = pd.to_datetime(output_df['Timestamp'])
    output_df['Date'] = output_df['Timestamp'].dt.date
    
    # Ensure Date column is the same type
    mother_presence_data['Date'] = pd.to_datetime(mother_presence_data['Date']).dt.date
    
    # Calculate daily nursing time (3-second bins)
    sampling_interval_seconds = 3  # Your data is binned every 3 seconds
    sampling_interval_minutes = sampling_interval_seconds / 60  # Convert to minutes
    
    # Group by date and optionally by pup
    if by_pup and 'Tag.ID' in output_df.columns:
        daily_nursing = output_df.groupby(['Date', 'Tag.ID']).agg({
            'Nursing_Flag': ['sum', 'count']
        }).reset_index()
        
        # Flatten column names
        daily_nursing.columns = ['Date', 'Tag.ID', 'Nursing_Episodes', 'Total_Records']
        
        # Merge with mother presence data on both Date and Tag.ID
        # Use inner join to only keep matched pairs
        nursing_summary = daily_nursing.merge(mother_presence_data, on=['Date', 'Tag.ID'], how='inner')
    
    # Calculate nursing time in minutes and hours (3-second bins)
    nursing_summary['Nursing_Time_Minutes'] = nursing_summary['Nursing_Episodes'] * sampling_interval_minutes
    nursing_summary['Nursing_Time_Hours'] = nursing_summary['Nursing_Time_Minutes'] / 60
    nursing_summary['Nursing_Time_Seconds'] = nursing_summary['Nursing_Episodes'] * sampling_interval_seconds
    
    # No need to handle missing mother status data since we're using inner join
    
    # Print information about matched vs unmatched data
    print(f"Original accelerometer data: {len(daily_nursing)} pup-day combinations")
    print(f"Matched with mother data: {len(nursing_summary)} pup-day combinations")
    print(f"Excluded (no mother data): {len(daily_nursing) - len(nursing_summary)} pup-day combinations")
    
    return nursing_summary, daily_nursing

def create_nursing_mother_presence_plots(nursing_summary, by_pup=False):
    """
    Create 4-panel visualizations for nursing time by mother presence
    """
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    if by_pup and 'Tag.ID' in nursing_summary.columns:
        # Create figure with 4 subplots for pup-specific analysis
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Nursing Behavior Analysis by Mother Presence (By Pup)', fontsize=16, fontweight='bold')
        
        # 1. Bar chart: Average nursing time by mother status (all pups combined)
        avg_nursing = nursing_summary.groupby('Mother_status')['Nursing_Time_Hours'].agg(['mean', 'std', 'count']).reset_index()
        
        ax1 = axes[0, 0]
        bars = ax1.bar(avg_nursing['Mother_status'], avg_nursing['mean'], 
                       yerr=avg_nursing['std'], capsize=5, alpha=0.7)
        ax1.set_title('Average Daily Nursing Time by Mother Presence\n(All Pups Combined)')
        ax1.set_ylabel('Nursing Time (hours)')
        ax1.set_xlabel('Mother Status')
        
        # Add value labels on bars
        for bar, mean_val, count in zip(bars, avg_nursing['mean'], avg_nursing['count']):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{mean_val:.1f}h\n(n={count})', 
                    ha='center', va='bottom', fontsize=10)
        
        # 2. Bar chart: Average nursing time by pup
        ax2 = axes[0, 1]
        pup_avg = nursing_summary.groupby(['Tag.ID', 'Mother_status'])['Nursing_Time_Hours'].mean().unstack(fill_value=0)
        pup_avg.plot(kind='bar', ax=ax2, alpha=0.8)
        ax2.set_title('Average Daily Nursing Time by Pup and Mother Status')
        ax2.set_ylabel('Nursing Time (hours)')
        ax2.set_xlabel('Pup ID')
        ax2.legend(title='Mother Status', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Heatmap: Pup vs Mother Status
        ax3 = axes[1, 0]
        pivot_data = nursing_summary.groupby(['Tag.ID', 'Mother_status'])['Nursing_Time_Hours'].mean().unstack(fill_value=0)
        sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax3)
        ax3.set_title('Average Nursing Hours Heatmap')
        ax3.set_ylabel('Pup ID')
        ax3.set_xlabel('Mother Status')
        
        # 4. Summary statistics table
        ax4 = axes[1, 1]
        ax4.axis('tight')
        ax4.axis('off')
        
        # Create summary table by mother status
        summary_stats = nursing_summary.groupby('Mother_status')['Nursing_Time_Hours'].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).round(2)
        summary_stats.columns = ['Days', 'Mean (h)', 'Std (h)', 'Min (h)', 'Max (h)']
        
        table = ax4.table(cellText=summary_stats.values,
                         rowLabels=summary_stats.index,
                         colLabels=summary_stats.columns,
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        ax4.set_title('Summary Statistics\n(Matched Data Only)')
        
    else:
        # Original plots for combined analysis - reduced to 4 panels
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Nursing Behavior Analysis by Mother Presence', fontsize=16, fontweight='bold')
        
        # 1. Bar chart: Average nursing time by mother status
        avg_nursing = nursing_summary.groupby('Mother_status')['Nursing_Time_Hours'].agg(['mean', 'std', 'count']).reset_index()
        
        ax1 = axes[0, 0]
        bars = ax1.bar(avg_nursing['Mother_status'], avg_nursing['mean'], 
                       yerr=avg_nursing['std'], capsize=5, alpha=0.7)
        ax1.set_title('Average Daily Nursing Time by Mother Presence')
        ax1.set_ylabel('Nursing Time (hours)')
        ax1.set_xlabel('Mother Status')
        
        # Add value labels on bars
        for bar, mean_val, count in zip(bars, avg_nursing['mean'], avg_nursing['count']):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{mean_val:.1f}h\n(n={count})', 
                    ha='center', va='bottom', fontsize=10)
        
        # 2. Time series plot: Nursing over time colored by mother status
        ax2 = axes[0, 1]
        
        # Convert Date back to datetime for plotting
        nursing_summary['Date_dt'] = pd.to_datetime(nursing_summary['Date'])
        
        status_colors = {'Present': 'green', 'Partial': 'orange', 'Absent': 'red'}
        
        for status in nursing_summary['Mother_status'].unique():
            status_data = nursing_summary[nursing_summary['Mother_status'] == status]
            ax2.scatter(status_data['Date_dt'], status_data['Nursing_Time_Hours'], 
                       c=status_colors.get(status, 'blue'), label=status, alpha=0.7, s=50)
        
        ax2.set_title('Daily Nursing Time Timeline')
        ax2.set_ylabel('Nursing Time (hours)')
        ax2.set_xlabel('Date')
        ax2.legend()
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Create a simple heatmap placeholder for combined analysis
        ax3 = axes[1, 0]
        # For combined analysis, create a simple summary visualization
        status_summary = nursing_summary.groupby('Mother_status')['Nursing_Time_Hours'].agg(['mean', 'count']).reset_index()
        bars = ax3.bar(status_summary['Mother_status'], status_summary['mean'], alpha=0.7)
        ax3.set_title('Mean Nursing Time by Status\n(Matched Data Only)')
        ax3.set_ylabel('Mean Nursing Time (hours)')
        ax3.set_xlabel('Mother Status')
        
        # 4. Summary statistics table
        ax4 = axes[1, 1]
        ax4.axis('tight')
        ax4.axis('off')
        
        # Create summary table
        summary_stats = nursing_summary.groupby('Mother_status')['Nursing_Time_Hours'].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).round(2)
        summary_stats.columns = ['Days', 'Mean (h)', 'Std (h)', 'Min (h)', 'Max (h)']
        
        table = ax4.table(cellText=summary_stats.values,
                         rowLabels=summary_stats.index,
                         colLabels=summary_stats.columns,
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        ax4.set_title('Summary Statistics\n(Matched Data Only)')
    
    plt.tight_layout()
    filename = 'nursing_mother_presence_by_pup_matched_only.png' if by_pup else 'nursing_mother_presence_analysis_matched_only.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig


# === MAIN EXECUTION ===

# Load your data
# Option 1: If you have a single combined file
output = pd.read_csv("allpups_predictions_final.csv")

# Load mother presence data
mother_data = pd.read_csv("mother_presence.csv")

# Run the analysis BY PUP
print("Analyzing nursing behavior by mother presence and pup (matched data only)...")
nursing_summary, daily_summary = analyze_nursing_by_mother_presence(output, mother_data, by_pup=True)

# Create visualizations with 4 panels only
print("Creating 4-panel visualizations...")
fig = create_nursing_mother_presence_plots(nursing_summary, by_pup=True)

# Print summary statistics by pup and mother status
print("\n=== NURSING TIME SUMMARY BY MOTHER PRESENCE AND PUP (MATCHED DATA ONLY) ===")

# Overall summary by mother status
overall_stats = nursing_summary.groupby('Mother_status')['Nursing_Time_Hours'].agg([
    'count', 'mean', 'std', 'min', 'max'
]).round(2)
print("Overall Summary (All Pups - Matched Data Only):")
print(overall_stats)

# Summary by pup
print("\nSummary by Individual Pup:")
pup_stats = nursing_summary.groupby(['Tag.ID', 'Mother_status'])['Nursing_Time_Hours'].agg([
    'count', 'mean', 'std'
]).round(2)
print(pup_stats)

print(f"\nTotal pups analyzed: {nursing_summary['Tag.ID'].nunique()}")
print(f"Total days analyzed: {len(nursing_summary)}")
print(f"Total nursing episodes detected: {nursing_summary['Nursing_Episodes'].sum()}")
print(f"Total nursing time: {nursing_summary['Nursing_Time_Hours'].sum():.1f} hours")

# Save detailed results
nursing_summary.to_csv("nursing_mother_analysis_matched_only.csv", index=False)
print("\n✅ Detailed analysis saved to: nursing_mother_analysis_matched_only.csv")
print("✅ Visualization saved to: nursing_mother_presence_by_pup_matched_only.png")

######################### 2 panel ############################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


def analyze_nursing_by_mother_presence(output_df, mother_presence_data=None, by_pup=False):
    """
    Analyze nursing time based on mother presence status
    """
    output_df['Timestamp'] = pd.to_datetime(output_df['Timestamp'])
    output_df['Date'] = output_df['Timestamp'].dt.date
    
    mother_presence_data['Date'] = pd.to_datetime(mother_presence_data['Date']).dt.date
    
    # Remove unknown or missing statuses
    valid_statuses = ['Present', 'Partial', 'Absent']
    mother_presence_data = mother_presence_data[mother_presence_data['Mother_status'].isin(valid_statuses)]

    sampling_interval_seconds = 3
    sampling_interval_minutes = sampling_interval_seconds / 60

    if by_pup and 'Tag.ID' in output_df.columns:
        daily_nursing = output_df.groupby(['Date', 'Tag.ID']).agg({
            'Nursing_Flag': ['sum', 'count']
        }).reset_index()
        daily_nursing.columns = ['Date', 'Tag.ID', 'Nursing_Episodes', 'Total_Records']
        nursing_summary = daily_nursing.merge(mother_presence_data, on=['Date', 'Tag.ID'], how='inner')
    else:
        raise ValueError("This version requires `by_pup=True` and 'Tag.ID' in your data.")

    nursing_summary['Nursing_Time_Minutes'] = nursing_summary['Nursing_Episodes'] * sampling_interval_minutes
    nursing_summary['Nursing_Time_Hours'] = nursing_summary['Nursing_Time_Minutes'] / 60
    nursing_summary['Nursing_Time_Seconds'] = nursing_summary['Nursing_Episodes'] * sampling_interval_seconds

    print(f"Original pup-day combinations: {len(daily_nursing)}")
    print(f"Matched with mother data: {len(nursing_summary)}")
    print(f"Excluded due to no match or unknown status: {len(daily_nursing) - len(nursing_summary)}")

    return nursing_summary, daily_nursing


def create_nursing_mother_presence_plots(nursing_summary):
    """
    Create a 2-panel figure (stacked vertically):
    - Boxplot: Nursing time by mother presence
    - Grouped bar chart: Avg nursing per pup by mother presence
    """
    valid_statuses = ['Present', 'Partial', 'Absent']
    nursing_summary = nursing_summary[nursing_summary['Mother_status'].isin(valid_statuses)]
    
    status_colors = {'Present': '#36B155', 'Partial': '#F18F01', 'Absent': '#E63946'}

    # Set fonts and style
    sns.set(style='whitegrid')
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 18,
        'axes.titlesize': 18,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16,
        'figure.titlesize': 20
    })

    # Changed from (1, 2) to (2, 1) for stacked layout
    fig, axes = plt.subplots(2, 1, figsize=(12, 14))  # Adjusted figsize for vertical layout
    fig.suptitle('Nursing Behavior by Mother Presence', fontsize=20, fontweight='bold')

    # 1. Boxplot: Nursing time by mother presence (top plot)
    ax1 = axes[0]
    sns.boxplot(
        data=nursing_summary,
        x='Mother_status',
        y='Nursing_Time_Hours',
        palette=status_colors,
        ax=ax1
    )
    ax1.set_title('Distribution of Daily Nursing Time by Mother Presence')
    ax1.set_ylabel('Nursing Time (hours)')
    ax1.set_xlabel('Mother Status')

    # 2. Grouped bar chart: Avg daily nursing by pup and mother status (bottom plot)
    ax2 = axes[1]
    pup_avg = nursing_summary.groupby(['Tag.ID', 'Mother_status'])['Nursing_Time_Hours'].mean().unstack(fill_value=0)
    pup_avg = pup_avg[valid_statuses]

    # Extract just the ID prefix before underscore for x-axis
    pup_ids_simple = pup_avg.index.to_series().apply(lambda x: x.split('_')[0])
    pup_avg.index = pup_ids_simple

    pup_avg.plot(kind='bar', ax=ax2, alpha=0.9,
                 color=[status_colors[status] for status in pup_avg.columns])
    ax2.set_title('Average Daily Nursing by Pup and Mother Status')
    ax2.set_ylabel('Nursing Time (hours)')  # Added back ylabel for bottom plot
    ax2.set_xlabel('Pup')

    # Remove individual legend from subplot
    ax2.get_legend().remove()

    # Add shared legend to entire figure at the bottom center
    fig.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=3
    )

    # Adjust spacing for stacked layout
    plt.subplots_adjust(top=0.92, bottom=0.15, hspace=0.3)
    
    ax2.tick_params(axis='x', rotation=0)

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    filename = 'nursing_mother_presence_2panel_stacked.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig



# === MAIN EXECUTION ===

# Load your prediction and mother presence data
output = pd.read_csv("allpups_predictions_final.csv")
mother_data = pd.read_csv("mother_presence.csv")

print("Analyzing nursing behavior by mother presence and pup (matched only)...")
nursing_summary, daily_summary = analyze_nursing_by_mother_presence(output, mother_data, by_pup=True)

print("Creating 2-panel visualizations...")
fig = create_nursing_mother_presence_plots(nursing_summary)

# Print summary statistics
print("\n=== SUMMARY STATISTICS (Matched Data Only) ===")
overall_stats = nursing_summary.groupby('Mother_status')['Nursing_Time_Hours'].agg([
    'count', 'mean', 'std', 'min', 'max'
]).round(2)
print("Overall Summary by Mother Status:")
print(overall_stats)

print("\nSummary by Pup and Mother Status:")
pup_stats = nursing_summary.groupby(['Tag.ID', 'Mother_status'])['Nursing_Time_Hours'].agg([
    'count', 'mean', 'std'
]).round(2)
print(pup_stats)

print(f"\nTotal pups analyzed: {nursing_summary['Tag.ID'].nunique()}")
print(f"Total days analyzed: {len(nursing_summary)}")
print(f"Total nursing episodes detected: {nursing_summary['Nursing_Episodes'].sum()}")
print(f"Total nursing time: {nursing_summary['Nursing_Time_Hours'].sum():.1f} hours")

# Save detailed results
nursing_summary.to_csv("nursing_mother_analysis_matched_only.csv", index=False)
print("\n✅ Data saved to: nursing_mother_analysis_matched_only.csv")
print("✅ Visualization saved to: nursing_mother_presence_2panel.png")



############### Frequency and Duration of nursing ###################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Assuming your dataframe is called 'output' and has columns: 
# 'Timestamp', 'Nursing_Flag', 'Tag.ID' (you'll need to add Tag.ID if not present)

# Add date column for daily grouping
output['Date'] = output['Timestamp'].dt.date

# Function to calculate nursing bout durations for a single pup
def calculate_bout_durations(pup_data):
    """Calculate nursing bout durations for a single pup's data"""
    if len(pup_data) == 0 or pup_data['Nursing_Flag'].sum() == 0:
        return pd.Series(dtype=float)
    
    # Detect nursing state changes
    pup_data = pup_data.copy().sort_values('Timestamp')
    pup_data['Nursing_State_Change'] = pup_data['Nursing_Flag'].diff().ne(0)
    
    # Assign bout IDs
    pup_data['Nursing_Bout_ID'] = (pup_data['Nursing_State_Change'] & 
                                   (pup_data['Nursing_Flag'] == 1)).cumsum()
    
    # Calculate bout durations (in minutes)
    bout_durations = pup_data[pup_data['Nursing_Flag'] == 1].groupby('Nursing_Bout_ID').apply(
        lambda x: (x['Timestamp'].max() - x['Timestamp'].min()).total_seconds() / 60
    )
    
    return bout_durations

# Create summary dataset by pup
pup_summary_list = []

for pup_id in output['Tag.ID'].unique():
    pup_data = output[output['Tag.ID'] == pup_id].copy()
    
    # Calculate daily nursing frequency (percentage of time nursing per day)
    daily_nursing_freq = pup_data.groupby('Date')['Nursing_Flag'].mean() * 100
    avg_daily_freq = daily_nursing_freq.mean()
    
    # Calculate daily nursing duration (total minutes nursing per day)
    # First get bout durations for this pup
    bout_durations = calculate_bout_durations(pup_data)
    
    if len(bout_durations) > 0:
        # Map bout durations back to dates
        pup_data_nursing = pup_data[pup_data['Nursing_Flag'] == 1].copy()
        pup_data_nursing['Nursing_State_Change'] = pup_data_nursing['Nursing_Flag'].diff().ne(0)
        pup_data_nursing['Nursing_Bout_ID'] = (pup_data_nursing['Nursing_State_Change'] & 
                                               (pup_data_nursing['Nursing_Flag'] == 1)).cumsum()
        
        # Calculate daily total nursing duration
        daily_duration = pup_data_nursing.groupby(['Date', 'Nursing_Bout_ID']).apply(
            lambda x: (x['Timestamp'].max() - x['Timestamp'].min()).total_seconds() / 60
        ).groupby('Date').sum()
        
        avg_daily_duration = daily_duration.mean()
        total_bouts = len(bout_durations)
        avg_bout_duration = bout_durations.mean()
    else:
        avg_daily_duration = 0
        total_bouts = 0
        avg_bout_duration = 0
    
    # Get observation period info
    first_obs = pup_data['Date'].min()
    last_obs = pup_data['Date'].max()
    total_days = len(pup_data['Date'].unique())
    
    # Create summary record
    pup_summary = {
        'Pup_ID': pup_id,
        'Avg_Daily_Nursing_Frequency_Percent': round(avg_daily_freq, 2),
        'Avg_Daily_Nursing_Duration_Minutes': round(avg_daily_duration, 2),
        'Total_Nursing_Bouts': total_bouts,
        'Avg_Bout_Duration_Minutes': round(avg_bout_duration, 2),
        'Total_Days_Observed': total_days,
        'First_Observation': first_obs,
        'Last_Observation': last_obs
    }
    
    pup_summary_list.append(pup_summary)

# Create the final summary dataframe
pup_nursing_summary = pd.DataFrame(pup_summary_list)

# Display the results
print("Individual Pup Nursing Summary:")
print("=" * 50)
print(pup_nursing_summary.to_string(index=False))

# Save to CSV
pup_nursing_summary.to_csv('pup_nursing_summary.csv', index=False)
print(f"\nSummary saved to 'pup_nursing_summary.csv'")

# Create some visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Average daily nursing frequency by pup
axes[0, 0].bar(pup_nursing_summary['Pup_ID'], 
               pup_nursing_summary['Avg_Daily_Nursing_Frequency_Percent'])
axes[0, 0].set_title('Average Daily Nursing Frequency by Pup')
axes[0, 0].set_ylabel('Percentage of time nursing (%)')
axes[0, 0].set_xlabel('Pup ID')
axes[0, 0].tick_params(axis='x', rotation=45)

# Plot 2: Average daily nursing duration by pup
axes[0, 1].bar(pup_nursing_summary['Pup_ID'], 
               pup_nursing_summary['Avg_Daily_Nursing_Duration_Minutes'])
axes[0, 1].set_title('Average Daily Nursing Duration by Pup')
axes[0, 1].set_ylabel('Minutes nursing per day')
axes[0, 1].set_xlabel('Pup ID')
axes[0, 1].tick_params(axis='x', rotation=45)

# Plot 3: Total nursing bouts by pup
axes[1, 0].bar(pup_nursing_summary['Pup_ID'], 
               pup_nursing_summary['Total_Nursing_Bouts'])
axes[1, 0].set_title('Total Nursing Bouts by Pup')
axes[1, 0].set_ylabel('Number of nursing bouts')
axes[1, 0].set_xlabel('Pup ID')
axes[1, 0].tick_params(axis='x', rotation=45)

# Plot 4: Average bout duration by pup
axes[1, 1].bar(pup_nursing_summary['Pup_ID'], 
               pup_nursing_summary['Avg_Bout_Duration_Minutes'])
axes[1, 1].set_title('Average Nursing Bout Duration by Pup')
axes[1, 1].set_ylabel('Minutes per bout')
axes[1, 1].set_xlabel('Pup ID')
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('pup_nursing_summary_plots.png', dpi=300, bbox_inches='tight')
plt.show()

# Print some basic statistics about the pup population
print(f"\nPopulation Statistics:")
print(f"Number of pups: {len(pup_nursing_summary)}")
print(f"Average nursing frequency across all pups: {pup_nursing_summary['Avg_Daily_Nursing_Frequency_Percent'].mean():.2f}%")
print(f"Average nursing duration across all pups: {pup_nursing_summary['Avg_Daily_Nursing_Duration_Minutes'].mean():.2f} minutes/day")
print(f"Range of nursing frequencies: {pup_nursing_summary['Avg_Daily_Nursing_Frequency_Percent'].min():.2f}% - {pup_nursing_summary['Avg_Daily_Nursing_Frequency_Percent'].max():.2f}%")
print(f"Range of nursing durations: {pup_nursing_summary['Avg_Daily_Nursing_Duration_Minutes'].min():.2f} - {pup_nursing_summary['Avg_Daily_Nursing_Duration_Minutes'].max():.2f} minutes/day")



########################Evaluate time spent in each behavior################
#### ACTIVITY BUDGETS ##### THIS CAN BE FOR WHOLE GROUP #
############################################################################

# seperate vostochni pups from reef pups from the main dataframe - make two seperate dataframes to compare their acitity budgets

behavior_counts = output['Filtered_Predicted_Behavior'].value_counts().sort_index()
print(behavior_counts)

behavior_proportions = behavior_counts / behavior_counts.sum()
print(behavior_proportions)

behavior_map = {
    0: 'Sleeping',
    1: 'Nursing',
    2: 'Active',
    3: 'Inactive',
    4: 'Swimming',
    -1: 'Uncertain'
}

# Apply the map
behavior_proportions_named = behavior_proportions.rename(index=behavior_map)
print(behavior_proportions_named)

import matplotlib.pyplot as plt
#BAR
plt.figure(figsize=(8, 4))
behavior_proportions_named.plot(kind='bar', color='skyblue', edgecolor='black')
plt.ylabel("Proportion of Time")
plt.title("Activity Budget from Behavior Predictions")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("activity_budget_bar.png")

#PIE
plt.figure(figsize=(6, 6))
behavior_proportions_named.plot(kind='pie', autopct='%1.1f%%', startangle=90)
plt.ylabel("")  # Hide y-label
plt.title("Activity Budget from Behavior Predictions")
plt.tight_layout()
plt.savefig("activity_budget_pie.png")

########################################################################
#### BREAK UP ACTIVITY BUDGETS BY TIME OF DAY ####
########################################################################
# group by ids too?

output['Timestamp'] = pd.to_datetime(output['Timestamp'])  # Ensure proper format

def assign_time_of_day(ts):
    hour = ts.hour
    if 6 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 17:
        return 'Afternoon'
    elif 17 <= hour < 21:
        return 'Evening'
    else:
        return 'Night'

output['TimeOfDay'] = output['Timestamp'].apply(assign_time_of_day)

behavior_timeofday_counts = output.groupby(['TimeOfDay', 'Filtered_Predicted_Behavior']).size().unstack(fill_value=0)

#this gives the proportions of each behvaior within each day time category
behavior_timeofday_props = behavior_timeofday_counts.div(behavior_timeofday_counts.sum(axis=1), axis=0)

behavior_map = {
    0: 'Sleeping',
    1: 'Nursing',
    2: 'Active',
    3: 'Inactive',
    4: 'Swimming',
    -1: 'Uncertain'
}
behavior_timeofday_props.rename(columns=behavior_map, inplace=True)

#STACKED BAR
behavior_timeofday_props.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.ylabel("Proportion of Time")
plt.title("Activity Budget by Time of Day")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("activity_by_time_of_day.png")

############################################### SPLINE ############################################################
#output.to_csv("RawBB_predictions_with_probs.csv", index=False) # change name for all accels!

### This is just line graphs ###
# Now get the hour as a decimal (e.g., 13.5 = 1:30 PM)
output['HourDecimal'] = output['Timestamp'].dt.hour + output['Timestamp'].dt.minute / 60

# Bin time into 48 half-hour slots
output['HourBin'] = pd.cut(output['HourDecimal'], bins=np.linspace(0, 24, 49), include_lowest=True, right=False)

# Count behaviors in each time bin
behavior_hourly = output.groupby(['HourBin', 'Filtered_Predicted_Behavior']).size().unstack(fill_value=0)

# Convert counts to proportions
behavior_hourly_prop = behavior_hourly.div(behavior_hourly.sum(axis=1), axis=0)

# Midpoint of each bin for plotting
behavior_hourly_prop['HourMid'] = [interval.left + 0.25 for interval in behavior_hourly_prop.index]

import matplotlib.pyplot as plt
import seaborn as sns

behavior_map = {
    0: 'Sleeping',
    1: 'Nursing',
    2: 'Active',
    3: 'Inactive',
    4: 'Swimming',
    -1: 'Uncertain'
}

# Drop Uncertain and inactive if you want
behavior_hourly_prop = behavior_hourly_prop.drop(columns=[-1], errors='ignore')
behavior_hourly_prop = behavior_hourly_prop.drop(columns=[3], errors='ignore')

behavior_hourly_prop.rename(columns=behavior_map, inplace=True)

plt.figure(figsize=(12, 6))
for behavior in behavior_hourly_prop.columns[:-1]:  # Exclude HourMid
    sns.lineplot(
        x=behavior_hourly_prop['HourMid'],
        y=behavior_hourly_prop[behavior],
        label=behavior,
        linewidth=2
    )

plt.xlabel("Time of Day (Hour)")
plt.ylabel("Proportion of Time")
plt.title("Diel Activity Pattern (Spline Smoothed)")
plt.xticks(np.arange(0, 25, 2))
plt.ylim(0, 1)
plt.legend(title="Behavior")
plt.tight_layout()
plt.savefig("activity_spline_diel_pattern.png")


##### focus on one behavior #####

behavior_to_focus_on = 'Nursing'

# Plotting just the chosen behavior
plt.figure(figsize=(12, 6))

sns.lineplot(
    x=behavior_hourly_prop['HourMid'],
    y=behavior_hourly_prop[behavior_to_focus_on],
    label=behavior_to_focus_on,
    linewidth=2
)

plt.xlabel("Time of Day (Hour)")
plt.ylabel("Proportion of Time")
plt.title(f"Diel Activity Pattern for {behavior_to_focus_on}")
plt.xticks(np.arange(0, 25, 2))
plt.ylim(0, 0.02)
plt.legend(title="Behavior")
plt.tight_layout()
plt.savefig(f"activity_spline_{behavior_to_focus_on.lower()}_pattern.png")

########## LOWESS SMOOTHING ##########

from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.pyplot as plt
import seaborn as sns

# Drop HourMid from behaviors
behavior_columns = behavior_hourly_prop.columns.drop('HourMid')
hour_mid = behavior_hourly_prop['HourMid']

# Plot smoothed curves
plt.figure(figsize=(12, 6))

for behavior in behavior_columns:
    y = behavior_hourly_prop[behavior]
    smoothed = lowess(y, hour_mid, frac=0.3, return_sorted=True)
    
    plt.plot(smoothed[:, 0], smoothed[:, 1], label=behavior, linewidth=2)

plt.xlabel("Time of Day (Hour)")
plt.ylabel("Proportion of Time")
plt.title("Smoothed Diel Activity Patterns for All Behaviors (LOWESS)")
plt.xticks(np.arange(0, 25, 2))
plt.ylim(0, 1)
plt.legend(title="Behavior")
plt.tight_layout()
plt.savefig("activity_spline_all_behaviors_lowess.png")

## if only nursing ##
behavior_to_focus_on = 'Nursing'

# Apply LOWESS smoothing
smoothed = lowess(
    endog=behavior_hourly_prop[behavior_to_focus_on],
    exog=behavior_hourly_prop['HourMid'],
    frac=0.3  # Adjust for more/less smoothing
)

# Plot
plt.figure(figsize=(12, 6))
sns.lineplot(x=smoothed[:, 0], y=smoothed[:, 1], label=f"{behavior_to_focus_on} (Smoothed)", linewidth=2)

plt.xlabel("Time of Day (Hour)")
plt.ylabel("Proportion of Time")
plt.title(f"Smoothed Diel Activity Pattern for {behavior_to_focus_on}")
plt.xticks(np.arange(0, 25, 2))
plt.ylim(0, 0.02)
plt.legend(title="Behavior")
plt.tight_layout()
plt.savefig(f"activity_spline_{behavior_to_focus_on.lower()}_lowess.png")



########### CUBIC SPLINE ############

from scipy.interpolate import make_interp_spline
import numpy as np

plt.figure(figsize=(12, 6))

x = behavior_hourly_prop['HourMid'].values
x_smooth = np.linspace(x.min(), x.max(), 300)

for behavior in behavior_columns:
    y = behavior_hourly_prop[behavior].values
    spline = make_interp_spline(x, y, k=3)
    y_smooth = spline(x_smooth)
    plt.plot(x_smooth, y_smooth, label=behavior, linewidth=2)

plt.xlabel("Time of Day (Hour)")
plt.ylabel("Proportion of Time")
plt.title("Smoothed Diel Activity Patterns for All Behaviors (Cubic Splines)")
plt.xticks(np.arange(0, 25, 2))
plt.ylim(0, 1)
plt.legend(title="Behavior")
plt.tight_layout()
plt.savefig("activity_spline_all_behaviors_cubic.png")

############# if only nursing ############
# Clean NaNs (in case of missing bins)
df = behavior_hourly_prop[['HourMid', behavior_to_focus_on]].dropna()
x = df['HourMid'].values
y = df[behavior_to_focus_on].values

# Create spline
xnew = np.linspace(x.min(), x.max(), 300)
spl = make_interp_spline(x, y, k=3)  # k=3 for cubic
y_smooth = spl(xnew)

# Plot
plt.figure(figsize=(12, 6))
plt.plot(xnew, y_smooth, label=f"{behavior_to_focus_on} (Spline Smoothed)", linewidth=2)

plt.xlabel("Time of Day (Hour)")
plt.ylabel("Proportion of Time")
plt.title(f"Cubic Spline Diel Activity Pattern for {behavior_to_focus_on}")
plt.xticks(np.arange(0, 25, 2))
plt.ylim(0, 0.02)
plt.legend(title="Behavior")
plt.tight_layout()
plt.savefig(f"activity_spline_{behavior_to_focus_on.lower()}_cubic.png")

####### Smoothing spline ##########
from scipy.interpolate import UnivariateSpline

behaviors = behavior_hourly_prop.columns.drop('HourMid')

plt.figure(figsize=(12, 6))
XX = np.linspace(0, 24, 300)  # smooth X values for spline prediction

for behavior in behaviors:
    df = behavior_hourly_prop[['HourMid', behavior]].dropna()
    X = df['HourMid'].values
    y = df[behavior].values
    
    # Fit smoothing spline; adjust 's' for smoothness (try 0.0001 to start)
    spline = UnivariateSpline(X, y, s=0.01)
    YY = spline(XX)
    
    plt.plot(XX, YY, label=behavior, linewidth=2)
    #plt.scatter(X, y, color='gray', alpha=0.5, label="Original")

plt.xlabel("Time of Day (Hour)")
plt.ylabel("Proportion of Time")
plt.title("Smoothing Spline for All Behaviors")
plt.xticks(np.arange(0, 25, 2))
plt.ylim(0, 1)
plt.legend(title="Behavior")
plt.tight_layout()
plt.savefig("activity_spline_all_behaviors_smoothspline.png")

# Example: Smooth the 'Nursing' behavior
df = behavior_hourly_prop[['HourMid', behavior_to_focus_on]].dropna()
X = df['HourMid'].values
y = df[behavior_to_focus_on].values

# Fit smoothing spline (s = smoothing factor)
spline = UnivariateSpline(X, y, s=0.01)  # Adjust s (higher = smoother)

XX = np.linspace(0, 24, 300)
YY = spline(XX)

plt.figure(figsize=(12, 6))
plt.plot(XX, YY, label="Smoothing Spline", linewidth=2)
plt.scatter(X, y, color='gray', alpha=0.5, label="Original")
plt.xlabel("Time of Day (Hour)")
plt.ylabel("Proportion of Time")
plt.title(f"Smoothing Spline for {behavior_to_focus_on}")
plt.legend()
plt.tight_layout()
plt.savefig(f"activity_spline_{behavior_to_focus_on.lower()}_smoothspline.png")


############# GAM ################# PROBABLY NEED TO DO THIS IN R FOR THE BETTER GAM PACKAGES - thin plate in particular. R sometimes better for cyclic data.
from pygam import GAM, s
import numpy as np
import matplotlib.pyplot as plt

# Prepare the data
behavior_columns = behavior_hourly_prop.columns.drop('HourMid')
X = behavior_hourly_prop['HourMid'].values.reshape(-1, 1)

# Set up the plot
plt.figure(figsize=(12, 6))

# Fit and plot a GAM for each behavior
for behavior in behavior_columns:
    y = behavior_hourly_prop[behavior].values

    # Drop NaNs if any (some bins may be missing data for a behavior)
    mask = ~np.isnan(y)
    X_masked = X[mask]
    y_masked = y[mask]

    # Fit GAM
    gam = GAM(s(0, n_splines=20, basis= 'cp')).fit(X_masked, y_masked) # Use: basis='tp' for thin plate spline. 'cr' is cubic

    # Run gridsearch to automatically tune lam, lam is smooting parameter
    gam.gridsearch(X_masked, y_masked)

    # Predict
    XX = np.linspace(0, 24, 300).reshape(-1, 1)
    YY = gam.predict(XX)
    CI = gam.confidence_intervals(XX, width=0.95)
    # Plot
    plt.plot(XX, YY, label=behavior, linewidth=2)
    plt.fill_between(XX.ravel(), CI[:, 0], CI[:, 1], color='blue', alpha=0.2, label='95% Confidence Interval')

# Format the plot
plt.xlabel("Time of Day (Hour)")
plt.ylabel("Proportion of Time")
plt.title("GAM-Smoothed Diel Activity Patterns for All Behaviors")
plt.xticks(np.arange(0, 25, 2))
plt.ylim(0, 1)
plt.legend(title="Behavior")
plt.tight_layout()
plt.savefig("activity_spline_all_behaviors_gam.png")

####### just nursing GAM #########

behavior_columns = behavior_hourly_prop.columns.drop('HourMid')
X = behavior_hourly_prop['HourMid'].values.reshape(-1, 1)
y = behavior_hourly_prop[behavior_to_focus_on].values

# Set up the plot
plt.figure(figsize=(12, 6))
plt.clf()
# Drop NaNs if any (some bins may be missing data for a behavior)
mask = ~np.isnan(y)
X_masked = X[mask]
y_masked = y[mask]

# Fit GAM
gam = GAM(s(0, n_splines=20, basis= 'cp')).fit(X_masked, y_masked) # Use: basis='tp' for thin plate spline. 'cr' is cubic

# Run gridsearch to automatically tune lam, lam is smooting parameter
gam.gridsearch(X_masked, y_masked)

# Predict
XX = np.linspace(0, 24, 300).reshape(-1, 1)
YY = gam.predict(XX)
CI = gam.confidence_intervals(XX, width=0.95)

# Plot
plt.plot(XX, YY, label="Nursing", linewidth=2)
plt.fill_between(XX.ravel(), CI[:, 0], CI[:, 1], color='blue', alpha=0.2, label='95% Confidence Interval')

# Format the plot
plt.scatter(X, y, color='gray', alpha=0.5, label="Original")
plt.xlabel("Time of Day (Hour)")
plt.ylabel("Proportion of Time")
plt.title("GAM-Smoothed Diel Activity Patterns for Nursing")
plt.xticks(np.arange(0, 25, 2))
plt.ylim(0, 0.01)
plt.legend(title="Behavior")
plt.tight_layout()
plt.savefig("activity_spline_nursing_gam.png")


########## USING P SPLINES #################

import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.linalg import cholesky, cho_solve
import pandas as pd

def fit_pspline(x, y, num_knots=20, degree=3, penalty_order=2, lambda_value=1.0, circular=True):
    """
    Fit a penalized B-spline (P-spline) to data
    
    Parameters:
    -----------
    x : array-like
        x values (e.g., hours of day)
    y : array-like
        y values (e.g., behavioral proportions)
    num_knots : int
        Number of knots to use
    degree : int
        Degree of B-spline (typically 3 for cubic)
    penalty_order : int
        Order of difference penalty (typically 2)
    lambda_value : float
        Smoothing parameter (higher = smoother)
    circular : bool
        Whether to make the spline circular (for time-of-day data)
        
    Returns:
    --------
    predict_func : function
        Function to predict values at new x positions
    """
    from scipy.interpolate import BSpline
    
    # Clean data - remove NaN values
    mask = ~np.isnan(y)
    x_clean = np.asarray(x)[mask]
    y_clean = np.asarray(y)[mask]
    
    # Define knots
    if circular:
        # For circular data (like time of day), we need to wrap the knots
        # Period is 24 hours
        period = 24
        
        # Create knots on extended domain
        extended_range = period * 1.5  # Extend by 50%
        knots = np.linspace(-extended_range/6, period + extended_range/6, num_knots + 2 * degree - 1)
    else:
        # Regular knots
        knots = np.linspace(min(x_clean), max(x_clean), num_knots)
        # Add boundary knots
        boundary_knots = np.concatenate([
            np.repeat(knots[0], degree),
            knots,
            np.repeat(knots[-1], degree)
        ])
        knots = boundary_knots
    
    # Create B-spline basis
    n_basis = len(knots) - degree - 1
    
    # Evaluate basis at data points
    basis_matrix = np.zeros((len(x_clean), n_basis))
    for i in range(n_basis):
        # Create B-spline basis function
        coefs = np.zeros(n_basis)
        coefs[i] = 1.0
        basis_func = BSpline(knots, coefs, degree)
        basis_matrix[:, i] = basis_func(x_clean)
    
    # Create difference penalty matrix for P-spline
    D = np.zeros((n_basis - penalty_order, n_basis))
    for i in range(n_basis - penalty_order):
        for j in range(penalty_order + 1):
            D[i, i+j] = (-1)**(penalty_order-j) * np.math.comb(penalty_order, j)
    
    # Penalty matrix
    P = lambda_value * D.T @ D
    
    # Solve penalized least squares
    XtX = basis_matrix.T @ basis_matrix
    Xty = basis_matrix.T @ y_clean
    
    # Add penalty to normal equations
    penalized_XtX = XtX + P
    
    # Solve using Cholesky decomposition for stability
    try:
        c = cho_solve((cholesky(penalized_XtX, lower=True), True), Xty)
    except np.linalg.LinAlgError:
        # If Cholesky fails, use least squares solver
        c = np.linalg.lstsq(penalized_XtX, Xty, rcond=None)[0]
    
    # Create prediction function using the basis functions
    def predict_func(x_new):
        basis_pred = np.zeros((len(x_new), n_basis))
        for i in range(n_basis):
            coefs = np.zeros(n_basis)
            coefs[i] = 1.0
            basis_func = BSpline(knots, coefs, degree)
            basis_pred[:, i] = basis_func(x_new)
        return basis_pred @ c
    
    return predict_func


# Load your behavioral data (assuming behavior_hourly_prop is already defined)
# behavior_hourly_prop = pd.read_csv('your_behavior_data.csv')  # Uncomment and modify as needed

# Get list of behaviors
behaviors = behavior_hourly_prop.columns.drop('HourMid')

# Set up the plotting
plt.figure(figsize=(12, 6))
XX = np.linspace(0, 24, 300)  # smooth X values for prediction

# Fit and plot P-spline for each behavior
for behavior in behaviors:
    df = behavior_hourly_prop[['HourMid', behavior]].dropna()
    X = df['HourMid'].values
    y = df[behavior].values
    
    # Fit P-spline (adjust lambda for desired smoothness)
    predict_func = fit_pspline(X, y, num_knots=12, lambda_value=1.0, circular=True)
    YY = predict_func(XX)
    
    plt.plot(XX, YY, label=behavior, linewidth=2)

plt.xlabel("Time of Day (Hour)")
plt.ylabel("Proportion of Time")
plt.title("Penalized B-Spline for All Behaviors")
plt.xticks(np.arange(0, 25, 2))
plt.ylim(0, 1)
plt.legend(title="Behavior")
plt.tight_layout()
plt.savefig("activity_pspline_all_behaviors.png")

# Focus on a specific behavior (replace 'Nursing' with your behavior_to_focus_on)
behavior_to_focus_on = 'Nursing'  # Modify based on your data
df = behavior_hourly_prop[['HourMid', behavior_to_focus_on]].dropna()
X = df['HourMid'].values
y = df[behavior_to_focus_on].values

# Fit P-spline with different lambda values to show effect
lambdas = [0.1, 1.0, 10.0]
plt.figure(figsize=(12, 8))

# Plot original data
plt.scatter(X, y, color='gray', alpha=0.5, label="Original")

for lam in lambdas:
    predict_func = fit_pspline(X, y, lambda_value=lam, circular=True)
    YY = predict_func(XX)
    plt.plot(XX, YY, label=f"λ = {lam}", linewidth=2)

plt.xlabel("Time of Day (Hour)")
plt.ylabel("Proportion of Time")
plt.title(f"P-Spline Smoothing for {behavior_to_focus_on} with Different Penalties")
plt.legend()
plt.tight_layout()
plt.savefig(f"activity_pspline_{behavior_to_focus_on.lower()}_comparison.png")

# Find optimal lambda using cross-validation
def find_optimal_lambda(x, y, lambda_range=np.logspace(-2, 2, 10), k_folds=5):
    """Find optimal smoothing parameter using k-fold cross-validation"""
    from sklearn.model_selection import KFold
    
    # Initialize
    cv_errors = np.zeros((len(lambda_range), k_folds))
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    # Clean data
    mask = ~np.isnan(y)
    x_clean = np.asarray(x)[mask]
    y_clean = np.asarray(y)[mask]
    
    # Cross-validation
    for i, lam in enumerate(lambda_range):
        for j, (train_idx, test_idx) in enumerate(kf.split(x_clean)):
            x_train, y_train = x_clean[train_idx], y_clean[train_idx]
            x_test, y_test = x_clean[test_idx], y_clean[test_idx]
            
            predict_func = fit_pspline(x_train, y_train, lambda_value=lam, circular=True)
            y_pred = predict_func(x_test)
            
            # Calculate mean squared error
            cv_errors[i, j] = np.mean((y_test - y_pred)**2)
    
    # Average error across folds
    mean_cv_errors = np.mean(cv_errors, axis=1)
    
    # Return lambda with minimum error
    best_idx = np.argmin(mean_cv_errors)
    best_lambda = lambda_range[best_idx]
    
    return best_lambda, mean_cv_errors

# Find optimal lambda for the focused behavior
lambda_range = np.logspace(-2, 2, 10)
best_lambda, cv_errors = find_optimal_lambda(X, y, lambda_range)

print(f"Best lambda for {behavior_to_focus_on}: {best_lambda}")

# Plot CV results
plt.figure(figsize=(10, 5))
plt.semilogx(lambda_range, cv_errors, '-o')
plt.axvline(x=best_lambda, color='red', linestyle='--', label=f'Best λ = {best_lambda:.3f}')
plt.xlabel('Lambda (log scale)')
plt.ylabel('Cross-Validation MSE')
plt.title(f'Cross-Validation for {behavior_to_focus_on} P-Spline')
plt.legend()
plt.tight_layout()
plt.savefig(f"pspline_cv_{behavior_to_focus_on.lower()}.png")

# Create final plot with optimal lambda
predict_func = fit_pspline(X, y, lambda_value=best_lambda, circular=True)
YY = predict_func(XX)

plt.figure(figsize=(12, 6))
plt.plot(XX, YY, label="P-Spline", linewidth=2)
plt.scatter(X, y, color='gray', alpha=0.5, label="Original")
plt.xlabel("Time of Day (Hour)")
plt.ylabel("Proportion of Time")
plt.title(f"Optimal P-Spline for {behavior_to_focus_on} (λ = {best_lambda:.3f})")
plt.legend()
plt.tight_layout()
plt.savefig(f"activity_pspline_{behavior_to_focus_on.lower()}_optimal.png")

############## P spline in a GAM Framework ###############
# move to R



################################# histograms for ADC values #######################################################
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
Raw = pd.read_csv("FinalBB_comp_fixed_raw.csv")

# Extract conductivity data
conductivity = Raw['ADC..raw.'].copy()

# Create figure with multiple plots to analyze ADC distribution
plt.figure(figsize=(15, 12))

# Plot 1: Histogram of all ADC values
plt.subplot(2, 2, 1)
plt.hist(conductivity, bins=50, color='skyblue', edgecolor='black')
plt.title('Histogram of All ADC Values')
plt.xlabel('ADC Value')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

# Plot 2: Histogram zoomed on lower values (0-100)
plt.subplot(2, 2, 2)
plt.hist(conductivity[conductivity <= 100], bins=50, color='lightgreen', edgecolor='black')
plt.title('Histogram of ADC Values (0-100 range)')
plt.xlabel('ADC Value')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

# Plot 3: KDE plot for density estimation
plt.subplot(2, 2, 3)
sns.kdeplot(conductivity, fill=True)
plt.title('Density Distribution of ADC Values')
plt.xlabel('ADC Value')
plt.ylabel('Density')
plt.grid(True, alpha=0.3)

# Plot 4: Box plot
plt.subplot(2, 2, 4)
sns.boxplot(x=conductivity)
plt.title('Box Plot of ADC Values')
plt.xlabel('ADC Value')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('adc_distribution.png')
plt.close()


##########################################################3

# Add this diagnostic code before your main analysis to check the calculations

def diagnose_nursing_calculations(output_df):
    """
    Diagnostic function to check nursing time calculations
    """
    print("=== NURSING TIME CALCULATION DIAGNOSTICS ===")
    
    # Convert timestamp to datetime if not already
    output_df['Timestamp'] = pd.to_datetime(output_df['Timestamp'])
    output_df['Date'] = output_df['Timestamp'].dt.date
    
    # Sample one pup and one day for detailed analysis
    sample_pup = output_df['Tag.ID'].iloc[0]
    sample_date = output_df['Date'].iloc[0]
    
    sample_data = output_df[(output_df['Tag.ID'] == sample_pup) & 
                           (output_df['Date'] == sample_date)]
    
    print(f"\nSample analysis for Pup {sample_pup} on {sample_date}:")
    print(f"Total records for this pup-day: {len(sample_data)}")
    print(f"Records with Nursing_Flag = 1: {sample_data['Nursing_Flag'].sum()}")
    print(f"Records with Nursing_Flag = 0: {(sample_data['Nursing_Flag'] == 0).sum()}")
    
    # Calculate time spans
    total_seconds = len(sample_data) * 3  # 3-second bins
    nursing_seconds = sample_data['Nursing_Flag'].sum() * 3
    nursing_minutes = nursing_seconds / 60
    nursing_hours = nursing_minutes / 60
    
    print(f"\nTime calculations:")
    print(f"Total observation time: {total_seconds} seconds = {total_seconds/3600:.1f} hours")
    print(f"Nursing time: {nursing_seconds} seconds = {nursing_minutes:.1f} minutes = {nursing_hours:.2f} hours")
    print(f"Nursing percentage: {(nursing_seconds/total_seconds)*100:.1f}%")
    
    # Overall statistics
    print(f"\n=== OVERALL DATASET STATISTICS ===")
    daily_nursing = output_df.groupby(['Date', 'Tag.ID']).agg({
        'Nursing_Flag': ['sum', 'count']
    }).reset_index()
    daily_nursing.columns = ['Date', 'Tag.ID', 'Nursing_Episodes', 'Total_Records']
    
    # Calculate times
    daily_nursing['Total_Hours'] = daily_nursing['Total_Records'] * 3 / 3600
    daily_nursing['Nursing_Hours'] = daily_nursing['Nursing_Episodes'] * 3 / 3600
    daily_nursing['Nursing_Percentage'] = (daily_nursing['Nursing_Episodes'] / daily_nursing['Total_Records']) * 100
    
    print(f"Number of pups: {output_df['Tag.ID'].nunique()}")
    print(f"Number of unique dates: {output_df['Date'].nunique()}")
    print(f"Total pup-day combinations: {len(daily_nursing)}")
    
    print(f"\nDaily nursing hours statistics:")
    print(f"Mean: {daily_nursing['Nursing_Hours'].mean():.2f} hours")
    print(f"Median: {daily_nursing['Nursing_Hours'].median():.2f} hours")
    print(f"Min: {daily_nursing['Nursing_Hours'].min():.2f} hours")
    print(f"Max: {daily_nursing['Nursing_Hours'].max():.2f} hours")
    print(f"Std: {daily_nursing['Nursing_Hours'].std():.2f} hours")
    
    print(f"\nDaily nursing percentage statistics:")
    print(f"Mean: {daily_nursing['Nursing_Percentage'].mean():.1f}%")
    print(f"Median: {daily_nursing['Nursing_Percentage'].median():.1f}%")
    print(f"Min: {daily_nursing['Nursing_Percentage'].min():.1f}%")
    print(f"Max: {daily_nursing['Nursing_Percentage'].max():.1f}%")
    
    print(f"\nDaily observation time statistics:")
    print(f"Mean observation time per pup-day: {daily_nursing['Total_Hours'].mean():.1f} hours")
    print(f"Min observation time per pup-day: {daily_nursing['Total_Hours'].min():.1f} hours")
    print(f"Max observation time per pup-day: {daily_nursing['Total_Hours'].max():.1f} hours")
    
    # Check for potential issues
    print(f"\n=== POTENTIAL ISSUES TO CHECK ===")
    
    # Check if any days have very little data
    short_days = daily_nursing[daily_nursing['Total_Hours'] < 12]  # Less than 12 hours of data
    if len(short_days) > 0:
        print(f"⚠️  {len(short_days)} pup-days have less than 12 hours of observation time")
        print("This might explain low nursing hours if data collection was incomplete")
    
    # Check if nursing flags seem reasonable
    zero_nursing_days = daily_nursing[daily_nursing['Nursing_Episodes'] == 0]
    if len(zero_nursing_days) > 0:
        print(f"⚠️  {len(zero_nursing_days)} pup-days have zero nursing episodes")
    
    high_nursing_days = daily_nursing[daily_nursing['Nursing_Percentage'] > 50]
    if len(high_nursing_days) > 0:
        print(f"⚠️  {len(high_nursing_days)} pup-days have >50% nursing time (might be too high)")
    
    return daily_nursing

# Run diagnostics
output = pd.read_csv("allpups_predictions_final.csv")
diagnostic_results = diagnose_nursing_calculations(output)

# Then run your normal analysis
mother_data = pd.read_csv("mother_presence.csv")
nursing_summary, daily_summary = analyze_nursing_by_mother_presence(output, mother_data, by_pup=True)