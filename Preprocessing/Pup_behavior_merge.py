import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('Pup_Behavior_Annotated.csv') # add in behavior data frame
#pd.set_option('display.max_rows', None)

# Force conversion to datetime with error handling
df['Behavior.st.UTC'] = pd.to_datetime(df['Behavior.st.UTC'], errors='coerce')
df['Behavior.end.UTC'] = pd.to_datetime(df['Behavior.end.UTC'], errors='coerce')

# Check for NaT (Not a Time) values that resulted from the conversion
#nat_mask = df['Behavior.st.UTC'].isna() | df['Behavior.end.UTC'].isna()
#print(f"Found {nat_mask.sum()} rows with invalid datetime values")
#if nat_mask.sum() > 0:
#    print(df[nat_mask][['Flipper.ID', 'Accel.ID', 'Behavior.st.UTC', 'Behavior.end.UTC']].head())

## Need to combine this with the ACC data : see below ##
## merge with raw data and then pull out the behaviors and then group by behvaior and then concatonate back together to train

# Convert the 'Animal.ID' and 'Behavior' columns to categorical data type
df = df.rename(columns={'Flipper ID': 'Flipper.ID', 'Accelerometer ID (letter and number)': 'Accel.ID'})
df['Flipper.ID'] = df['Flipper.ID'].astype('category')
df['Accel.ID'] = df['Accel.ID'].astype('category')
df['Behavior'] = df['Behavior'].astype('category')
#df.head()

df = df.drop(['Location','Mother flipper ID', 'Audio recording number','Behavior.st','Behavior.end','Notes/Comments'],axis=1)
#print(df)
df['Behavior'].value_counts()
## create groups of behaviors - fixing typos
def categorize_behavior(behavior):
    original_behavior = behavior
    behavior = behavior.lower()
    if 'nursing' in behavior or 'nusing' in behavior:
        return 'Nursing'
    elif 'rest' in behavior or behavior in ['asleep', 'sleeping']:
        return 'Resting'
    elif 'inactive' in behavior:
        return 'Inactive'
    elif 'walk' in behavior or 'run' in behavior or 'explor' in behavior or 'active' in behavior or 'Active' in behavior:
        return 'Active'
    elif 'groom' in behavior or 'scratch' in behavior or 'bit' in behavior or 'rub' in behavior:
        return 'Grooming'
    elif 'interact' in behavior or "agression" in behavior:
        return 'Interacting'
    elif 'vocal' in behavior:
        return 'Vocalizing'
    else:
        return original_behavior

df['Behavior'] = df['Behavior'].apply(categorize_behavior)
df['Behavior'].value_counts() #before extending the dataframes

# Delete certain rows
#df = df.drop(df[df['Behavior'] == 'Resting'].index)
#df = df.drop(df[df['Behavior'] == 'Unlisted behavior'].index)

# Your specified behavior order
behaviors = ['Resting', 'Nursing', 'Active', 'Grooming', 'Inactive', 'Vocalizing', 'shaking', 'Interacting']

# Create a custom mapping dictionary where each behavior maps to its index in the behaviors list
behavior_mapping = {behavior: idx for idx, behavior in enumerate(behaviors)}

df['Behavior_label'] = df['Behavior'].apply(lambda x: behavior_mapping[x])

# def custom_transform(x):
#     return behavior_mapping[x],
# df['Behavior_label'] = df['Behavior'].apply(custom_transform)

# For inverse transformation (if needed), create a reverse mapping
#reverse_mapping = {v: k for k, v in behavior_mapping.items()}
#def custom_inverse_transform(x):
#    return reverse_mapping[x]

# Verify the mapping
#print("Behavior Mapping:")
#for behavior, label in behavior_mapping.items():
#    print(f"{behavior}: {label}")

# Check the results
#print("\nSample of transformed data:")
#print(df[['Behavior', 'Behavior_label']].head())

# Verify no NaN values
#nan_behavior = df['Behavior'].isna()
#print("\nRows with NaN behaviors:")
#print(df[nan_behavior])

# Check value counts to verify distribution
#print("\nValue counts of behavior labels:")
#print(df['Behavior_label'].value_counts().sort_index())

# Check for NaN values in the 'Behavior' column
#nan_behavior = df['Behavior'].isna()
#print(df[nan_behavior])
#df.head()

# Assuming you have a DataFrame called 'behavior_df' (df) with columns 'animal.id', 'behavior-type', 'behavior', 'start_time', and 'end_time'
#df = pd.DataFrame({'animal.id': [1, 2, 3],
                             #'behavior-type': ['locomotion', 'resting', 'locomotion'],
                             #'behavior': ['walking', 'sitting', 'running'],
                             #'start_time': ['2023-03-29 10:00:00', '2023-03-29 10:10:00', '2023-03-29 10:20:00'],
                             #'end_time': ['2023-03-29 10:09:59', '2023-03-29 10:19:59', '2023-03-29 10:29:59']})



# Create an empty list to store the expanded rows
expanded_rows = []

# Loop through each row in the original DataFrame
for _, row in df.iterrows():
    # Get the start and end times for this behavior
    start_time = row['Behavior.st.UTC']
    end_time = row['Behavior.end.UTC']
    
    # Create a range of timestamps for each second the behavior occurred
    timestamps = pd.date_range(start_time, end_time, freq='s')
    
    # Add a new row to the expanded_rows list for each timestamp
    for timestamp in timestamps:
        expanded_rows.append({'Flipper.ID': row['Flipper.ID'],
                              'Accel.ID': row['Accel.ID'],
                               'Behavior': row['Behavior_label'],
                               'Time_UTC': timestamp})

# Create a new DataFrame from the expanded_rows list
expanded_df = pd.DataFrame(expanded_rows)

# Print the new DataFrame
#print(expanded_df)

# Check the duration of each behavior in the original df
# useful if something does not seem right. Ex. A very large dataset that equals more hours than you were observing.

#df['duration'] = (pd.to_datetime(df['Behavior.end.UTC']) - pd.to_datetime(df['Behavior.st.UTC'])).dt.total_seconds()

# Look at the summary statistics
#print("Duration statistics (in seconds):")
#print(df['duration'].describe())

# Check for any suspiciously long durations
#print("\nLongest 5 durations:")
#print(df.nlargest(5, 'duration')[['Flipper.ID', 'Behavior_label', 'duration', 'Behavior.st.UTC', 'Behavior.end.UTC']])

#print(expanded_df['Behavior'].value_counts().sort_index())
#print(expanded_df)


#############################################################################
## FIXING CODE FOR MISSING ACCEL IDS ##
#############################################################################

# Check for problematic time ranges
#time_issues = df[df['Behavior.end.UTC'] <= df['Behavior.st.UTC']]
#print(f"Found {len(time_issues)} rows where end time <= start time")
#pd.set_option('display.max_columns', None)  # Show all columns
#display_cols = ['Accel.ID', 'Behavior', 'Behavior.st.UTC', 'Behavior.end.UTC']
#print(time_issues[display_cols])

# # Check which Accel.IDs exist in original dataset
# original_accel_ids = set(df['Accel.ID'])
# print(f"Original dataset has {len(original_accel_ids)} unique Accel.IDs")
# print(f"All oriingal Accel.IDs: {original_accel_ids}")

# # After expansion, check which Accel.IDs exist
# expanded_accel_ids = set(expanded_df['Accel.ID'])
# print(f"Expanded dataset has {len(expanded_accel_ids)} unique Accel.IDs")

# # Find missing Accel.IDs
# missing_accel_ids = original_accel_ids - expanded_accel_ids
# print(f"Missing Accel.IDs: {missing_accel_ids}")

# # For each missing ID, check its data in the original dataframe
# if missing_accel_ids:
#     for accel_id in missing_accel_ids:
#         rows = df[df['Accel.ID'] == accel_id]
#         print(f"\nChecking data for Accel.ID {accel_id}, found {len(rows)} rows:")
#         for _, row in rows.iterrows():
#             time_diff = (row['Behavior.end.UTC'] - row['Behavior.st.UTC']).total_seconds()
#             print(f"  Behavior: {row['Behavior_label']}, Start: {row['Behavior.st.UTC']}, End: {row['Behavior.end.UTC']}, Duration: {time_diff} seconds")


### FIX ABOVE STUFF, delete comments, AND RESAVE PUP_BEHAVIOR_ANNOTATED TO SSH FIRST

# Save to CSV
expanded_df.to_csv('expanded_behaviors.csv', index=False)

########################## START HERE - TO READ IN EXPANDED CSV #######################################

## Code for single file - made functions below off of this code ##


# If your Time_UTC column needs to be converted back to datetime
# expanded_df['Time_UTC'] = pd.to_datetime(expanded_df['Time_UTC'])

# ## merge with raw accel data

# ## make sure raw data is in table format ##
#df2 = pd.read_csv('/home/dio3/williamslab/SarahKerr/AccelRaw/C_0000_S2.csv', delimiter=";") # make sure it is the right animal!
#print(df2.head())  # Check the first few rows
#print(df2.info())  # Check the column types and data structure

# # First remove the milliseconds with str.replace
# df2['Timestamp'] = df2['Timestamp'].str.replace(r'\.\d+', '', regex=True)
# # Convert back to datetime
# df2['Timestamp'] = pd.to_datetime(df2['Timestamp'])
# # Now you can use dt.strftime
# df2['Timestamp'] = df2['Timestamp'].dt.strftime('%m/%d/%Y %H:%M:%S')
# df2.head()

# df2['X'] = pd.to_numeric(df2['X'])
# df2['Y'] = pd.to_numeric(df2['Y'])
# df2['Z'] = pd.to_numeric(df2['Z'])
# df2['Temp. (°C)'] = pd.to_numeric(df2['Temp. (°C)'])
# df2['Press. (mBar)'] = pd.to_numeric(df2['Press. (mBar)'])
# df2['ADC (raw)'] = pd.to_numeric(df2['ADC (raw)'])

# df2 = df2.drop(['Batt. V. (V)', 'Metadata'], axis = 1)
# df2.head()

# csv_file_path = "C_raw.csv"

# df2.to_csv(csv_file_path, index=False)

# ### Add swimming behavior ###
# # trying to add swimming as a new behavior to list anytime the conductivity was lower than 250
# ### maybe just calculate this after ###

# # def add_swimming_behavior(df2, wet_threshold=250):
    
# #     processed_df2 = df2.copy()
    
# #     # Initialize Behavior column
# #     processed_df2['Behavior'] = 'dry'
    
# #     # Mark swimming periods where ADC is below threshold
# #     swimming_mask = processed_df2['ADC (raw)'].notna() & (processed_df2['ADC (raw)'] < wet_threshold)
# #     processed_df2.loc[swimming_mask, 'Behavior'] = 'Swimming'
    
# #     return processed_df2


# # # Apply swimming behavior detection
# # df2 = add_swimming_behavior(df2, wet_threshold=250)
# # df2.head()

# # ## combine behavior df with raw accelerometery df2 ## 
# expanded_df['Behavior_UTC'] = pd.to_datetime(expanded_df['Behavior_UTC'])
# expanded_df['Behavior_UTC'] = expanded_df['Behavior_UTC'].dt.strftime('%m/%d/%Y %H:%M:%S')
# expanded_df = pd.DataFrame(expanded_df)

# # df2['Behavior_UTC'] = pd.to_datetime(df2['Timestamp'])
# # df2['Behavior_UTC'] = df2['Behavior_UTC'].dt.strftime('%m/%d/%Y %H:%M:%S')
# # df2.head()

# # # Separate swimming rows not in expanded_df 
# # swimming_rows = df2[
# #     (df2['Behavior'] == 'Swimming') & 
# #     (~df2['Behavior_UTC'].isin(expanded_df['Behavior_UTC']))
# # ]

# # # Merge with expanded_df and append swimming rows
# # merged_df = pd.merge(
# #     expanded_df, 
# #     df2, 
# #     on='Behavior_UTC', 
# #     how='outer'
# # )

# # If you want to ensure swimming rows are at the end
# #merged_df = merged_df.sort_values(by='Behavior', na_position='last')

# merged_df = pd.merge(expanded_df, df2, on='Behavior_UTC') # also on correct accel ID though...delete non matched rows?

# pd.set_option('display.max_columns', None)
# print(merged_df)

# columns_to_delete =['Timestamp'] # include all columns to delete in total between both dfs.
# merged_df = merged_df.drop(columns=columns_to_delete)

# print(merged_df)

# csv_file_path = "C_merged.csv"

# merged_df.to_csv(csv_file_path, index=False)


### More efficient way to process all raw accelerometer dfs at once ###

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import os
from glob import glob

expanded_df = ('expanded_behaviors.csv')
#expanded_df.head()

#pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', None)
#print(expanded_df)

def process_accelerometer_data(acc_file_path, behavior_df):
    """
    Process a single accelerometer file and merge it with behavior data
    Ensures all 25 rows/second are included with corresponding behavior label
    
    Parameters:
    acc_file_path (str): Path to the accelerometer CSV file
    behavior_df (pd.DataFrame): DataFrame containing behavior data
    
    Returns:
    pd.DataFrame: Processed DataFrame with behavior-matched accelerometer data
    """
    # Extract accelerometer letter from filename
    accel_letter = os.path.basename(acc_file_path).split('_')[0]
    
    # Filter behavior DataFrame to only include rows with matching Accelerometer ID
    # Create a copy of the behavior DataFrame to avoid SettingWithCopyWarning
    filtered_behavior_df = behavior_df[behavior_df['Accel.ID'] == accel_letter].copy()

    # Convert behavior timestamps to datetime
    filtered_behavior_df['Time_UTC'] = pd.to_datetime(filtered_behavior_df['Time_UTC'])
    
    # Read accelerometer data
    df2 = pd.read_csv(acc_file_path, delimiter=";", low_memory=False) # for handling mixed data types
    
    # Process timestamp
    df2['Timestamp'] = pd.to_datetime(
    df2['Timestamp'], 
    format='%m/%d/%Y %H:%M:%S.%f'
            ).dt.floor('s')  # This removes milliseconds, keeping only seconds
    
    # Convert numeric columns
    numeric_columns = ['X', 'Y', 'Z', 'Temp. (°C)', 'Press. (mBar)', 'ADC (raw)', 'Batt. V. (V)']
    for col in numeric_columns:
        if col in df2.columns:
            df2[col] = pd.to_numeric(df2[col], errors='coerce')
    
    # Drop unnecessary columns
        columns_to_drop = ['Metadata', 'Tag ID', 'Batt. V. (V)']
        df2 = df2.drop(columns=[col for col in columns_to_drop if col in df2.columns], axis=1)
    

    # Sort DataFrames by timestamp
        filtered_behavior_df.sort_values('Time_UTC', inplace=True)
        df2.sort_values('Timestamp', inplace=True)


    merged_df = df2.merge(filtered_behavior_df, left_on='Timestamp', right_on='Time_UTC', how='right') # how='right' preserves all data in behavior df. 
    # how='left' would preserve accel data and how= 'inner' would only perserve direct matches (may also work). How = ' outer' perserves all rows in both dfs.
    
    return merged_df

def batch_process_accelerometer_files(accelerometer_dir, behavior_df, output_dir=None):
    """
    Batch process all accelerometer files in a directory
    
    Parameters:
    accelerometer_dir (str): Directory containing accelerometer CSV files
    behavior_df (pd.DataFrame): DataFrame containing behavior data
    output_dir (str, optional): Directory to save processed files
    
    Returns:
    list: List of processed DataFrames
    """
    # Find all CSV files in the directory
    acc_files = glob(os.path.join(accelerometer_dir, '*.csv'))
    
    # Sort files to ensure consistent processing
    acc_files.sort()
    
    # List to store processed DataFrames
    processed_dfs = []
    
    # Process each file
    for file_path in acc_files:
        try:
            # Extract filename for potential output naming
            filename = os.path.basename(file_path)
            base_filename = os.path.splitext(filename)[0]
            
            # Process the file
            processed_df = process_accelerometer_data(file_path, behavior_df)
            
            # Create output filename
            output_filename = f'processed_{base_filename}.csv'
            output_path = os.path.join(output_dir, output_filename)
            
            # Save the processed DataFrame
            processed_df.to_csv(output_path, index=False)
            print(f"Processed and saved: {output_path}")
            
            # Store the output file path
            processed_dfs.append(output_path)
        
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    return processed_dfs

# Specify directories and behavior DataFrame
accelerometer_dir = '/home/dio3/williamslab/SarahKerr/AccelRaw/'
output_dir = '/home/dio3/williamslab/SarahKerr/processed_csvs/'
behavior_df = expanded_df  # Your existing behavior DataFrame

# Batch process files
processed_dataframes = batch_process_accelerometer_files(
    accelerometer_dir,
    behavior_df, 
    output_dir
)

# Optional: print out all processed file paths
#print("\nProcessed Files:")
#for path in processed_dataframes:
#    print(path)


# If you want to concatenate all processed DataFrames
def concatenate_processed_files(processed_dir):
    """
    Concatenate all processed accelerometer CSV files into a single DataFrame
    
    Parameters:
    processed_dir (str): Directory containing processed CSV files
    
    Returns:
    pd.DataFrame: Concatenated DataFrame of all processed files
    """
    # Find all processed CSV files in the directory
    processed_files = glob(os.path.join(processed_dir, 'processed_*.csv'))
    
    # Sort files to ensure consistent order
    processed_files.sort()
    
    # Read and concatenate all files
    concatenated_df = pd.concat(
        (pd.read_csv(file) for file in processed_files), 
        ignore_index=True
    )
    
    print(f"Concatenated {len(processed_files)} files")
    print(f"Resulting DataFrame shape: {concatenated_df.shape}")
    
    return concatenated_df


# Concatenate all processed files
final_merged_df = concatenate_processed_files(output_dir)

# Save the concatenated DataFrame if desired
final_merged_df.to_csv('/home/dio3/williamslab/SarahKerr/processed_csvs/FinalMergedAccelerometer.csv', index=False)

## Then clean up that final merged df before sending to R preprocessing (NFSP_adaptivewindow.R) ##
