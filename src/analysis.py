import os
import numpy as np
import h5py
import pandas as pd
import shutil
import glob
import time
import nibabel as nib
import seaborn as sns
from statsmodels.graphics.functional import rainbowplot
from scipy.interpolate import make_interp_spline
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from itertools import combinations
import nibabel as nib

def read_ANTs_transform(transform_path):
    """
    Read an ANTs transform file and return the affine matrix.
    """
     # Open the HDF5 file
    with h5py.File(transform_path, 'r') as h5_file:
        # Extract TransformType and TransformParameters from TransformGroup/1
        transform_type = h5_file['TransformGroup/1/TransformType'][()]
        transform_parameters = h5_file['TransformGroup/1/TransformParameters'][()]
        transform_fixed_parameters = h5_file['TransformGroup/1/TransformFixedParameters'][()]

        # Reshape the transform parameters to get the affine matrix (3x3)
        affine_matrix = transform_parameters[:9].reshape((3, 3))
        translation_vector = transform_parameters[9:]
        

        return affine_matrix, translation_vector

def extract_scaling_and_shearing(affine_matrix):
    # Ensure affine_matrix is a 3x3 matrix
    assert affine_matrix.shape == (3, 3), "The affine matrix must be 3x3."

    # Step 1: Extract scaling factors
    # Scaling factors are the lengths of the column vectors of the affine matrix
    scaling_factors = np.linalg.norm(affine_matrix, axis=0)

    # Step 2: Normalize the affine matrix by the scaling factors to extract shear and rotation
    normalized_matrix = affine_matrix / scaling_factors

    # Step 3: Extract shearing components
    # The upper triangular part of the normalized matrix represents shearing
    shear_xy = normalized_matrix[0, 1]  # Shear between X and Y
    shear_xz = normalized_matrix[0, 2]  # Shear between X and Z
    shear_yz = normalized_matrix[1, 2]  # Shear between Y and Z

    shear_factors = [shear_xy, shear_xz, shear_yz]

    return scaling_factors, shear_factors

def process_csv_and_calculate_scaling_factors(csv_file_path):
    """
    Processes the CSV file, calculates the scaling factors for each transform file,
    and stores the result in new columns for scaling factors.
    
    Parameters:
        csv_file_path (str): Path to the input CSV file.
    """
    # Load the CSV file
    df = pd.read_csv(csv_file_path)

    # Iterate through each row and calculate/store the scaling factors
    for idx, row in df.iterrows():
        transform_path = row['transform_path']
        scan_id = row['scan_id']
        # Check if transform_path is NaN (true for each middle scan for trio)
        if pd.isna(transform_path):
            # Set scaling factors to 1.0 when transform_path is NaN
            df.at[idx, 'scaling_x'] = 1.0
            df.at[idx, 'scaling_y'] = 1.0
            df.at[idx, 'scaling_z'] = 1.0
            df.at[idx, 'scaling_avg'] = 1.0
            continue
        
        # print(transform_path)
        # rel_transform_path = f"{rel_path}/{transform_path[6:]}"  # Path to the ANTs transform file

        try:
            # Read the affine matrix from the ANTs transform
            affine_matrix, _ = read_ANTs_transform(transform_path)
            
            # Extract scaling factors
            scaling_factors, _ = extract_scaling_and_shearing(affine_matrix)
        except Exception as e:
            print(f"Error processing transform for scan_id {scan_id}: {e}")
            scaling_factors = [np.nan, np.nan, np.nan]  # If there's an error, store NaN

        # Add the scaling factors to the DataFrame
        df.at[idx, 'scaling_x'] = scaling_factors[0]
        df.at[idx, 'scaling_y'] = scaling_factors[1]
        df.at[idx, 'scaling_z'] = scaling_factors[2]
        # Calculate geometric mean of scaling factors
        df.at[idx, 'scaling_avg'] = (scaling_factors[0] * scaling_factors[1] * scaling_factors[2]) ** (1/3)

    # Save the updated DataFrame with the new columns for scaling factors
    df.to_csv(csv_file_path, index=False)

def save_transform_paths_CP(csv_file_path, rel_path, transfo_type):
    """
    Processes scan pairs from the input CSV, stores paths to transformation files for each pair,
    and updates the CSV with the transformation paths on the first and last lines of each trio.
    
    Parameters:
        csv_file_path (str): Path to the input CSV file.
    """
    # Load the CSV file
    df = pd.read_csv(csv_file_path)

    processed_pairs = set()  # Track processed scan pairs
    saved_paths_for_pairs = {}  # Store previous paths for registered pairs

    # Iterate through each subject in the DataFrame
    for sub_id, group in df.groupby('sub_id_bids'):
        # Since the data is grouped by subject, each group represents a subject with multiple trios
        trios = [group.iloc[i:i + 3] for i in range(0, len(group), 3)]
        
        for trio in trios:
            # Extract the paths for the trio
            path_1 = trio.iloc[0]['path']
            path_2 = trio.iloc[1]['path']  # This is the reference
            path_3 = trio.iloc[2]['path']

            trio_id = trio.iloc[1]['trio_id']  # Using trio_id from the second scan

            # Create tuples to represent the scan pairs (scan_1 -> scan_2 and scan_3 -> scan_2)
            pair_1_to_2 = (trio.iloc[0]['scan_id'], trio.iloc[1]['scan_id'])
            pair_3_to_2 = (trio.iloc[2]['scan_id'], trio.iloc[1]['scan_id'])

            # Define paths for the transformation files
            transform_1_to_2 = f'{rel_path}/{sub_id}/{trio_id}/{transfo_type}_mov2fix_{pair_1_to_2[0]}_{pair_1_to_2[1]}InverseComposite.h5'
            transform_3_to_2 = f'{rel_path}/{sub_id}/{trio_id}/{transfo_type}_mov2fix_{pair_3_to_2[0]}_{pair_3_to_2[1]}InverseComposite.h5'

            ### Handle pair_1_to_2 (scan_1 -> scan_2)
            if pair_1_to_2 not in saved_paths_for_pairs:
                # If the pair has not been processed yet, save the transformation path
                saved_paths_for_pairs[pair_1_to_2] = transform_1_to_2

                # Add the transform path to the DataFrame (on the first row of the trio)
                df.loc[trio.index[0], 'transform_path'] = transform_1_to_2
                print(f"Storing transform path for scan {pair_1_to_2[0]} to {pair_1_to_2[1]}: {transform_1_to_2}")

                # Add the pair to the processed set
                processed_pairs.add(pair_1_to_2)
            else:
                # If already processed, reuse the transformation file from the previous registration
                previous_transform_1_to_2 = saved_paths_for_pairs[pair_1_to_2]
                df.loc[trio.index[0], 'transform_path'] = previous_transform_1_to_2
                print(f"Reusing existing transform path for scan {pair_1_to_2[0]} to {pair_1_to_2[1]}")

            ### Handle pair_3_to_2 (scan_3 -> scan_2)
            if pair_3_to_2 not in saved_paths_for_pairs:
                # If the pair has not been processed yet, save the transformation path
                saved_paths_for_pairs[pair_3_to_2] = transform_3_to_2

                # Add the transform path to the DataFrame (on the third row of the trio)
                df.loc[trio.index[2], 'transform_path'] = transform_3_to_2
                print(f"Storing transform path for scan {pair_3_to_2[0]} to {pair_3_to_2[1]}: {transform_3_to_2}")

                # Add the pair to the processed set
                processed_pairs.add(pair_3_to_2)
            else:
                # If already processed, reuse the transformation file from the previous registration
                previous_transform_3_to_2 = saved_paths_for_pairs[pair_3_to_2]
                df.loc[trio.index[2], 'transform_path'] = previous_transform_3_to_2
                print(f"Reusing existing transform path for scan {pair_3_to_2[0]} to {pair_3_to_2[1]}")

            print(f"Done with trio: {trio_id}")

    # Save the updated DataFrame to a new CSV file
    updated_csv_path = csv_file_path.replace(".csv", "_with_transforms.csv")
    df.to_csv(updated_csv_path, index=False)
    print(f"Updated CSV saved to {updated_csv_path}")


def calculate_avg_intensity(img_path):
    # Load the NIfTI file
    nifti_img = nib.load(img_path)

    # Get the image data as a numpy array
    nifti_data = nifti_img.get_fdata()

    # Calculate the average intensity for values > 0
    avg_intensity = np.mean(nifti_data[nifti_data > 0])

    return avg_intensity


def process_csv_and_calculate_averages(csv_file_path):
    """
    Processes the CSV file, calculates the average intensity for each NIfTI image,
    and stores the result in a new column 'avg_intensity'. It avoids recalculating
    the average intensity for duplicate scan IDs.
    
    Parameters:
        csv_file_path (str): Path to the input CSV file.
        output_csv_path (str): Path where the updated CSV will be saved.
    """
    # Load the CSV file
    df = pd.read_csv(csv_file_path)

    # Dictionary to store pre-calculated averages for each scan_id
    average_intensity_cache = {}

    # Iterate through each row and calculate/store the average intensity
    for idx, row in df.iterrows():
        scan_id = row['scan_id']

        if scan_id not in average_intensity_cache:
            # Calculate the average intensity if it hasn't been calculated already
            try:
                avg_intensity = calculate_avg_intensity(row['path'])
            except Exception as e:
                print(f"Error loading NIfTI file for scan_id {scan_id}: {e}")
                avg_intensity = np.nan  # If there's an error, store NaN
            average_intensity_cache[scan_id] = avg_intensity
        else:
            # Reuse the cached value for the same scan_id
            avg_intensity = average_intensity_cache[scan_id]

        # Add the average intensity value to the DataFrame
        df.at[idx, 'avg_intensity'] = avg_intensity
        print('Processed scan:', scan_id)
    # Save the updated DataFrame with the new column
    df.to_csv(csv_file_path, index=False)

def plot_mean_scaling_factors(input_csv, value_column, y_title):
    # Load the CSV file
    df = pd.read_csv(input_csv)

    # Define the trios to exclude from plotting
    exclude_trios = ['trio-001', 'trio-177', 'trio-180', 'trio-182', 'trio-183', 'trio-186', 'trio-188', 
                    'trio-189', 'trio-191', 'trio-192', 'trio-193']

    # Filter the dataframe to exclude the specified trios
    filtered_df = df[~df['trio_id'].isin(exclude_trios)]

    # Plot the remaining trios, color based on 'sex' column (1 = Male, 0 = Female)
    plt.figure(figsize=(10, 6))

    male_plotted, female_plotted = False, False  # To track if male/female has been added to the legend
    for trio_id, group in filtered_df.groupby('trio_id'):
        sex_color = '#ADD8E6' if group['sex'].iloc[0] == 1 else '#F08080'
        label = None
        if group['sex'].iloc[0] == 1 and not male_plotted:
            label = 'Male'
            male_plotted = True
        elif group['sex'].iloc[0] == 0 and not female_plotted:
            label = 'Female'
            female_plotted = True
        plt.plot(group['age'], group[f'{value_column}'], marker='o', color=sex_color, label=label)

    # Group the data by 'trio_id' and calculate the mean and standard deviation for each trio element (1st, 2nd, 3rd)
    first_mean = filtered_df.groupby('trio_id').nth(0)[f'{value_column}'].mean()
    first_std = filtered_df.groupby('trio_id').nth(0)[f'{value_column}'].std()
    
    second_mean = filtered_df.groupby('trio_id').nth(1)[f'{value_column}'].mean()
    second_std = filtered_df.groupby('trio_id').nth(1)[f'{value_column}'].std()
    
    third_mean = filtered_df.groupby('trio_id').nth(2)[f'{value_column}'].mean()
    third_std = filtered_df.groupby('trio_id').nth(2)[f'{value_column}'].std()

    # Calculate the mean ages for the first, second, and third elements in each trio
    mean_age_first = filtered_df.groupby('trio_id').nth(0)['age'].mean()
    mean_age_second = filtered_df.groupby('trio_id').nth(1)['age'].mean()
    mean_age_third = filtered_df.groupby('trio_id').nth(2)['age'].mean()

    # Plot the average scaling values centered on the mean ages
    x_values = [mean_age_first, mean_age_second, mean_age_third]
    y_values = [first_mean, second_mean, third_mean]
    error_values = [first_std, second_std, third_std]

    # Plot the means with error bars for standard deviations
    plt.errorbar(x_values, y_values, yerr=error_values, fmt='o-', color='black', capsize=5, label='Mean ± Std Dev')

    # Adding labels and title
    plt.xlabel('Age')
    plt.ylabel(f'{y_title}')
    plt.title(f'{y_title} vs Age for All Trios with Mean Curve (Centered by Age)')

    # Display the plot
    plt.legend()
    plt.show()

def create_rainbow_plot(input_csv, value_column, y_title):
    """
    Creates a rainbow plot plotting the curves of the specified values over age.

    Parameters:
        input_csv (str): Path to the input CSV file.
        value_column (str): The column in the DataFrame to be plotted (e.g., 'avg_intensity', 'scaling_avg').
        y_title (str): Label for the Y-axis (e.g., 'Average Intensity', 'Scaling Avg').
    """
    # Load the CSV file
    df = pd.read_csv(input_csv)
    
    # Step 1: Remove rows with NaN or inf values in value_column or 'age' columns
    df_cleaned = df.dropna(subset=[value_column, 'age'])
    df_cleaned = df_cleaned[np.isfinite(df_cleaned[value_column]) & np.isfinite(df_cleaned['age'])]

    # Step 2: Group by 'participant_id' and 'age', then take the mean of only the numeric columns (e.g., value_column)
    df_grouped = df_cleaned.groupby(['participant_id', 'age'], as_index=False)[[value_column]].mean()

    # Get the unique participant ids
    participants = df_grouped['participant_id'].unique()

    # Define a fixed list of distinct colors for consistency
    color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
                  '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78',
                  '#98df8a', '#ff9896', '#c5b0d5', '#c49c94', '#f7b6d2', '#c7c7c7',
                  '#dbdb8d', '#9edae5', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                  '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    # Create a color map (participant_id -> color) using the predefined color list
    alternate_color_map = {participant_id: color_list[i % len(color_list)] for i, participant_id in enumerate(participants)}

    # Define the number of points for smooth curves
    smooth_points = 300

    # Split participants into 8 groups (6 participants per subplot for 2x4 layout)
    groups = np.array_split(participants, 8)

    # Create the plot with 2x4 subplots, each with up to 6 participants using the alternate color map
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))

    # Flatten axes for easier iteration
    axes = axes.flatten()

    # Iterate over each group of participants (6 per subplot)
    for i, group in enumerate(groups):
        ax = axes[i]
        
        # Plot curves for each participant in the current group
        for j, participant_id in enumerate(group):
            # Filter the dataframe for the current participant
            participant_data = df_grouped[df_grouped['participant_id'] == participant_id]

            # If there's no data left after filtering, skip this participant
            if len(participant_data) == 0:
                continue

            # Sort the group by age to ensure smooth curves
            participant_data = participant_data.sort_values(by='age')

            # Generate smooth values for age
            age_smooth = np.linspace(participant_data['age'].min(), participant_data['age'].max(), smooth_points)

            # Apply quadratic spline interpolation
            spline = make_interp_spline(participant_data['age'], participant_data[value_column], k=2)
            intensity_smooth = spline(age_smooth)

            # Use the alternate color map for the participant
            color = alternate_color_map[participant_id]

            # Plot the smooth curve
            ax.plot(age_smooth, intensity_smooth, color=color, alpha=0.7, label=f'P{j + 1}')
            
            # Plot the original average intensity points
            ax.scatter(participant_data['age'], participant_data[value_column], color=color, edgecolor='black', zorder=5)

        # Add a legend with participant numbers (P1, P2, P3, ...)
        ax.legend(title='Participants', loc='upper right')

        # Customize each subplot
        ax.set_title(f'Participants {i * 6 + 1} to {i * 6 + len(group)}', fontsize=12)
        ax.set_xlabel('Age', fontsize=10)
        ax.set_ylabel(y_title, fontsize=10)
        ax.grid(True)

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()

def analyze_mean_scaling_factors(input_csv):
    df = pd.read_csv(input_csv)
    # Select the 1st and 3rd rows in each trio (for groups of 3)
    first_in_trio = df.groupby('trio_id').nth(0)['scaling_avg']
    third_in_trio = df.groupby('trio_id').nth(2)['scaling_avg']
    print(first_in_trio.shape)
    
    # Calculate mean of scaling_avg
    mean_scaling_avg_scan_1_2 = first_in_trio.mean()
    mean_scaling_avg_scan_3_2 = third_in_trio.mean()
    
    # Compare directionality, expansion, and shrinkage
    directionality_bool = (first_in_trio > third_in_trio).tolist()
    expansion_bool = (first_in_trio > 1).tolist()
    shrinkage_bool = (third_in_trio < 1).tolist()
    
    # Print results
    print('Mean 1-->2:', mean_scaling_avg_scan_1_2)
    print('Mean 3-->2:', mean_scaling_avg_scan_3_2)
    print('Number of trios where Smean 1-->2 > Smean 3-->2:', directionality_bool.count(True), '/', len(directionality_bool))
    print('Number of trios where Smean 1-->2 > 1:', expansion_bool.count(True), '/', len(expansion_bool))
    print('Number of trios where Smean 3-->2 < 1:', shrinkage_bool.count(True), '/', len(shrinkage_bool))
    
    # Find which trios have False in shrinkage_bool and return their details
    false_shrinkage_indices = [i for i, val in enumerate(shrinkage_bool) if not val]
    false_shrinkage_trios = df.groupby('trio_id').nth(2).iloc[false_shrinkage_indices]
    
    print('Trios with False in shrinkage_bool:')
    print(false_shrinkage_trios)

    # Find which trios have False in expansion_bool and return their details
    false_expansion_indices = [i for i, val in enumerate(shrinkage_bool) if not val]
    false_expansion_trios = df.groupby('trio_id').nth(2).iloc[false_expansion_indices]
    
    print('Trios with False in expansion_bool:')
    print(false_expansion_trios)

if __name__ == "__main__":
    # input_csv = '/home/andjela/Documents/CP/trios_sorted_by_age.csv'
    input_csv = './data/CP/trios_sorted_by_age.csv'
    transfo_type = 'rigid_affine'
    abbey_path = '/home/GRAMES.POLYMTL.CA/andim/joplin-intra-inter/CP_rigid_trios/CP'
    save_transform_paths_CP(input_csv, abbey_path, transfo_type)

    # input_csv = '/home/andjela/Documents/CP/trios_sorted_by_age_with_transforms.csv'
    # rel_path = '/home/andjela/joplin-intra-inter/CP_rigid_trios'
    process_csv_and_calculate_scaling_factors('./data/CP/trios_sorted_by_age_with_transforms.csv')
    # input_csv = 'C:\\Users\\andje\\Downloads\\trios_sorted_by_age_with_transforms.csv'
    # create_rainbow_plot(input_csv, 'scaling_avg', 'Scaling Avg')
    # analyze_mean_scaling_factors(input_csv)
    # plot_mean_scaling_factors(input_csv, 'scaling_avg', 'Scaling Avg')
    # plot_mean_scaling_factors(input_csv, 'scaling_z', 'Scaling Z')

    # process_csv_and_calculate_averages(input_csv)
    # create_rainbow_plot(input_csv, 'avg_intensity', 'Average Intensity')