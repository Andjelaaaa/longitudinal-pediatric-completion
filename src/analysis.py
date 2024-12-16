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
from scipy.optimize import curve_fit
import statsmodels.formula.api as smf
from numpy.polynomial.polynomial import Polynomial
from skimage import io, color
from skimage.util import img_as_float
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from itertools import combinations
import nibabel as nib
from scipy.stats import entropy
from scipy.ndimage import zoom
import json

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
    # Check if the file exists
    if not os.path.exists(img_path):
        return np.nan, np.nan

    # Load the NIfTI file
    nifti_img = nib.load(img_path)

    # Get the image data as a numpy array
    nifti_data = nifti_img.get_fdata()

    # Calculate the average and std intensity for values > 0
    avg_intensity = np.mean(nifti_data[nifti_data > 0])
    std_intensity = np.std(nifti_data[nifti_data > 0])

    return avg_intensity, std_intensity

def histogram_smoothness_variance(hist):
        differences = np.diff(hist)
        return np.var(differences)

def calculate_volume_for_labels(image_path, labels_of_interest):
    """
    Calculate the volume for specific labels in a segmentation image.
    
    Parameters:
        image_path (str): Path to the segmentation image.
        labels_of_interest (list): List of labels to calculate the volume for.
    
    Returns:
        float: The volume in cm³ for the specified labels.
    """
    try:
        # Load the segmentation image
        segmentation = nib.load(image_path).get_fdata()

        # Get voxel dimensions from the affine
        voxel_dims = nib.load(image_path).header.get_zooms()
        voxel_volume = np.prod(voxel_dims)  # Voxel volume in mm³

        # Create a mask for the regions corresponding to the labels of interest
        mask = np.isin(segmentation, labels_of_interest)

        # Count the number of voxels in the region of interest
        region_voxels = np.sum(mask)

        # Calculate the total volume in mm³ and convert to cm³
        total_volume_cm3 = (region_voxels * voxel_volume) / 1000  # Convert to cm³

        return total_volume_cm3

    except Exception as e:
        print(f"Error calculating volume for {image_path}: {e}")
        return np.nan


def analyze_volume_for_regions(csv_path, output_csv_path):
    """
    Process a CSV file and calculate volumes for specific labels in the segmentation images.
    
    Parameters:
        csv_path (str): Path to the input CSV file.
        output_csv_path (str): Path to save the updated CSV file.
        labels_of_interest (list): List of labels to calculate the volume for.
    """
    # Load the CSV file
    data = pd.read_csv(csv_path)

    # Initialize a new column for volumes
    data["wm_vol_cm3"] = np.nan
    data["gm_vol_cm3"] = np.nan
     
    # Loop through each row in the DataFrame
    for index, row in data.iterrows():
        print(f"Processing row {index}...")

        # Get the segmentation image path
        seg_path = f"/home/GRAMES.POLYMTL.CA/andim/intra-inter-ddfs/work_dir2/cbf2mni_wdir/{row['participant_id']}/{row['scan_id']}/wf/brainextraction/{row['scan_id']}_seg.nii.gz"
        

        # Calculate volume for the specified labels
        label_dict = ExtractLabelsForRegions()
        wm_labels = label_dict['WM']
        gm_labels = label_dict['GM']
        volume_wm = calculate_volume_for_labels(seg_path, wm_labels)
        volume_gm = calculate_volume_for_labels(seg_path, gm_labels)

        # Assign the calculated volume to the new column
        data.at[index, "wm_vol_cm3"] = volume_wm
        data.at[index, "gm_vol_cm3"] = volume_gm

    # Save the updated DataFrame back to a new CSV file
    data.to_csv(output_csv_path, index=False)
    print(f"Updated CSV saved to {output_csv_path}")

def analyze_n4_CV_for_seg_region(csv_path, output_csv_path, labels_of_interest):
    # Load the CSV file
    data = pd.read_csv(csv_path)

    # Initialize new columns for corrected and non-corrected metrics
    data["CV_before_n4"] = np.nan
    data["CV_after_n4"] = np.nan

    # Loop through each row in the DataFrame
    for index, row in data.iterrows():
        print(f'At row {index}...')
        # Load both image
        corrected_image_path = row['n4_path']
        non_corrected_image_path = f"/home/GRAMES.POLYMTL.CA/andim/intra-inter-ddfs/CP_dataset/Year-{int(row['age']) + (1 if row['age'] - int(row['age']) >= 0.5 else 0)}/{row['sub_id_bids']}-{row['session']}-T1w.nii.gz"
        seg_path = f"/home/GRAMES.POLYMTL.CA/andim/intra-inter-ddfs/work_dir2/cbf2mni_wdir/{row['participant_id']}/{row['scan_id']}/wf/brainextraction/{row['scan_id']}_seg.nii.gz"
        # corrected_hist, corrected_bins = calculate_histogram(corrected_img_path)
        # non_corrected_hist, non_corrected_bins = calculate_histogram(non_corrected_img_path)
        
        # smooth_after = histogram_smoothness_variance(corrected_hist)
        # smooth_bef = histogram_smoothness_variance(non_corrected_hist)
        ratio_aft = calculate_ratio_for_labels(corrected_image_path, seg_path, labels_of_interest)
        print(f"Std/Avg Ratio for labels {labels_of_interest}: {ratio_aft}")

        ratio_bef = calculate_ratio_for_labels(non_corrected_image_path, seg_path, labels_of_interest)
        print(f"Std/Avg Ratio for labels not corrected {labels_of_interest}: {ratio_bef}")

        # Assign values to the corresponding columns
        data.at[index, "CV_before_n4"] = ratio_bef
        data.at[index, "CV_after_n4"] = ratio_aft
        
        
    # Save the updated DataFrame back to a new CSV file
    data.to_csv(output_csv_path, index=False)
    print(f"Updated CSV saved to {output_csv_path}")

def ThreeRegionsMapping():
    """
    Define a mapping of three brain regions (WM, GM, CSF) to corresponding anatomical structures.

    Returns:
        Dict[str, List[str]]: Mapping of brain regions to anatomical structures.
    """
    mapping = {
        'WM': ['cerebral white matter', 'brain-stem', 'cerebellum white matter', 'pallidum', 'ventral DC'],
        'GM': ['cerebral cortex', 'caudate', 'thalamus', 'putamen', 'hippocampus', 'amygdala', 'cerebellum cortex', 'accumbens area'],
        'CSF': ['CSF', 'lateral ventricle', '4th ventricle', '3rd ventricle', 'inferior lateral ventricle']
    }
    return mapping

def LabelsMappingSS():
    """
    Define a mapping of anatomical structure names to their corresponding label values in SynthSeg.

    Returns:
        Dict[str, List[int]]: Mapping of anatomical structure names to label values.
    """
    label_names = {
    'cerebral white matter': [2, 41],
    'cerebral cortex': [3, 42],
    'lateral ventricle': [4, 43],
    'inferior lateral ventricle': [5, 44],
    'cerebellum white matter': [7, 46],
    'cerebellum cortex': [8, 47],
    'thalamus': [10, 49],
    'caudate': [11, 50],
    'putamen': [12, 51],
    'pallidum': [13, 52],
    '3rd ventricle': [14],
    '4th ventricle': [15],
    'brain-stem': [16],
    'hippocampus': [17, 53],
    'amygdala': [18, 54],
    'CSF': [24],
    'accumbens area': [26, 58],
    'ventral DC': [28, 60]
    }  
    return label_names

def ExtractLabelsForRegions():
    """
    Extract values corresponding to anatomical regions from SynthSeg label mapping.

    Returns:
        Dict[str, List[int]]: Label values associated with White Matter (WM), Gray Matter (GM), and Cerebrospinal Fluid (CSF).
    """
    # Get the region mapping
    regions_mapping = ThreeRegionsMapping()
    
    # Get the label mapping
    labels_mapping = LabelsMappingSS()

    # Initialize dictionaries to store the values for each region
    values_by_region = {
        'WM': [],
        'GM': [],
        'CSF': []
    }

    # Iterate through each region and collect associated values
    for region, structures in regions_mapping.items():
        for structure in structures:
            if structure in labels_mapping:
                values_by_region[region].extend(labels_mapping[structure])

    return values_by_region

def analyze_n4_correction(csv_path, output_csv_path):
    # Load the CSV file
    data = pd.read_csv(csv_path)

    # Initialize new columns for corrected and non-corrected metrics
    data["n4_avg"] = np.nan
    data["n4_std"] = np.nan
    data["no_n4_avg"] = np.nan
    data["no_n4_std"] = np.nan

    # Loop through each row in the DataFrame
    for index, row in data.iterrows():
        print(f'At row {index}...')
        # Calculate stats for the corrected image
        corrected_img_path = row['n4_path']
        corrected_mean, corrected_std = calculate_avg_intensity(corrected_img_path)

        # Calculate stats for the non-corrected image
        non_corrected_img_path = row['path']
        non_corrected_mean, non_corrected_std = calculate_avg_intensity(non_corrected_img_path)

        # Assign values to the corresponding columns
        data.at[index, "n4_avg"] = corrected_mean
        data.at[index, "n4_std"] = corrected_std
        data.at[index, "no_n4_avg"] = non_corrected_mean
        data.at[index, "no_n4_std"] = non_corrected_std

    # Save the updated DataFrame back to a new CSV file
    data.to_csv(output_csv_path, index=False)
    print(f"Updated CSV saved to {output_csv_path}")

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
    exclude_trios = ['trio-001']

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

def make_ticks_bold(ax):
    """
    Helper function to make tick labels bold for a given axis.
    """
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')

def create_rainbow_plot(input_csv, value_column, y_title, fig_title):
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
    print(participants, len(participants))

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
        # ax.legend(title='Participants', loc='upper right')

        # Customize each subplot
        ax.set_title(f'Participants {i * 6 + 1} to {i * 6 + len(group)}', fontsize=16, fontweight='bold')
        # ax.set_xlabel('Age (years)', fontsize=10)
        # ax.set_ylabel(y_title, fontsize=10)
        ax.grid(True)
        # Set tick label size
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.set_xlim(2, 7)

        # Make tick labels bold
        make_ticks_bold(ax)
     # Remove individual x and y labels for subplots
    for ax in axes:
        ax.set_xlabel("")
        ax.set_ylabel("")

    # Add centralized x and y labels
    fig.text(0.5, 0.04, 'Age (years)', ha='center', va='center', fontsize=18, fontweight='bold')
    fig.text(0.04, 0.5, y_title, ha='center', va='center', rotation='vertical', fontsize=18, fontweight='bold')


    # Adjust layout and show the plot
    plt.tight_layout(rect=[0.05, 0.05, 1, 0.95]) 
    # Adjust layout and show the plot
    # plt.tight_layout()
    # plt.show()
    plt.savefig(f'{fig_title}.png')
   

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
    print('Mean 1-->2:', mean_scaling_avg_scan_1_2, first_in_trio.std())
    print('Mean 3-->2:', mean_scaling_avg_scan_3_2, third_in_trio.std())
    print('Number of trios where Smean 1-->2 > Smean 3-->2:', directionality_bool.count(True), '/', len(directionality_bool))
    print('Number of trios where Smean 1-->2 > 1:', expansion_bool.count(True), '/', len(expansion_bool))
    print('Number of trios where Smean 3-->2 < 1:', shrinkage_bool.count(True), '/', len(shrinkage_bool))
    
    # Find which trios have False in shrinkage_bool and return their details
    false_shrinkage_indices = [i for i, val in enumerate(shrinkage_bool) if not val]
    false_shrinkage_trios = df.groupby('trio_id').nth(2).iloc[false_shrinkage_indices]
    
    print('Trios with False in shrinkage_bool:')
    print(false_shrinkage_trios)

    # Find which trios have False in expansion_bool and return their details
    false_expansion_indices = [i for i, val in enumerate(expansion_bool) if not val]
    false_expansion_trios = df.groupby('trio_id').nth(2).iloc[false_expansion_indices]
    
    print('Trios with False in expansion_bool:')
    print(false_expansion_trios)

def analyze_stats_per_csv(input_csv, value_column):
    # Load the data
    df = pd.read_csv(input_csv)

    # Display global volume summary statistics
    print(df[f'{value_column}'].describe())

    # Group by 'trio_id' and calculate the average global volume per trio
    average_global_volume_per_trio = df.groupby('trio_id')[f'{value_column}'].mean()

    # Sort by 'trio_id' and 'age' within each trio to ensure chronological order
    sorted_data = df.sort_values(by=['trio_id', 'age'])

    # Group by trio_id and check if volumes are increasing
    volume_check = sorted_data.groupby('trio_id')[f'{value_column}'].apply(
        lambda x: all(x.iloc[i] < x.iloc[i + 1] for i in range(len(x) - 1))
    )

    # Filter to find the trios where the volume increasing check is False
    trios_with_false_check = volume_check[~volume_check].index

    # Detailed information about where the volume is not increasing
    non_increasing_details = []
    for trio_id in trios_with_false_check:
        trio_data = sorted_data[sorted_data['trio_id'] == trio_id]
        for i in range(len(trio_data) - 1):
            if trio_data[f'{value_column}'].iloc[i] >= trio_data[f'{value_column}'].iloc[i + 1]:
                non_increasing_details.append({
                    "trio_id": trio_id,
                    "age1": trio_data['age'].iloc[i],
                    "volume1": trio_data[f'{value_column}'].iloc[i],
                    "age2": trio_data['age'].iloc[i + 1],
                    "volume2": trio_data[f'{value_column}'].iloc[i + 1]
                })

    # Count and list the trios
    number_of_false_trios = len(trios_with_false_check)
    false_trios_list = list(trios_with_false_check)

    

    return {
        "average_global_volume_per_trio": average_global_volume_per_trio,
        "volume_increasing_check": volume_check,
        "number_of_false_trios": number_of_false_trios,
        "false_trios_list": false_trios_list,
        "non_increasing_details": non_increasing_details
    }

def plot_mean_value_column(input_csv, value_column, y_title, fig_title):

    # Load the CSV file
    filtered_df = pd.read_csv(input_csv)

    # Plot the trios, color based on 'sex' column (1 = Male, 0 = Female)
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
    first_mean = filtered_df.groupby('trio_id').nth(0)[value_column].mean()
    first_std = filtered_df.groupby('trio_id').nth(0)[value_column].std()
    
    second_mean = filtered_df.groupby('trio_id').nth(1)[value_column].mean()
    second_std = filtered_df.groupby('trio_id').nth(1)[value_column].std()
    
    third_mean = filtered_df.groupby('trio_id').nth(2)[value_column].mean()
    third_std = filtered_df.groupby('trio_id').nth(2)[value_column].std()

    # Calculate the mean ages for the first, second, and third elements in each trio
    mean_age_first = filtered_df.groupby('trio_id').nth(0)['age'].mean()
    mean_age_second = filtered_df.groupby('trio_id').nth(1)['age'].mean()
    mean_age_third = filtered_df.groupby('trio_id').nth(2)['age'].mean()

    # Prepare x (age) and y (mean) values
    x_values = np.array([mean_age_first, mean_age_second, mean_age_third])
    y_values = np.array([first_mean, second_mean, third_mean])
    lower_bounds = y_values - np.array([first_std, second_std, third_std])
    upper_bounds = y_values + np.array([first_std, second_std, third_std])

    # Remove duplicates and ensure sufficient points for interpolation
    unique_indices = np.unique(x_values, return_index=True)[1]
    x_values_unique = x_values[unique_indices]
    lower_bounds_unique = lower_bounds[unique_indices]
    upper_bounds_unique = upper_bounds[unique_indices]
    y_values_unique = y_values[unique_indices]

    # Perform interpolation for smooth curves
    x_smooth = np.linspace(x_values_unique.min(), x_values_unique.max(), 500)  # Smooth x values

    # Use linear interpolation for fallback to avoid boundary condition issues
    lower_smooth = np.interp(x_smooth, x_values_unique, lower_bounds_unique)
    upper_smooth = np.interp(x_smooth, x_values_unique, upper_bounds_unique)
    mean_smooth = np.interp(x_smooth, x_values_unique, y_values_unique)

    # Plot the results
    plt.plot(x_smooth, mean_smooth, label='Mean', color='black', linewidth=2)
    plt.fill_between(x_smooth, lower_smooth, upper_smooth, color='blue', alpha=0.3, label='Mean ± Std Dev')

    # Adding labels and title
    plt.xlabel('Age')
    plt.ylabel(f'{y_title}')
    plt.title(f'{y_title} vs Age with Smooth Mean ± Std Dev')

    # Display the legend
    plt.legend()
    plt.savefig(f'{fig_title}.png')
    # plt.show()

def plot_extrapolated_mean(input_csv, value_column, y_title, fig_title):

    # Load the CSV file
    df = pd.read_csv(input_csv)

    # Plot the trios, color based on 'sex' column (1 = Male, 0 = Female)
    plt.figure(figsize=(10, 6))

    male_plotted, female_plotted = False, False  # To track if male/female has been added to the legend
    for trio_id, group in df.groupby('trio_id'):
        sex_color = '#ADD8E6' if group['sex'].iloc[0] == 1 else '#F08080'
        label = None
        if group['sex'].iloc[0] == 1 and not male_plotted:
            label = 'Male'
            male_plotted = True
        elif group['sex'].iloc[0] == 0 and not female_plotted:
            label = 'Female'
            female_plotted = True
        plt.plot(group['age'], group[f'{value_column}'], marker='o', color=sex_color, label=label)

    # Group by 'age' to calculate the mean and std
    grouped = df.groupby('age')[value_column].agg(['mean', 'std']).reset_index()

    # Handle missing or NaN values
    grouped = grouped.dropna()

    # Normalize the x-values for numerical stability
    x = grouped['age']
    x_norm = (x - x.mean()) / x.std()
    y_mean = grouped['mean']
    y_std = grouped['std']

    # Fit a lower-degree polynomial regression (degree=2)
    try:
        mean_poly = Polynomial.fit(x_norm, y_mean, deg=2)
        std_poly = Polynomial.fit(x_norm, y_std, deg=2)
    except Exception as e:
        print(f"Polynomial fitting failed: {e}")
        return

    # Define extended x range for extrapolation
    x_extended = np.linspace(x_norm.min() - 0.5, x_norm.max() + 0.5, 500)

    # Predict mean and std for the extended range
    mean_extended = mean_poly(x_extended)
    std_extended = std_poly(x_extended)

    # De-normalize x for plotting
    x_extended_original = x_extended * x.std() + x.mean()

    # Plot the mean and extrapolated standard deviation curves
    plt.plot(x_extended_original, mean_extended, label='Mean', color='black', linewidth=2)
    plt.fill_between(x_extended_original, mean_extended - std_extended, mean_extended + std_extended,
                     color='blue', alpha=0.3, label='Mean ± Std Dev')

    # Adding labels and title
    plt.xlabel('Age')
    plt.ylabel(f'{y_title}')
    plt.title(f'{y_title} vs Age with Extrapolated Mean ± Std Dev')

    # Display the legend
    plt.legend()
    plt.savefig(f'{fig_title}.png')

def fit_mix_effects_model(csv_file_path, value_column, y_title, fig_title):
    # Load the data
    data = pd.read_csv(csv_file_path)

    data = data.dropna(subset=[f"{value_column}", "age", "sex"])
    
    # Fit a linear mixed-effects model
    model = smf.mixedlm(
        f"{value_column} ~ age + sex",  # Fixed effects
        data, 
        groups=data["sub_id_bids"],    # Random effects grouped by subject
        re_formula="~age"             # Random slope for age
    )
    result = model.fit()

    # Summary of the model
    print(result.summary())

    # Predict growth trajectories
    data[f"predicted_{value_column}"] = result.predict(data)

    # Define colors based on sex (1 = Male, 0 = Female)
    color_map = {1: "blue", 0: "red"}
    shade_map = {1: "#add8e6", 0: "#ffcccb"}  # Lighter shades for predictions

    # Plotting the trajectories
    plt.figure(figsize=(10, 6))
    for subject_id, group in data.groupby("sub_id_bids"):
        sex = group["sex"].iloc[0]  # Assuming sex is constant for each subject
        color = color_map.get(sex, "gray")  # Default to gray if sex is not 0 or 1
        plt.plot(group["age"], group[value_column], marker="o", color=color, linestyle='--', alpha=0.4)
        color = shade_map.get(sex, "gray")
        plt.plot(group["age"], group[f"predicted_{value_column}"], color=color, alpha=1)

    # Add a legend only for the sex
    for sex, color in color_map.items():
        plt.scatter([], [], color=color, label="Male" if sex == 1 else "Female")
    plt.legend(title="Sex")

    plt.xlabel("Age")
    plt.ylabel(f"{y_title}")
    # plt.title("Growth Trajectories by Sex")
    plt.grid(True)
    plt.savefig(f'{fig_title}.png')
    # plt.show()

def plot_non_increasing_volumes(results):
    non_increasing_data = results['non_increasing_details']
    # Create a DataFrame
    non_increasing_df = pd.DataFrame(non_increasing_data)
    # Prepare the data for plotting from non-increasing details
    non_increasing_df['volume_decrease'] = non_increasing_df['volume1'] - non_increasing_df['volume2']
    non_increasing_df['age_difference'] = non_increasing_df['age2'] - non_increasing_df['age1']

    # Plot the volume decrease with color based on age difference
    plt.figure(figsize=(10, 6))
    sc = plt.scatter(non_increasing_df['age1'], non_increasing_df['volume_decrease'],
                    c=non_increasing_df['age_difference'], cmap='viridis', s=100, alpha=0.7)
    
    # Add text annotations for trio_id
    for i, row in non_increasing_df.iterrows():
        plt.text(row['age1'], row['volume_decrease'], row['trio_id'], fontsize=8, ha='right', alpha=0.8)

    

    # # Add text annotations for trio_id without overlapping
    # # Add or update text annotations for trio_id
    # for i, row in non_increasing_df.iterrows():
    #     text_key = (row['age1'], row['volume_decrease'])  # Unique key for the text position
    #     # Check if text already exists at the position
    #     existing_text = None
    #     for text in plt.gca().texts:
    #         if text.get_position() == text_key:
    #             existing_text = text
    #             break

    #     if existing_text:
    #         # Update the existing text to bold
    #         existing_text.set_weight('bold')
    #     else:
    #         # Add a new text annotation if none exists
    #         plt.text(
    #             row['age1'], 
    #             row['volume_decrease'], 
    #             row['trio_id'], 
    #             fontsize=8, 
    #             ha='right', 
    #             alpha=0.8, 
    #             weight='bold'  # Add text in bold
    #         )

    # Add a colorbar to show the age difference
    cbar = plt.colorbar(sc)
    cbar.set_label('Age Difference (years)')

    # Add labels and title
    plt.xlabel('Age at Volume Decrease (Age 1)')
    plt.ylabel('Volume Decrease (cm³)')
    plt.title('Volume Decrease for Non-Increasing Trios')

    # Display the plot
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# def fit_non_linear_mix_effects_model(csv_file_path, value_column, y_title, fig_title):
#     # Load the data
#     data = pd.read_csv(csv_file_path)

#     data = data.dropna(subset=[f"{value_column}", "age", "sex"])
    
#     # Add a polynomial term for non-linearity (e.g., quadratic)
#     data["age_squared"] = data["age"] ** 2

#     # Fit a mixed-effects model with non-linear terms
#     model = smf.mixedlm(
#         f"{value_column} ~ age + age_squared + sex",  # Fixed effects
#         data, 
#         groups=data["sub_id_bids"],                 # Random effects grouped by subject
#         re_formula="~age + age_squared"            # Random slopes for age and age_squared
#     )
#     result = model.fit()

#     # Print the summary
#     print(result.summary())

#     # Predict average growth trajectory by sex
#     data[f"predicted_{value_column}"] = result.predict(data)
#     avg_predictions_by_sex = data.groupby(["age", "sex"])[f"predicted_{value_column}"].mean().reset_index()

#     # Define colors for sexes
#     color_map = {1: "blue", 0: "red"}
#     shade_map = {1: "#add8e6", 0: "#ffcccb"}  # Lighter shades for observed points

#     # for subject_id, group in data.groupby("sub_id_bids"):
#     #     sex = group["sex"].iloc[0]  # Assuming sex is constant for each subject

#     #     # Observed data (points)
#     #     plt.scatter(group["age"], group[value_column], marker="o", 
#     #                 color=color_map.get(sex, "gray"), alpha=0.5, label=f"Observed ({subject_id})")

#     #     # Predicted data (points)
#     #     plt.scatter(group["age"], group[f"predicted_{value_column}"], marker="x", 
#     #                 color=shade_map.get(sex, "gray"), alpha=0.8, label=f"Predicted ({subject_id})")

#     # Plot the observed trajectories
#     plt.figure(figsize=(10, 6))
#     for subject_id, group in data.groupby("sub_id_bids"):
#         sex = group["sex"].iloc[0]  # Assuming sex is constant for each subject
        
#         # Observed data (lighter, dashed line)
#         plt.plot(group["age"], group[value_column], linestyle='--', marker="o", 
#                  color=color_map.get(sex, "gray"), alpha=0.5)

#     # # Plot the average predicted trajectory by sex
#     # for sex, sex_group in avg_predictions_by_sex.groupby("sex"):
#     #     color = color_map.get(sex, "black")  # Use primary color for the average curve
#     #     plt.plot(sex_group["age"], sex_group[f"predicted_{value_column}"], 
#     #              color=color, linewidth=4, label="Male Avg" if sex == 1 else "Female Avg")

#     # Add a legend for sexes
#     for sex, color in color_map.items():
#         plt.scatter([], [], color=color, label="Male" if sex == 1 else "Female")
#     plt.legend(title="Sex")

#     # Add labels and title
#     plt.xlabel("Age (years)")
#     plt.ylabel(f"{y_title}")
#     plt.title("Growth Trajectories by Sex with Average Non-Linear Prediction")
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(f'{fig_title}')
#     # plt.show()

def fit_non_linear_mix_effects_model(csv_file_path, value_column, y_title, fig_title):
    # Load the data
    data = pd.read_csv(csv_file_path)

    # Drop rows with missing values in relevant columns
    data = data.dropna(subset=[f"{value_column}", "age", "sex"])
    
    # Add a polynomial term for non-linearity (e.g., quadratic)
    data["age_squared"] = data["age"] ** 2

    # Fit a mixed-effects model with non-linear terms
    model = smf.mixedlm(
        f"{value_column} ~ age + age_squared + sex",  # Fixed effects
        data, 
        groups=data["sub_id_bids"],                 # Random effects grouped by subject
        re_formula="~age + age_squared"            # Random slopes for age and age_squared
    )
    result = model.fit()

    # Print the summary
    print(result.summary())

    # Predict average growth trajectory by sex
    data[f"predicted_{value_column}"] = result.predict(data)
    avg_predictions_by_sex = data.groupby(["age", "sex"])[f"predicted_{value_column}"].mean().reset_index()

    # Calculate residuals and standard deviation of residuals by sex
    data["residual"] = data[value_column] - data[f"predicted_{value_column}"]
    std_by_sex = data.groupby("sex")["residual"].std().to_dict()

    # Define colors for sexes
    color_map = {1: "blue", 0: "red"}
    shade_map = {1: "#add8e6", 0: "#ffcccb"}  # Lighter shades for observed points

    # Plot the observed trajectories
    plt.figure(figsize=(10, 6))
    male_plotted, female_plotted = False, False
    for trio_id, group in data.groupby('trio_id'):
        sex_color = '#ADD8E6' if group['sex'].iloc[0] == 1 else '#F08080'
        label = None
        if group['sex'].iloc[0] == 1 and not male_plotted:
            label = 'Male'
            male_plotted = True
        elif group['sex'].iloc[0] == 0 and not female_plotted:
            label = 'Female'
            female_plotted = True
        plt.plot(group['age'], group[f'{value_column}'], marker='o', color=sex_color, label=label)
    # for subject_id, group in data.groupby("sub_id_bids"):
    #     sex = group["sex"].iloc[0]  # Assuming sex is constant for each subject
        
    #     # Observed data (lighter, dashed line)
    #     plt.plot(group["age"], group[value_column], linestyle='--', marker="o", 
    #              color=color_map.get(sex, "gray"), alpha=0.5)

    # Plot the average predicted trajectory by sex
    for sex, sex_group in avg_predictions_by_sex.groupby("sex"):
        color = color_map.get(sex, "black")  # Use primary color for the average curve
        plt.plot(sex_group["age"], sex_group[f"predicted_{value_column}"], 
                 color=color, linewidth=4, label="Male Avg" if sex == 1 else "Female Avg")
    # Plot the average predicted trajectory with shaded zones for ±1 and ±2 SD
    for sex, sex_group in avg_predictions_by_sex.groupby("sex"):
        color = color_map.get(sex, "black")  # Use primary color for the average curve
        std = std_by_sex.get(sex, 0)  # Get the standard deviation for this sex
    # Plot shaded regions for ±1 and ±2 SD
        plt.fill_between(
            sex_group["age"],
            sex_group[f"predicted_{value_column}"] - std,
            sex_group[f"predicted_{value_column}"] + std,
            color=color,
            alpha=0.2,
            label=f"{'Male' if sex == 1 else 'Female'} ±1 SD"
        )
        
    # Make x and y ticks bold and larger
    plt.xticks(fontsize=14, fontweight='bold')
    plt.yticks(fontsize=14, fontweight='bold')

    # Add legend with bold and larger font
    legend = plt.legend(fontsize=14, loc='best', frameon=True)
    if legend.get_title() is not None:  # Check if the legend has a title
        legend.get_title().set_fontsize(16)  # Set title font size
        legend.get_title().set_fontweight('bold')  # Set title font weight
    for text in legend.get_texts():
        text.set_fontweight('bold')
    # Add labels and title
    plt.xlabel("Age (years)", fontsize=18, fontweight='bold')
    plt.ylabel(f"{y_title}", fontsize=18, fontweight='bold')
    # plt.title("Growth Trajectories by Sex with Average Non-Linear Prediction")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{fig_title}.png')
    # plt.show()

def fit_non_linear_mix_effects_model_wo_std(csv_file_path, value_column, y_title, fig_title):
    # Load the data
    data = pd.read_csv(csv_file_path)

    # Drop rows with missing values in relevant columns
    data = data.dropna(subset=[f"{value_column}", "age", "sex"])
    
    # Add a polynomial term for non-linearity (e.g., quadratic)
    data["age_squared"] = data["age"] ** 2

    # Fit a mixed-effects model with non-linear terms
    model = smf.mixedlm(
        f"{value_column} ~ age + age_squared + sex",  # Fixed effects
        data, 
        groups=data["sub_id_bids"],                 # Random effects grouped by subject
        re_formula="~age + age_squared"            # Random slopes for age and age_squared
    )
    result = model.fit()

    # Print the summary
    print(result.summary())

    # Predict average growth trajectory by sex
    data[f"predicted_{value_column}"] = result.predict(data)
    avg_predictions_by_sex = data.groupby(["age", "sex"])[f"predicted_{value_column}"].mean().reset_index()

    # Define colors for sexes
    color_map = {1: "blue", 0: "red"}
    # shade_map = {1: "#87CEEB", 0: "#FF9999"}  # Lighter shades for observed points
    shade_map = {1: "#ADD8E6", 0: "#F08080"}

    # Plot the observed trajectories by trio
    plt.figure(figsize=(10, 6))
    for trio_id, group in data.groupby("trio_id"):
        sex = group["sex"].iloc[0]  # Assuming sex is constant for each trio

        # Observed data (connect points within the same trio)
        plt.plot(group["age"], group[value_column], linestyle='-', marker="o", 
                 color=shade_map.get(sex, "gray"), alpha=0.5)

    # Plot the average predicted trajectory by sex
    for sex, sex_group in avg_predictions_by_sex.groupby("sex"):
        color = color_map.get(sex, "black")  # Use primary color for the average curve
        plt.plot(sex_group["age"], sex_group[f"predicted_{value_column}"], 
                 color=color, linewidth=4, label="Male Avg" if sex == 1 else "Female Avg")

    # Add a legend for sexes
    for sex, color in shade_map.items():
        plt.scatter([], [], color=color, label="Male" if sex == 1 else "Female")
    plt.legend(title="Sex")

    # Add labels and title
    plt.xlabel("Age (years)")
    plt.ylabel(f"{y_title}")
    # plt.title("Growth Trajectories by Trio with Average Non-Linear Prediction")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{fig_title}.png')
    plt.show()


# Function to compute and plot Fourier Transform
def plot_fourier_transform_3d(image_path):
    image = nib.load(image_path).get_fdata()

    # Apply 3D Fourier Transform
    ft_image = np.fft.fftn(image)
    ft_image_shifted = np.fft.fftshift(ft_image)

    # Compute magnitude spectrum
    magnitude_spectrum = np.log(1 + np.abs(ft_image_shifted))

    # Extract frequency components
    freqs_x = np.fft.fftshift(np.fft.fftfreq(image.shape[0]))
    freqs_y = np.fft.fftshift(np.fft.fftfreq(image.shape[1]))

    return image, magnitude_spectrum, freqs_x, freqs_y

def plot_with_fft_n4():
    # Compute Fourier Transform for both images
    corrected_image, corrected_spectrum, corrected_freqs_x, corrected_freqs_y = plot_fourier_transform_3d(corrected_image_path)
    non_corrected_image, non_corrected_spectrum, non_corrected_freqs_x, non_corrected_freqs_y = plot_fourier_transform_3d(non_corrected_image_path)

    # Select a representative slice (e.g., middle slice along the z-axis)
    slice_index_corrected = corrected_image.shape[2] // 2  # Middle slice for corrected image
    slice_index_non_corrected = non_corrected_image.shape[2] // 2  # Middle slice for non-corrected image

    corrected_slice = corrected_image[:, :, slice_index_corrected]
    corrected_spectrum_slice = corrected_spectrum[:, :, slice_index_corrected]

    non_corrected_slice = non_corrected_image[:, :, slice_index_non_corrected]
    non_corrected_spectrum_slice = non_corrected_spectrum[:, :, slice_index_non_corrected]

    # Plot the original images, their Fourier Transforms, and the magnitude graphs
    plt.figure(figsize=(16, 12))

    # Corrected Image and Fourier Transform
    plt.subplot(3, 2, 1)
    plt.title("Corrected Image (Middle Slice)")
    plt.imshow(corrected_slice, cmap='gray')
    plt.axis('off')

    plt.subplot(3, 2, 2)
    plt.title("Corrected FT (Magnitude Spectrum, Middle Slice)")
    plt.imshow(corrected_spectrum_slice, cmap='gray')
    plt.axis('off')

    # Non-Corrected Image and Fourier Transform
    plt.subplot(3, 2, 3)
    plt.title("Non-Corrected Image (Middle Slice)")
    plt.imshow(non_corrected_slice, cmap='gray')
    plt.axis('off')

    plt.subplot(3, 2, 4)
    plt.title("Non-Corrected FT (Magnitude Spectrum, Middle Slice)")
    plt.imshow(non_corrected_spectrum_slice, cmap='gray')
    plt.axis('off')

    # Magnitude Graph for Corrected
    plt.subplot(3, 2, 5)
    plt.title("Corrected Image Frequency Magnitude")
    plt.plot(corrected_freqs_x, np.sum(corrected_spectrum_slice, axis=0), label='X-Frequency', color='blue')
    plt.plot(corrected_freqs_y, np.sum(corrected_spectrum_slice, axis=1), label='Y-Frequency', color='green')
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.grid(True)

    # Magnitude Graph for Non-Corrected
    plt.subplot(3, 2, 6)
    plt.title("Non-Corrected Image Frequency Magnitude")
    plt.plot(non_corrected_freqs_x, np.sum(non_corrected_spectrum_slice, axis=0), label='X-Frequency', color='blue')
    plt.plot(non_corrected_freqs_y, np.sum(non_corrected_spectrum_slice, axis=1), label='Y-Frequency', color='green')
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('fourier_bef_af_n4_with_magnitude.png')

def histogram_entropy(hist):
    """
    Calculate the entropy of a histogram as a measure of smoothness.
    """
    hist = hist + 1e-10  # Avoid log(0) errors by adding a small value
    return entropy(hist)

def calculate_histogram(image_path, bins=100):
    # Load the image
    image = nib.load(image_path).get_fdata()

    # Flatten the 3D image to a 1D array and filter non-zero intensities
    image_flat = image[image > 0].flatten()

    # Compute histogram
    hist, bin_edges = np.histogram(image_flat, bins=bins, density=True)

    return hist, bin_edges

def resample_image_to_target(image, target_shape):
    """
    Resamples an input 3D image to match the target shape using interpolation.
    
    Parameters:
        image (np.ndarray): Input 3D image with shape (512, 512, 210).
        target_shape (tuple): Desired output shape (e.g., (231, 231, 189)).
        
    Returns:
        np.ndarray: Resampled image with the target shape.
    """
    # Calculate the scale factors for each dimension
    scale_factors = [t / s for t, s in zip(target_shape, image.shape)]
    
    # Resample the image using the scale factors
    resampled_image = zoom(image, scale_factors, order=3)  # Order=3 for cubic interpolation
    
    return resampled_image

def calculate_brain_volume(img_path):

    # Load the NIfTI image
    img = nib.load(img_path)
    data = img.get_fdata()

    # Count the number of non-zero voxels
    brain_voxels = np.count_nonzero(data)

    # Get voxel dimensions from the affine
    voxel_dims = img.header.get_zooms()
    voxel_volume = np.prod(voxel_dims)  # Voxel volume in mm³

    # Calculate total brain volume in mm³
    total_brain_volume = brain_voxels * voxel_volume

    # Convert to cm³ (optional)
    total_brain_volume_cm3 = total_brain_volume / 1000

    # print(f"Total Brain Volume: {total_brain_volume} mm³ ({total_brain_volume_cm3} cm³)")
    return total_brain_volume_cm3


def process_csv_and_calculate_volume_after_skull_stripping(csv_file_path):
    """
    Processes the CSV file, calculates the mean volume for each skull-stripped image,
    and stores the result in new columns for global volume.
    
    Parameters:
        csv_file_path (str): Path to the input CSV file.
    """
    # Load the CSV file
    df = pd.read_csv(csv_file_path)

    # Iterate through each row and calculate/store the scaling factors
    for idx, row in df.iterrows():
        print(f'Doing trio-{idx}...')
        n4_skull_strip_path = row['path']
        df.at[idx, 'global_volume'] = calculate_brain_volume(n4_skull_strip_path)

    df.to_csv(csv_file_path, index=False)

def calculate_ratio_for_labels(image_path, seg_path, labels_of_interest):
    """
    Calculate the std/avg ratio for intensities in regions defined by specific labels.

    Parameters:
        image_path (str): Path to the intensity image (e.g., T1w image).
        seg_path (str): Path to the segmentation image.
        labels_of_interest (list): List of labels to include in the calculation (e.g., [2, 41]).

    Returns:
        float: The std/avg ratio for intensities in the regions defined by labels_of_interest.
    """
    try:
        # Load the intensity image and segmentation
        image = nib.load(image_path).get_fdata()
        segmentation = nib.load(seg_path).get_fdata()

        # Resample the image to match the segmentation's shape
        resampled_image = resample_image_to_target(image, segmentation.shape)

        # Create a mask for the regions corresponding to the labels of interest
        mask = np.isin(segmentation, labels_of_interest)

        # Extract intensities from the regions of interest
        intensities = resampled_image[mask]

        # Exclude zero intensities (background)
        intensities = intensities[intensities > 0]

        # Calculate mean and standard deviation
        mean_intensity = np.mean(intensities)
        std_intensity = np.std(intensities)

        # Calculate and return the ratio
        if mean_intensity == 0:  # Avoid division by zero
            return np.nan
        return std_intensity / mean_intensity

    except Exception as e:
        # If there's an error (e.g., file not found or loading failure), return NaN
        print(f"Error processing files: {e}")
        return np.nan

def plot_dataset_demographics(input_tsv):
    data = pd.read_csv(input_tsv, sep='\t')

    # Define the subset of participants
    subset_participants = [
        10006, 10007, 10008, 10009, 10010, 10014, 10020, 10021, 10022, 10025,
        10027, 10032, 10044, 10046, 10047, 10049, 10053, 10054, 10056, 10059,
        10061, 10064, 10065, 10068, 10073, 10077, 10080, 10082, 10083, 10085,
        10087, 10088, 10089, 10090, 10094, 10096, 10097, 10098, 10104, 10107,
        10109, 10110, 10117, 10118, 10121, 10123, 10136, 10137
    ]

    # Filter the data for the subset of participants
    subset_data = data[data['participant_id'].isin(subset_participants)]

    # Count the total number of subjects and scans in the subset
    num_subjects = subset_data['participant_id'].nunique()
    num_images = subset_data.shape[0]

    # Calculate the number of trios (if data is grouped by participant and session)
    num_trios = 490

    # Extract columns for boxplot
    scans_per_subject = subset_data.groupby('participant_id').size()
    ages = subset_data['age']

    # Sex distribution
    sex_counts = subset_data['sex'].value_counts()
    sex_labels = ['Female', 'Male']
    sex_values = [sex_counts.get(0, 0), sex_counts.get(1, 0)]

    # Create a 1x3 subplot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), gridspec_kw={'width_ratios': [1, 1, 1]})
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1, wspace=0.3)

    # Bar chart for dataset overview
    plt.subplot(1, 3, 1)
    categories = ['Subjects', 'T1w Scans', 'Trios']
    values = [num_subjects, num_images, num_trios]
    bars = plt.bar(categories, values, color=['skyblue', 'orange', 'green'])
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10, str(val), ha='center', fontsize=16, fontweight='bold')
    plt.title("Subset Dataset Overview", fontsize=18, fontweight='bold')
    plt.ylabel("Count", fontsize=14, fontweight='bold')
    plt.xticks(categories, fontsize=16, fontweight='bold')
    plt.ylim(0, max(values) + 50)
    # Remove y-axis ticks
    plt.yticks([])

    # Box plots for scans and age
    plt.subplot(1, 3, 2)
    box = plt.boxplot([scans_per_subject, ages], vert=False, patch_artist=True,
                      boxprops=dict(facecolor='lightblue'))
    plt.yticks([1, 2], ['Scans/\nSubject', 'Age\n(years)'], fontsize=16, fontweight='bold')
    # Set x-ticks with bold and larger font
    plt.xticks(fontsize=14, fontweight='bold')

    # Annotate boxplot with Avg and SD
    stats = [
        (scans_per_subject.mean(), scans_per_subject.std()),
        (ages.mean(), ages.std())
    ]
    for i, (avg, sd) in enumerate(stats):
        plt.text(avg, i + 1.15, f"Avg={avg:.2f}, SD={sd:.2f}", fontsize=16, ha='center', color='black', fontweight='bold')

    plt.title("Scans and Age Distribution", fontsize=18, fontweight='bold')
    plt.grid(True)

    # Pie chart for sex distribution
    plt.subplot(1, 3, 3)
    wedges, texts, autotexts = plt.pie(sex_values, labels=sex_labels, autopct='%1.1f%%', startangle=90,
                                       colors=['pink', 'lightblue'], textprops={'fontsize': 16, 'fontweight': 'bold'}, radius=1.2)
    for i, count in enumerate(sex_values):
        wedges[i].set_edgecolor('black')  # Add edge for clarity
        plt.text(wedges[i].theta2 / 2, 0.75, f"{count}", fontsize=16, color='black', ha='center', fontweight='bold')
    plt.title("Sex Distribution", fontsize=18, fontweight='bold')

    # Adjust layout
    plt.tight_layout()

    # Save the figure (optional)
    plt.savefig("subset_dataset_demographics.png", dpi=300)  # Save as high-resolution image
    # plt.show()

def plot_boxplot(input_csv, columns_to_keep, y_label, fig_title, palette, n4 = False):
    if len(columns_to_keep) != len(palette):
        raise ValueError(f"Mismatch between columns to keep ({len(columns_to_keep)}) and palette size ({len(palette)}).")
    data = pd.read_csv(input_csv)
    print(len(data))
    
    new_data = data.drop_duplicates(subset=['scan_id'], ignore_index=True)
    print(len(new_data))
    for col in columns_to_keep:
        print(f'Stats for {col}')
        print(new_data[col].mean())
        print(new_data[col].std())
        print(new_data[col].mean())
        print(new_data[col].std())

    # sns.boxplot(x=data['CV_before_n4'])
    # sns.boxplot(x=data['CV_after_n4'])

    ax = sns.boxplot(
    x="variable", 
    y="value", 
    data=pd.melt(new_data[columns_to_keep]), palette=palette)
    # Update the y-axis label
    ax.set_ylabel(f'{y_label}', fontsize=18, fontweight='bold')
    # Remove x-axis label and update ticks
    ax.set(xlabel=None)
    bold_font = fm.FontProperties(weight='bold', size=14)
    ax.set_xticklabels(ax.get_xticklabels(), fontproperties=bold_font)
    ax.set_yticklabels(ax.get_yticklabels(), fontproperties=bold_font)

    plt.setp(ax.lines, color='k')
    plt.setp(ax.artists, edgecolor = 'k')
    if n4:
        ax.set_xticklabels(["Without N4", "With N4"], fontsize=18, fontweight='bold')
    else:
        ax.set_xticklabels(["Atlas-based", "BET"], fontsize=18, fontweight='bold') 
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'{fig_title}.png')
    # print(data[data['CV_before_n4'] == np.nan].count())
    # print(data[data['CV_before_n4'] > data['CV_after_n4']].count())
    # print(data[data['CV_before_n4'] < data['CV_after_n4']].count())

def plot_histograms(corrected_image_path, non_corrected_image_path):

    # # Calculate histograms for corrected and non-corrected images
    corrected_hist, corrected_bins = calculate_histogram(corrected_image_path)
    non_corrected_hist, non_corrected_bins = calculate_histogram(non_corrected_image_path)
    # # Plot the histograms
    plt.figure(figsize=(12, 4))

    # Plot for corrected image
    plt.subplot(1, 2, 1)
    plt.title("With N4 Correction", fontsize='18', fontweight='bold')
    plt.bar(corrected_bins[:-1], corrected_hist, width=np.diff(corrected_bins), color='blue', alpha=0.7, label='Corrected')
    plt.xlabel("Intensity", fontsize='14', fontweight='bold')
    plt.ylabel("Frequency", fontsize='14', fontweight='bold')
    plt.grid(True)
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.legend()

    # Plot for non-corrected image
    plt.subplot(1, 2, 2)
    plt.title("Without N4 Correction", fontsize='18', fontweight='bold')
    plt.bar(non_corrected_bins[:-1], non_corrected_hist, width=np.diff(non_corrected_bins), color='red', alpha=0.7, label='Non-Corrected')
    plt.xlabel("Intensity", fontsize='14', fontweight='bold')
    plt.ylabel("Frequency", fontsize='14', fontweight='bold')
    plt.grid(True)
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.legend()

    # Save and display the figure
    plt.tight_layout()
    plt.savefig("histogram_comparison_corrected_non_corrected.png")
 

if __name__ == "__main__":
    # input_csv = '/home/andjela/Documents/CP/trios_sorted_by_age.csv'
    # input_csv = '../data/CP/trios_sorted_by_age.csv'
    
    # transfo_type = 'rigid_affine'
    # abbey_path = '/home/GRAMES.POLYMTL.CA/andim/joplin-intra-inter/CP_rigid_trios/CP'
    # save_transform_paths_CP(input_csv, abbey_path, transfo_type)

    # input_csv = '/home/andjela/Documents/CP/trios_sorted_by_age_with_transforms.csv'
    # rel_path = '/home/andjela/joplin-intra-inter/CP_rigid_trios'
    # process_csv_and_calculate_scaling_factors('./data/CP/trios_sorted_by_age_with_transforms.csv')
    # input_csv = 'C:\\Users\\andje\\Downloads\\trios_sorted_by_age_with_transforms.csv'
    # input_csv = 'C:\\Users\\andje\\Downloads\\trios_sorted_by_age_BET.csv'
    # fit_mix_effects_model(input_csv, "global_volume")
    # fit_non_linear_mix_effects_model(input_csv, "global_volume", 'Intracranial Volume (cm³)')
    
    # create_rainbow_plot(input_csv, 'global_volume_BET', 'Intracranial Volume (cm³)')
    # results = analyze_stats_per_csv(input_csv, 'global_volume_BET')
    # print(results)
    # plot_non_increasing_volumes(results)
    # create_rainbow_plot(input_csv, 'scaling_avg', 'Scaling Avg')
    # analyze_mean_scaling_factors(input_csv)
    # results = analyze_stats_per_csv(input_csv)
    
    # plot_mean_scaling_factors(input_csv, 'scaling_avg', 'Scaling Avg')
    # plot_mean_scaling_factors(input_csv, 'scaling_y', 'Scaling Y')

    # process_csv_and_calculate_averages(input_csv)
    # create_rainbow_plot(input_csv, 'avg_intensity', 'Average Intensity')

    # output_path = 'trios_sorted_by_age_n4.csv'
    # # analyze_n4_correction(input_csv, output_path)
    # # analyze_stats_per_csv(output_path, 'n4_avg')
    # data = pd.read_csv(output_path)

    #####################################################################################
    ##################### N4 CORRECTION #################################################
    # corrected_image_path = "/home/GRAMES.POLYMTL.CA/andim/intra-inter-ddfs/work_dir/reg_n4_wdir/10006/PS14_001/wf/n4/PS14_001_corrected.nii.gz"  
    # non_corrected_image_path = "/home/GRAMES.POLYMTL.CA/andim/intra-inter-ddfs/CP_dataset/Year-4/sub-001-ses-001-T1w.nii.gz"  
    # non_corrected_image_path = '/home/GRAMES.POLYMTL.CA/andim/intra-inter-ddfs/work_dir2/cbf2mni_wdir/10006/PS14_001/wf/brainextraction/PS14_001_dtype.nii.gz'
    
    corrected_image_path = "/home/GRAMES.POLYMTL.CA/andim/intra-inter-ddfs/work_dir/reg_n4_wdir/10039/PS14_036/wf/n4/sub-071-ses-001-T1w_corrected.nii.gz"  
    non_corrected_image_path = "/home/GRAMES.POLYMTL.CA/andim/intra-inter-ddfs/CP_dataset/Year-3/sub-071-ses-001-T1w.nii.gz"
    plot_histograms(corrected_image_path, non_corrected_image_path)
    # input_csv = 'trios_sorted_by_age_n4.csv'
    # output_csv_path = 'trios_sorted_by_age_with_cv.csv'
    # # labels_of_interest = [2, 41]  # Labels for cerebral white matter
    # # analyze_n4_CV_for_seg_region(input_csv, output_csv_path, labels_of_interest)
    # plot_boxplot(output_csv_path, ['CV_before_n4', 'CV_after_n4'], 'Coefficient of Variation (σ/μ)', 'boxplot_bef_aft_n4', ['#FF4C4C', '#4C4CFF'], n4=True)

    # output_csv_path = 'trios_sorted_by_age_with_cv_new.csv'
    # label_dict = ExtractLabelsForRegions()
    # wm_labels = label_dict['WM']
    # analyze_n4_CV_for_seg_region(input_csv, output_csv_path, wm_labels)
    # plot_boxplot(output_csv_path, ['CV_before_n4', 'CV_after_n4'], 'Coefficient of Variation (σ/μ)', 'boxplot_bef_aft_n4_WM', ['#FF4C4C', '#4C4CFF'])

    #####################################################################################
    ##################### DATA DEMOGRAPHICS #############################################
    # Replace 'your_data.tsv' with the path to your input TSV file
    # plot_dataset_demographics("/home/GRAMES.POLYMTL.CA/andim/intra-inter-ddfs/all-participants.tsv")

    #####################################################################################
    ##################### SKULL-STRIPPING ###############################################
    # input_csv = 'trios_sorted_by_age_n4.csv'
    # plot_boxplot(input_csv, ['global_volume', 'global_volume_BET'], 'Intracranial Volume (cm³)', 'boxplot_skull_strip', ['#0A765F', '#69A1AB'])

    #####################################################################################
    ##################### VOLUME CALCULATIONS ###########################################
    # input_csv = 'trios_sorted_by_age_with_cv.csv'
    # output_csv_path = 'trios_sorted_by_age_with_segs.csv'
    # analyze_volume_for_regions(input_csv, output_csv_path)
    # create_rainbow_plot(output_csv_path, 'wm_vol_cm3', 'WM volume (cm³)')
    # create_rainbow_plot(output_csv_path, 'gm_vol_cm3', 'GM volume (cm³)')
    # fit_non_linear_mix_effects_model_wo_std(output_csv_path, 'wm_vol_cm3', 'WM Volume (cm³)', 'WM_fixed_effects_model')
    # fit_non_linear_mix_effects_model_wo_std(output_csv_path, 'gm_vol_cm3', 'GM Volume (cm³)', 'GM_fixed_effects_model')
    # fit_non_linear_mix_effects_model(output_csv_path, 'wm_vol_cm3', 'WM Volume (cm³)', 'WM_fixed_effects_std_model')
    # fit_non_linear_mix_effects_model(output_csv_path, 'gm_vol_cm3', 'GM Volume (cm³)', 'GM_fixed_effects__stdmodel')
    # plot_extrapolated_mean(output_csv_path, 'wm_vol_cm3', 'WM Volume (cm³)', 'WM_mixed_effects_std_model')
    # plot_extrapolated_mean(output_csv_path, 'gm_vol_cm3', 'GM Volume (cm³)', 'GM_mixed_effects_std_model')
    


    





    