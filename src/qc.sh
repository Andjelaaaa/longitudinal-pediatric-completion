#!/bin/bash

# Determine the directory where the script is located
SCRIPT_DIR=$(dirname "$(realpath "$0")")

# Base directory relative to the script's directory
BASE_DIR="${SCRIPT_DIR}/../data/$1/"


# Check if the user provided the data subfolder argument (like CP or BCP)
if [ -z "$1" ]; then
    echo "Usage: $0 <data-name>"
    exit 1
fi

# Check if the base directory exists
if [ ! -d "${BASE_DIR}" ]; then
    echo "Error: Directory ${BASE_DIR} does not exist."
    exit 1
fi

# Loop through all subfolders named "sub-*" in the base directory
for sub_dir in ${BASE_DIR}/sub-*; do
    # Loop through all subfolders named "trio-*" in each "sub-*" folder
    for trio_dir in ${sub_dir}/trio-*; do
        # Collect the three .nii.gz files in the current "trio-*" folder
        nii_files=(${trio_dir}/*.nii.gz)

        # Check if there are exactly three .nii.gz files
        if [ ${#nii_files[@]} -eq 3 ]; then
            echo "Opening ${nii_files[0]}, ${nii_files[1]}, ${nii_files[2]} in fsleyes..."
            
            # Open the three files in fsleyes
            fsleyes "${nii_files[0]}" "${nii_files[1]}" "${nii_files[2]}"
            
            # Wait until fsleyes is closed before moving to the next trio
            echo "fsleyes closed, moving to the next trio..."
        else
            echo "Skipping ${trio_dir}, expected 3 .nii.gz files but found ${#nii_files[@]}"
        fi
    done
done
