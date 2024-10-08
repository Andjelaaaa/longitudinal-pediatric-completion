import os
import numpy as np
import h5py
import pandas as pd
import shutil
import glob
import time
import nibabel as nib
import seaborn as sns
import SimpleITK as sitk
from statsmodels.graphics.functional import rainbowplot
from scipy.interpolate import make_interp_spline
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from itertools import combinations
import nibabel as nib
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
import torch
from torch.utils.data import Dataset
import torchio as tio
from torchio import transforms as tio_transforms

class CP(Dataset):
    def __init__(self, root_dir, age_csv, opt=None):
        """
        Args:
            root_dir (string): Directory with all the trios of images (nii.gz).
            age_csv (string): Path to the CSV file containing age information for target images.
            opt (optional): Options for dataset processing, such as image size or voxel size.
        """
        self.root_dir = root_dir

        # Load the CSV file with age and train/val/test information
        self.age_info = pd.read_csv(age_csv)

        # Filter the CSV file to include only the trios marked for training
        self.train_info = self.age_info[self.age_info["traintest"] == "train"].head(144)

        # Set image dimensions for resizing (if needed)
        # if opt is None or 'image_size' not in opt:
        #     self.img_size = (64, 64, 64)  # Default image size for 3D data (depth, height, width)
        # else:
        #     self.img_size = opt['image_size']

        # Set voxel size for resampling (default to 2mm isotropic)
        if opt is None or 'voxel_size' not in opt:
            self.voxel_size = (2, 2, 2)  # Default to 2mm isotropic voxel size
        else:
            self.voxel_size = opt['voxel_size']

        # Define the transform to resample the image to 2mm isotropic voxel size
        self.resample_transform = tio.Resample(self.voxel_size)

        # # Define 3D resizing transform (optional)
        # self.resize_transform = tio.Resize(self.img_size) if opt and 'image_size' in opt else None

        # Collect all available trios marked as "train"
        self.trio_paths = self.get_trio_paths()

    def get_trio_paths(self):
        """
        Collects the paths to all trios that are marked for training in the CSV file.
        """
        trio_paths = []
        for _, row in self.train_info.iterrows():
            subject_id = row["sub_id_bids"]
            trio_id = row["trio_id"]
            trio_dir = os.path.join(self.root_dir, f"{subject_id}", f"{trio_id}")

            # Each subject might have multiple trios, check for .nii.gz files in the directory
            nii_files = sorted([f for f in os.listdir(trio_dir) if f.endswith(".nii.gz")])

            # Ensure we have sets of 3 files (preceding, target, subsequent)
            for i in range(0, len(nii_files) - 2, 3):
                trio_paths.append((os.path.join(trio_dir, nii_files[i]),
                                   os.path.join(trio_dir, nii_files[i + 1]),
                                   os.path.join(trio_dir, nii_files[i + 2])))

        return trio_paths

    def __len__(self):
        return len(self.trio_paths)

    def load_and_normalize_nii(self, file_path):
        """
        Load a NIfTI file, normalize its pixel values to [0, 1], and apply resampling to the desired voxel size.
        """
        img = nib.load(file_path)
        img_data = img.get_fdata()

        # Normalize to [0, 1]
        img_data_min = np.min(img_data)
        img_data_max = np.max(img_data)

        if img_data_max > img_data_min:  # Avoid division by zero
            img_data = (img_data - img_data_min) / (img_data_max - img_data_min)
        else:
            img_data = np.zeros_like(img_data)

        # Convert to TorchIO ScalarImage for resampling, ensuring float64 data type
        tio_image = tio.ScalarImage(tensor=torch.tensor(img_data, dtype=torch.float64).unsqueeze(0))  # Add channel dimension

        # Apply resampling transform (e.g., to 2mm isotropic voxel size)
        resampled_image = self.resample_transform(tio_image)

        return resampled_image.data  # Return the resampled tensor

    def get_age_for_target(self, target_image_name):
        """
        Get the age of the target image from the CSV file.
        """
        age_row = self.train_info[self.train_info["scan_id"] == target_image_name].head(144)
        if age_row.empty:
            raise ValueError(f"Age information not found for image: {target_image_name}")

        return age_row["age"].values[0]

    def __getitem__(self, idx):
        # Get the file paths for the trio
        img_1_path, img_2_path, img_3_path = self.trio_paths[idx]

        # Load, normalize, and resample each image in the trio
        img_1 = self.load_and_normalize_nii(img_1_path)  # Preceding
        img_2 = self.load_and_normalize_nii(img_2_path)  # Target
        img_3 = self.load_and_normalize_nii(img_3_path)  # Subsequent

        # Optionally resize the 3D images using torchio
        # if self.resize_transform:
        #     img_1 = torch.tensor(self.resize_transform(img_1))
        #     img_2 = torch.tensor(self.resize_transform(img_2))
        #     img_3 = torch.tensor(self.resize_transform(img_3))

        # Get the target image's name and extract its age from the CSV
        target_image_name = os.path.basename(img_2_path).replace(".nii.gz", "")
        target_age = self.get_age_for_target(target_image_name)

        # Convert age to tensor
        age_tensor = torch.tensor([target_age], dtype=torch.float64)

        # Return the preceding, target, subsequent images and the age
        return img_1, img_2, img_3, age_tensor

class BCP(Dataset):
    def __init__(self, root='./data_midslice/affine-aligned-midslice/', transform=None, trainvaltest='train', opt = None):

        self.trainvaltest = trainvaltest
        self.imgdir = os.path.join(root, 'images/')
        self.targetname = opt.targetname

        if 'demoname' in opt:
            meta = pd.read_csv(os.path.join(root, opt.demoname), index_col=0)
        else:
            meta = pd.read_csv(os.path.join(root, 'demo-healthy-longitudinal.csv'), index_col=0)

        meta = meta[meta.trainvaltest == trainvaltest].reset_index()
        IDunq = np.unique(meta['Site-ID'])
        index_combination = np.empty((0, 2))
        for sid in IDunq:
            indices = np.where(meta['Site-ID'] == sid)[0]
            ### all possible pairs
            tmp_combination = np.array(
                np.meshgrid(np.array(range(len(indices))), np.array(range(len(indices))))).T.reshape(-1, 2)
            index_combination = np.append(index_combination, indices[tmp_combination], 0)

        if opt == None:
            img_height, img_width = [320, 300]
        else:
            img_height, img_width = opt.image_size

        self.resize = transforms.Compose([
            transforms.Resize((img_height, img_width), Image.BICUBIC),
            transforms.ToTensor(),
        ])

        self.index_combination = index_combination
        self.transform = transform
        self.demo = meta

    def __getitem__(self, index):
        index1, index2 = self.index_combination[index]
        target1, target2 = self.demo[self.targetname][index1], self.demo[self.targetname][index2]
        img1 = Image.open(os.path.join(self.imgdir, self.demo.fname[index1]))
        img1 = self.resize(img1) 
        img2 = Image.open(os.path.join(self.imgdir, self.demo.fname[index2]))
        img2 = self.resize(img2) 

        if self.transform:
            augmentation = transforms.Compose([
                transforms.RandomApply(torch.nn.ModuleList(
                    [transforms.RandomAffine(degrees=(-10, 10), translate=(0.05,0.05), #scale=(0.9, 1.1),
                                             interpolation=InterpolationMode.BILINEAR)]),
                    p=0.5),
            ])

            img1 = augmentation(img1)
            img2 = augmentation(img2)

        return [np.array(img1), target1], [np.array(img2), target2]

    def __len__(self):
        return len(self.index_combination)
    
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

def process_csv_and_calculate_scaling_factors(csv_file_path, rel_path):
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
        
        print(transform_path)
        rel_transform_path = f"{rel_path}/{transform_path[6:]}"  # Path to the ANTs transform file

        try:
            # Read the affine matrix from the ANTs transform
            affine_matrix, _ = read_ANTs_transform(rel_transform_path)
            
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

    
def preprocess_BCP(filename, path):
    
    # filename = 'HCPBaby_image03_xnat_T1wT2w_2022-01-26.xlsx'
    # df_scans = pd.read_excel(f'/home/adimitri/Downloads/{filename}')

    # Read all sheets into a dictionary of DataFrames
    df_sheets = pd.read_excel(f'{path}{filename}', sheet_name=None)

    # Display the sheet names
    print(df_sheets.keys())

    # Access individual DataFrames using the sheet names
    df_scans_T1w = df_sheets['T1w']  
    df_scans_T2w = df_sheets['T2w'] 
    df_scans_vars = df_sheets['NameVar_Definition']  

    # For T1w scans
    pass_T1w = df_scans_T1w[df_scans_T1w['qc_outcome'] == 'pass'].reset_index()

    cleaned_pass_T1w = pass_T1w.drop_duplicates(subset=['interview_age', 'Site-ID'], keep='first').drop(columns=['index'])
    T1w_onecount = cleaned_pass_T1w['Site-ID'].value_counts() > 1
    T1w_long_subjects = np.array(T1w_onecount[T1w_onecount].index)
    data_T1w_long_subjects = cleaned_pass_T1w[cleaned_pass_T1w['Site-ID'].isin(T1w_long_subjects)].reset_index().drop(columns=['index'])


    data_T1w_long_subjects.to_csv('./data/bcp/demo-healthy-longitudinal.csv')


    # train/val/test
    data_T1w_long_subjects = pd.read_csv('./data/bcp/demo-healthy-longitudinal.csv')
    HClongsubjects = np.unique(data_T1w_long_subjects['Site-ID'])
    totalN = len(HClongsubjects)
    trainN, valN, testN = int(totalN*0.6), int(totalN*0.2), int(totalN*0.2)
    permutedsubjects = HClongsubjects[np.random.permutation(range(totalN))]

    trainvaltest = np.zeros(totalN).astype('str')
    trainvaltest[:trainN] = 'train'
    trainvaltest[trainN:trainN+valN] = 'val'
    trainvaltest[trainN+valN:] = 'test'
    # assign trainvlatest
    demo_trainvaltest = np.zeros(len(data_T1w_long_subjects)).astype('str')
    for i in range(len(permutedsubjects)):
        indices = np.where(data_T1w_long_subjects['Site-ID'] == permutedsubjects[i])[0]
        demo_trainvaltest[indices] = trainvaltest[i]

    data_T1w_long_subjects['trainvaltest'] = demo_trainvaltest
    data_T1w_long_subjects.to_csv('./data/bcp/demo-healthy-longitudinal.csv')

    set(data_T1w_long_subjects['Site-ID'][data_T1w_long_subjects['trainvaltest'] == 'val'])\
        .intersection(set(data_T1w_long_subjects['Site-ID'][data_T1w_long_subjects['trainvaltest'] == 'test']))

    # Rigid atlas building

    # local_path = '/home/adimitri/empenn-nas-stk/share/dbs/HCP-Baby_Data/HCP-Baby_Data_MRI/image03/'
    # full_paths = []
    # for i in range(len(data_T1w_long_subjects)):
    #     # Retrieve relative path
    #     path_components = data_T1w_long_subjects.image_file[i].split('/')
    #     last_six_elements = path_components[-6:]
    #     relative_path = '/'.join(last_six_elements)

    #     # Split the string by '-'
    #     parts = path_components[-1]#.split('-')

    #     # Extract the first part
    #     subject_id = parts[:-len('.nii.gz')]#'_'.join(parts[:-1])

    #     # Fetch full path
    #     full_path = f'{local_path}{relative_path}'
    #     full_paths.append(full_path)
    # np.savetxt('/home/adimitri/learning-to-compare-longitudinal-images/data_midslice/affine-alignment/imagelist.csv', np.array(full_paths), fmt='%s')

    outputPath = '/home/adimitri/learning-to-compare-longitudinal-images/data_midslice/affine-alignment/'
    # inputPathcsv = '/home/adimitri/learning-to-compare-longitudinal-images/data_midslice/affine-alignment/imagelist.csv'

    # os.system(f'antsMultivariateTemplateConstruction2.sh \
    #               -d 3 \
    #               -o {outputPath}T_ \
    #               -i 1 \
    #               -g 0.25 \
    #               -j 4 \
    #               -k 1 \
    #               -v 10 \
    #               -c 2 \
    #               -q 100x100x70x20 \
    #               -n 0 \
    #               -r 0 \
    #               -m MI \
    #               -l 1 \
    #               -t Affine \
    #               {inputPathcsv}')


    # get mid save png with non-aligned images
    # sliceidx = 160
    # outdir = './data_midslice/affine-aligned-midslice/images_without_align/'

    # local_path = '/home/adimitri/empenn-nas-stk/share/dbs/HCP-Baby_Data/HCP-Baby_Data_MRI/image03/'

    # for i in range(len(data_T1w_long_subjects)):
    #     # Retrieve relative path
    #     path_components = data_T1w_long_subjects.image_file[i].split('/')
    #     last_six_elements = path_components[-6:]
    #     relative_path = '/'.join(last_six_elements)

    #     # Split the string by '-'
    #     parts = path_components[-1]#.split('-')

    #     # Extract the first part
    #     subject_id = parts[:-len('.nii.gz')]#'_'.join(parts[:-1])

    #     # Fetch full path
    #     full_path = f'{local_path}{relative_path}'

    #     # Load the 3D image data from the file
    #     image = np.asanyarray(np.squeeze(nib.load(full_path).dataobj, axis=-1))[:, sliceidx, :]

    #     imagerescaled = (((image - image.min()) / (image.max() - image.min())) * 256).astype(np.uint8)
    #     image2d = Image.fromarray(imagerescaled)
    #     fname2d = data_T1w_long_subjects['Site-ID'].iloc[i] + '_' + str(data_T1w_long_subjects['interview_age'].iloc[i])
    #     image2d.save(os.path.join(outdir, f'{fname2d}.png'))

    # get mid save png with aligned images to template
    # sliceidx = 104
    outdir = './data_midslice/affine-aligned-midslice/images/'
    # imgs_aligned_template_paths = glob.glob(os.path.join(outputPath, 'T_template-modality0*'))

    # for i in range(len(data_T1w_long_subjects)):
    #     # Retrieve relative path
    #     path_components = data_T1w_long_subjects.image_file[i].split('/')
    #     filename = path_components[-1].replace(".nii.gz", "")

    #     # Check if the filename is in any of the full paths
    #     for path in imgs_aligned_template_paths:
    #         if filename in path:
    #             full_path = path
    #             # Load the 3D image data from the file
    #             image = np.asanyarray(nib.load(full_path).dataobj)[:, :, sliceidx].T

    #             imagerescaled = (((image - image.min()) / (image.max() - image.min())) * 256).astype(np.uint8)
    #             image2d = Image.fromarray(imagerescaled)
    #             fname2d = data_T1w_long_subjects['Site-ID'].iloc[i] + '_' + str(data_T1w_long_subjects['interview_age'].iloc[i])
    #             image2d.save(os.path.join(outdir, f'{fname2d}.png'))
    #             break

    # data_T1w_long_subjects_slice = pd.read_csv('./data/bcp/demo-healthy-longitudinal.csv', index_col=[0])
    # data_T1w_long_subjects_slice['fname'] = outdir + data_T1w_long_subjects['Site-ID'] + '_' + str(data_T1w_long_subjects['interview_age'])+ '.png'
    data_T1w_long_subjects['fname'] = data_T1w_long_subjects['Site-ID'] + '_' + data_T1w_long_subjects['interview_age'].astype(str) + '.png'

    # # match mriinfo
    # # label = data_T1w_long_subjects_slice['id-session'].str.replace('-ses-', '_MR_')
    # # sex = []
    # # age = []
    # # for l in range(len(label)):
    # #     labelindex = np.where(mriinfo.Label == label.iloc[l])[0]
    # #     sex.append(np.array(mriinfo['M/F'].iloc[labelindex])[0])
    # #     age.append(np.array(mriinfo['Age'].iloc[labelindex])[0])

    # # data_T1w_long_subjects['sex'] = sex
    # # data_T1w_long_subjects['age'] = age
    # # data_T1w_long_subjects_slice['sex'] = sex
    # # data_T1w_long_subjects_slice['age'] = age

    # data_T1w_long_subjects.to_csv('/nfs04/data/OASIS3/demo/demo-healthy-longitudinal.csv')
    # data_T1w_long_subjects_slice.to_csv(f'{outdir}/demo-healthy-longitudinal.csv')

    # time point
    unqID = np.unique(data_T1w_long_subjects['Site-ID'])
    timepoint = np.zeros(len(data_T1w_long_subjects)).astype('str')
    for s in range(len(unqID)):
        subjectidx = np.where(data_T1w_long_subjects['Site-ID'] == unqID[s])[0]
        sortedage = np.sort(np.unique(data_T1w_long_subjects['interview_age'].iloc[subjectidx]))
        # save sorted age index
        for sidx in subjectidx:
            timepoint[sidx] = np.where(sortedage == data_T1w_long_subjects['interview_age'].iloc[sidx])[0][0]


    # # data_T1w_long_subjects_slice['timepoint'] = timepoint.astype('int')
    data_T1w_long_subjects['timepoint'] = timepoint.astype('int')
    data_T1w_long_subjects.to_csv('./data/bcp/demo-healthy-longitudinal.csv')
    # # data_T1w_long_subjects_slice.to_csv(f'{outdir}/demo-healthy-longitudinal.csv')

    # # data_T1w_long_subjects_slice.fname = 'images/' + data_T1w_long_subjects['Site-ID'] + '_' + data_T1w_long_subjects['session-id'] + '.png'
    # # data_T1w_long_subjects_slice.to_csv(f'{outdir}/demo-healthy-longitudinal.csv')

# Function to perform rigid registration
def perform_registration(mov_path, fix_path, scan_id_mov, scan_id_fix, save_path):
    print(f"Registering {os.path.basename(mov_path)} to {os.path.basename(fix_path)}")
    os.system(f"antsRegistration --dimensionality 3 --float 0 " \
            f"--output [ {save_path}/mov2fix_{scan_id_mov}_{scan_id_fix}, {save_path}/{scan_id_mov}.nii.gz] " \
            f"--interpolation Linear " \
            f"--winsorize-image-intensities [0.005,0.995] " \
            f"--use-histogram-matching 1 " \
            f"--write-composite-transform 1 " \
            f"--transform Rigid[0.1] " \
            f"--metric Mattes[ {fix_path}, {mov_path}, 1, 32, Regular, 0.3 ] " \
            f"--convergence [500x250x100,1e-6,10] " \
            f"--shrink-factors 4x2x1 " \
            f"--smoothing-sigmas 2x1x0vox " \
            f"--verbose 1"
            )
    
# Function to perform affine registration
def perform_affine_registration(mov_path, fix_path, scan_id_mov, scan_id_fix, transfo_type, save_path):
    # Can perform rigid + affine by giving as input the rigid pre-registred images
    print(f"Registering {os.path.basename(mov_path)} to {os.path.basename(fix_path)}")
    os.system(f"antsRegistration --float 0 --collapse-output-transforms 1 --dimensionality 3 " \
            f"--initial-moving-transform [ {fix_path}, {mov_path}, 1 ] " \
            f"--initialize-transforms-per-stage 0 " \
            f"--interpolation Linear " \
            f"--output [ {save_path}/{transfo_type}_mov2fix_{scan_id_mov}_{scan_id_fix}, {save_path}/{scan_id_mov}_{transfo_type}.nii.gz ] " \
            f"--transform Affine[ 0.1 ] " \
            f"--metric Mattes[ {fix_path}, {mov_path}, 1, 32, Regular, 0.3 ] " \
            f"--convergence [ 500x250x100, 1e-6, 10 ] " \
            f"--shrink-factors 4x2x1 " \
            f"--smoothing-sigmas 2x1x0vox " \
            f"--use-histogram-matching 1 " \
            f"--winsorize-image-intensities [ 0.005, 0.995 ] " \
            f"--write-composite-transform 1 " \
            f"--verbose 1"
            )
    # os.system(f"antsRegistration --float 0 --dimensionality 3 " \
    #         f"--initial-moving-transform [ {fix_path}, {mov_path}, 1 ] " \
    #         f"--initialize-transforms-per-stage 0 " \
    #         f"--interpolation Linear " \
    #         f"--output [ {save_path}/mov2fix_{scan_id_mov}_{scan_id_fix}, {save_path}/{scan_id_mov}.nii.gz ] " \
    #         f"--transform Rigid[ 0.1 ] " \
    #         f"--metric Mattes[ {fix_path}, {mov_path}, 1, 32, Regular, 0.3 ] " \
    #         f"--convergence [ 500x250x100, 1e-6, 10 ] " \
    #         f"--shrink-factors 4x2x1 " \
    #         f"--smoothing-sigmas 2x1x0vox " \
    #         f"--use-histogram-matching 1 " \
    #         f"--transform Affine[ 0.1 ] " \
    #         f"--metric Mattes[ {fix_path}, {mov_path}, 1, 32, Regular, 0.3 ] " \
    #         f"--convergence [ 500x250x100, 1e-6, 10 ] " \
    #         f"--shrink-factors 4x2x1 " \
    #         f"--smoothing-sigmas 2x1x0vox " \
    #         f"--use-histogram-matching 1 " \
    #         f"--winsorize-image-intensities [ 0.005, 0.995 ] " \
    #         f"--write-composite-transform 1 " \
    #         f"--verbose 1"
    #         )

# Function to skull-strip an image
def skull_strip(image_path, scan_id):
    # Placeholder for actual skull-stripping code
    # e.g., using an image processing library FSL BET
    if scan_id not in ['PS14_001', 'PS14_005']:
        os.system(f'/usr/local/fsl/bin/bet {image_path} ' \
          f'{os.path.dirname(image_path)}/{scan_id}_skull.nii.gz -f 0.5 -g 0')
    else:
        os.system(f'/usr/local/fsl/bin/bet {image_path} ' \
            f'{os.path.dirname(image_path)}/{scan_id}_skull_it_1.nii.gz -f 0.8 -g 0')
        os.system(f'/usr/local/fsl/bin/bet {os.path.dirname(image_path)}/{scan_id}_skull_it_1.nii.gz ' \
          f'{os.path.dirname(image_path)}/{scan_id}_skull.nii.gz -f 0.3 -g 0')
    print(f"Skull-stripped {os.path.basename(image_path)}")
    
def create_df_CP(df, work_dir_rel_path, save_path):

    # Add a new column 'n4_path' with the specified path format
    df['n4_path'] = df.apply(lambda row: f'{work_dir_rel_path}/work_dir/reg_n4_wdir/{row.participant_id}/{row.scan_id}/wf/n4/{row.scan_id}_corrected.nii.gz', axis=1)

    # Filter to keep only subjects with >=3 timepoints
    data_T1w_long_subjects = df.groupby('sub_id_bids').filter(lambda x: len(x) >= 3)

    # Count the number of subjects with >=3 timepoints
    HClongsubjects = np.unique(data_T1w_long_subjects['sub_id_bids'])

    # train/val/test
    totalN = len(HClongsubjects)
    trainN = int(totalN*0.8)
    permutedsubjects = HClongsubjects[np.random.permutation(range(totalN))]

    traintest = np.zeros(totalN).astype('str')
    traintest[:trainN] = 'train'
    traintest[trainN:] = 'test'
    # assign trainvlatest
    demo_trainvaltest = np.zeros(len(data_T1w_long_subjects)).astype('str')
    for i in range(len(permutedsubjects)):
        indices = np.where(data_T1w_long_subjects['sub_id_bids'] == permutedsubjects[i])[0]
        demo_trainvaltest[indices] = traintest[i]

    data_T1w_long_subjects['traintest'] = demo_trainvaltest
    
    # Save the new DataFrame to a CSV file
    data_T1w_long_subjects.to_csv("./data/CP/ind_data.csv", index=False)
    # set(data_T1w_long_subjects['sub_id_bids'][data_T1w_long_subjects['traintest'] == 'val'])\
    #     .intersection(set(data_T1w_long_subjects['sub_id_bids'][data_T1w_long_subjects['traintest'] == 'test']))
    
    # data_T1w_long_subjects.to_csv('./data/CP/CP-healthy-longitudinal.csv')

    # # Filter to keep only subjects with >3 sessions
    # subjects_with_more_than_3_sessions = df.groupby('sub_id_bids').filter(lambda x: len(x) > 3)

    # # Count the number of trios that can be formed for each subject
    # trio_count = subjects_with_more_than_3_sessions.groupby('sub_id_bids').apply(lambda x: len(list(combinations(x['session'], 3)))).sum()
    
    # # Count the number of subjects that have exactly 3 time points
    # subjects_with_exactly_3_sessions = df.groupby('sub_id_bids').filter(lambda x: len(x) == 3).sub_id_bids.nunique()

    # Initialize a list to store the new rows
    new_rows = []
    trio_number = 1  # Initialize a counter for trio numbering

    # Iterate over each subject
    for sub_id, group in data_T1w_long_subjects.groupby('sub_id_bids'):
        if len(group) == 3:
            # If the subject has exactly 3 timepoints, just sort them by age
            sorted_trio = group.sort_values(by='age')
            sorted_trio['trio_id'] = f'trio-{trio_number:03d}'  # Assign trio number

            # Add a 'path' column for each trio based on participant_id and scan_id
            sorted_trio['path'] = sorted_trio.apply(
                lambda row: f'{work_dir_rel_path}/work_dir/reg_n4_wdir/{row.participant_id}/{row.scan_id}/wf/n4/{row.scan_id}_skull.nii.gz', axis=1)
            new_rows.extend(sorted_trio.itertuples(index=False))
            trio_number += 1  # Increment the trio number
        elif len(group) > 3:
            # Generate all possible trios of sessions for subjects with more than 3 sessions
            for trio in combinations(group.itertuples(index=False), 3):
                # Sort the trio by age
                sorted_trio = sorted(trio, key=lambda x: x.age)
                # Convert the sorted trio to a DataFrame
                sorted_trio_df = pd.DataFrame(sorted_trio)
                sorted_trio_df.columns = group.columns  # Maintain the original column names
                sorted_trio_df['trio_id'] = f'trio-{trio_number:03d}'  # Assign trio number
                sorted_trio_df['path'] = sorted_trio_df.apply(
                    # lambda row: f'/home/andjela/joplin-intra-inter/work_dir/reg_n4_wdir/{row.participant_id}/{row.scan_id}/wf/n4/{row.scan_id}_skull.nii.gz', axis=1)
                    lambda row: f'{work_dir_rel_path}/work_dir2/cbf2mni_wdir/{row.participant_id}/{row.scan_id}/wf/brainextraction/{row.scan_id}_dtype.nii.gz', axis=1)
                new_rows.extend(sorted_trio_df.itertuples(index=False))
                trio_number += 1  # Increment the trio number

    # Create a new DataFrame from the new rows
    trios_df = pd.DataFrame(new_rows, columns=list(data_T1w_long_subjects.columns) + ['trio_id', 'path'])

    # Save the new DataFrame to a CSV file
    trios_df.to_csv(save_path, index=False)

    return trios_df

def preprocess_CP(trios_df):
    # # Skull-strip each N4 corrected scan with 2 times bet command
    # start_time = time.time()
    # # Only to be done on participants which have multiple time points
    # for index, row in df.iterrows():
    #     if row['scan_id'] in trios_df['scan_id'].values:
    #         print(f'Scan nbr: {index}')
    #         skull_strip(row.n4_path, row.scan_id)
    # end_time = time.time()

    # # Calculate the duration
    # duration = end_time - start_time

    # # Print the time taken
    # print(f"Time taken for skull stripping: {duration} seconds")
    
    # Rigid reg to middle time point
    processed_pairs = set()
    # Dictionary to store previous paths for already registered pairs
    saved_paths_for_pairs = {}
    # Iterate through each subject in the DataFrame
    for sub_id, group in trios_df.groupby('sub_id_bids'):
        # Since the data is grouped by subject, each group represents a subject with multiple trios
        trios = [group.iloc[i:i+3] for i in range(0, len(group), 3)]
        
        for trio in trios:
            # Extract the paths for the trio
            path_1 = trio.iloc[0]['path']
            path_2 = trio.iloc[1]['path']  # This is the reference
            path_3 = trio.iloc[2]['path']

            # Create a directory to save the preprocessed images if it doesn't exist
            save_path = f'./data/CP/{sub_id}/{trio.iloc[1]["trio_id"]}/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            # Create a tuple to represent the scan pair (scan_1 -> scan_2)
            pair_1_to_2 = (trio.iloc[0]['scan_id'], trio.iloc[1]['scan_id'])
            pair_3_to_2 = (trio.iloc[2]['scan_id'], trio.iloc[1]['scan_id'])

            # Save scan_2 in dedicated space (./data/CP/sub_id_bids/trio_id/) if not already there
            if not os.path.exists(f'{save_path}/{os.path.basename(path_2)}'):
                shutil.copy(path_2, f'{save_path}/{trio.iloc[1]["scan_id"]}.nii.gz')

            # If pair_1_to_2 has been processed, copy the previous scan_1 to the current trio directory
            if pair_1_to_2 not in saved_paths_for_pairs:
                # If the pair has not been processed, perform registration and save the path
                perform_registration(path_1, path_2, trio.iloc[0]['scan_id'], trio.iloc[1]['scan_id'], save_path)
                # perform_affine_registration(path_1, path_2, trio.iloc[0]['scan_id'], trio.iloc[1]['scan_id'], save_path)
                # Save the path where scan_1 is saved
                saved_paths_for_pairs[pair_1_to_2] = f'{save_path}/{trio.iloc[0]["scan_id"]}.nii.gz'
                # Add the pair to the processed set
                processed_pairs.add(pair_1_to_2)
            else:
                previous_scan_1_path = saved_paths_for_pairs[pair_1_to_2]
                shutil.copy(previous_scan_1_path, f'{save_path}/{trio.iloc[0]["scan_id"]}.nii.gz')

            # Perform registration for scan_3 -> scan_2 if it hasn't been processed yet
            if pair_3_to_2 not in processed_pairs:
                perform_registration(path_3, path_2, trio.iloc[2]['scan_id'], trio.iloc[1]['scan_id'], save_path)
                # perform_affine_registration(path_3, path_2, trio.iloc[2]['scan_id'], trio.iloc[1]['scan_id'], save_path)
                # Save the path where scan_3 is saved
                saved_paths_for_pairs[pair_3_to_2] = f'{save_path}/{trio.iloc[2]["scan_id"]}.nii.gz'
                # Add the pair to the processed set
                processed_pairs.add(pair_3_to_2)
            else:
                # Copy the already registered scan_3 from previous trio to the current trio directory
                previous_scan_3_path = saved_paths_for_pairs[pair_3_to_2]
                shutil.copy(previous_scan_3_path, f'{save_path}/{trio.iloc[2]["scan_id"]}.nii.gz')

            print('Done with trio:', trio.iloc[1]["trio_id"])

def preprocess_affine_CP(trios_df, transfo_type):
    # transfo_type = affine or rigid_affine
    # Rigid reg to middle time point
    processed_pairs = set()
    # Dictionary to store previous paths for already registered pairs
    saved_paths_for_pairs = {}
    # Iterate through each subject in the DataFrame
    for sub_id, group in trios_df.groupby('sub_id_bids'):
        # Since the data is grouped by subject, each group represents a subject with multiple trios
        trios = [group.iloc[i:i+3] for i in range(0, len(group), 3)]
        
        for trio in trios:
            save_path = f'./data/CP/{sub_id}/{trio.iloc[1]["trio_id"]}'
            # Extract the paths for the trio
            if transfo_type == 'affine':
                path_1 = trio.iloc[0]['path']
                path_2 = trio.iloc[1]['path']  # This is the reference
                path_3 = trio.iloc[2]['path']
            # If rigid_affine transfo
            else:
                path_1 = f'{save_path}/{trio.iloc[0]["scan_id"]}.nii.gz'
                path_2 = f'{save_path}/{trio.iloc[1]["scan_id"]}.nii.gz'  # This is the reference
                path_3 = f'{save_path}/{trio.iloc[2]["scan_id"]}.nii.gz'

            # Create a tuple to represent the scan pair (scan_1 -> scan_2)
            pair_1_to_2 = (trio.iloc[0]['scan_id'], trio.iloc[1]['scan_id'])
            pair_3_to_2 = (trio.iloc[2]['scan_id'], trio.iloc[1]['scan_id'])

            # If pair_1_to_2 has been processed, copy the previous scan_1 to the current trio directory
            if pair_1_to_2 not in saved_paths_for_pairs:
                # If the pair has not been processed, perform registration and save the path
                perform_affine_registration(path_1, path_2, trio.iloc[0]['scan_id'], trio.iloc[1]['scan_id'], transfo_type, save_path)
                # Save the path where scan_1 is saved
                saved_paths_for_pairs[pair_1_to_2] = f'{save_path}/{trio.iloc[0]["scan_id"]}_{transfo_type}.nii.gz'
                # Add the pair to the processed set
                processed_pairs.add(pair_1_to_2)
            else:
                previous_scan_1_path = saved_paths_for_pairs[pair_1_to_2]
                shutil.copy(previous_scan_1_path, f'{save_path}/{trio.iloc[0]["scan_id"]}_{transfo_type}.nii.gz')

            # Perform registration for scan_3 -> scan_2 if it hasn't been processed yet
            if pair_3_to_2 not in processed_pairs:
                perform_affine_registration(path_3, path_2, trio.iloc[2]['scan_id'], trio.iloc[1]['scan_id'], transfo_type, save_path)
                # Save the path where scan_3 is saved
                saved_paths_for_pairs[pair_3_to_2] = f'{save_path}/{trio.iloc[2]["scan_id"]}_{transfo_type}.nii.gz'
                # Add the pair to the processed set
                processed_pairs.add(pair_3_to_2)
            else:
                # Copy the already registered scan_3 from previous trio to the current trio directory
                previous_scan_3_path = saved_paths_for_pairs[pair_3_to_2]
                shutil.copy(previous_scan_3_path, f'{save_path}/{trio.iloc[2]["scan_id"]}_{transfo_type}.nii.gz')

            print('Done with trio:', trio.iloc[1]["trio_id"])

def save_transform_paths_CP(csv_file_path, transfo_type):
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
            transform_1_to_2 = f'./data/CP/{sub_id}/{trio_id}/{transfo_type}_mov2fix_{pair_1_to_2[0]}_{pair_1_to_2[1]}Composite.h5'
            transform_3_to_2 = f'./data/CP/{sub_id}/{trio_id}/{transfo_type}_mov2fix_{pair_3_to_2[0]}_{pair_3_to_2[1]}Composite.h5'

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

def load_and_preprocess_data():
    
    # Preprocess the data
    filename = 'all-participants.tsv'
    path = './data/CP'
    # Correct way to read the TSV data into a pandas DataFrame
    df = pd.read_csv(f'{path}/{filename}', sep='\t')
    # # joplin_path = '/home/GRAMES.POLYMTL.CA/andim/intra-inter-ddfs'
    # andjela_path = '/home/andjela/joplin-intra-inter'
    abbey_path = '/home/GRAMES.POLYMTL.CA/andim/joplin-intra-inter/CP_rigid_trios/CP'
    save_path = "./data/CP/trios_sorted_by_age.csv"
    trios_data = create_df_CP(df, abbey_path, save_path)
    # preprocess_CP(trios_data)

    # # # Preprocess the data with affine registration
    # trios_data = pd.read_csv(abbey_path)
    # trios_data = pd.read_csv('/home/andjela/Documents/CP/trios_sorted_by_age.csv')

    preprocess_affine_CP(trios_data, 'rigid_affine')
    
    # # Create a tf.data.Dataset
    # dataset = tf.data.Dataset.from_tensor_slices(data)
    
    # return dataset

if __name__ == "__main__":
    load_and_preprocess_data()
    # input_csv = './data/CP/trios_sorted_by_age.csv'  # Path to your input CSV
    # input_csv = '/home/andjela/Documents/CP/trios_sorted_by_age.csv'
    # transfo_type = 'rigid_affine'
    # save_transform_paths_CP(input_csv, transfo_type)

    # input_csv = '/home/andjela/Documents/CP/trios_sorted_by_age_with_transforms.csv'
    # rel_path = '/home/andjela/joplin-intra-inter/CP_rigid_trios'
    # process_csv_and_calculate_scaling_factors(input_csv, rel_path)
    # create_rainbow_plot(input_csv, 'scaling_avg', 'Scaling Avg')

    # process_csv_and_calculate_averages(input_csv)
    # create_rainbow_plot(input_csv, 'avg_intensity', 'Average Intensity')
    
   