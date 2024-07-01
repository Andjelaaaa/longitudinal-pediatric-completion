import os
import numpy as np
import pandas as pd
import glob
import nibabel as nib
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import cv2
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
import torch

class CP(Dataset):
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



def load_and_preprocess_data():
    # Load the data
    data = np.load("data.npy")
    
    # Preprocess the data
    data = preprocess_data(data)
    
    # Create a tf.data.Dataset
    dataset = tf.data.Dataset.from_tensor_slices(data)
    
    return dataset