import imp


import torch
from torch.utils.data import Dataset
import h5py
import json
import os

class CaptionDataset(Dataset):
    """
    A pytroch Dataset class to be used in a Pytorch Dataloader to create batches.
    """

    def __init__(self,data_folder,data_name,split,transform=None):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, on of "TRAIN",'VAL' or 'TEST'
        :param transform:image transform pipeline
        """
        self.split = split
        assert self.split in {'TRAIN','VAL','TEST'}

        # open hdf5 files where images are stored
        file_path = os.path.join(data_folder,self.split + '_IMAGES_' + data_name + '.hdf5')
        self.h = h5py.File(file_path, 'r')
        self.imgs = self.h['images']
        
        # Captions per image
        self.cpi = self.h.attrs['captions_per_image']

        # Load encoded captions (completely into memory)
        cap_path = os.path.join(data_folder, self.split + '_CAPTIONS_' + data_name + '.json')
        with open(cap_path, 'r') as j:
            self.captions = json.load(j)

        # Load caption lengths (completely into memory)
        cap_lens_path = os.path.join(data_folder, self.split + '_CAPLENS_' + data_name + '.json')
        with open(cap_lens_path, 'r') as j:
            self.caplens = json.load(j)

        # Pytorch transformation pipeline for the image(normalizing,etc.)
        self.transform = transform

        # Total number of datapoints
        self.dataset_size = len(self.captions)

    def __getitem__(self,i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        img = torch.FloatTensor(self.imgs[ i // self.cpi]/255.)
        if self.transform is not None:
            img = self.transform(img)

        caption = torch.LongTensor(self.captions[i])
        caplen = torch.LongTensor([self.caplens[i]])

        if self.split is 'TRAIN':
            return img,caption,caplen
        else:
            # For validation of tetsing, also return all 'captions_per_image' captions to find BLEU-4 score
            all_captions = torch.LongTensor(
                self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
            return img, caption, caplen, all_captions

    def __len__(self):
        return self.dataset_size

