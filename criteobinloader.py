import os
import numpy as np
from torch.utils.data import Dataset
import torch
import time
import math
from tqdm import tqdm
import argparse


def _transform_features(x_int_batch,
                        x_cat_batch,
                        y_batch,
                        max_ind_range,
                        flag_input_torch_tensor=False):

    if max_ind_range > 0:
        x_cat_batch = x_cat_batch % max_ind_range

    if flag_input_torch_tensor:
        x_int_batch = torch.log(
            x_int_batch.clone().detach().type(torch.float) + 1)
        x_cat_batch = x_cat_batch.clone().detach().type(torch.long)
        y_batch = y_batch.clone().detach().type(torch.float32).view(-1, 1)
    else:
        x_int_batch = torch.log(
            torch.tensor(x_int_batch, dtype=torch.float) + 1)
        x_cat_batch = torch.tensor(x_cat_batch, dtype=torch.long)
        y_batch = torch.tensor(y_batch, dtype=torch.float32).view(-1, 1)

    batch_size = x_cat_batch.shape[0]
    feature_count = x_cat_batch.shape[1]
    lS_o = torch.arange(batch_size).reshape(-1, 1).repeat(feature_count, 1)

    return x_int_batch, lS_o, x_cat_batch.t(), y_batch.view(-1, 1)


class CriteoBinDataset(Dataset):
    """
    Binary version of the criteo dataset 
    """

    def __init__(self,
                 data_file,
                 counts_file,
                 batch_size=1,
                 max_ind_range=-1,
                 bytes_per_feature=4):

        self.tar_fea = 1
        self.den_fea = 13
        self.spa_fea = 26
        self.tad_fea = self.tar_fea + self.den_fea
        self.tot_fea = self.tad_fea + self.spa_fea

        self.batch_size = batch_size
        self.max_ind_range = max_ind_range
        self.bytes_per_entry = (bytes_per_feature * self.tot_fea * batch_size)

        self.num_entries = math.ceil(
            os.path.getsize(data_file) / self.bytes_per_entry)

        print("data file:", data_file, "num of batches:", self.num_entries)

        self.file = open(data_file, 'rb')
        print(f"{self.file} opened")

        with np.load(counts_file) as data:
            self.counts = data["counts"]

        self.m_den = 13

    def __len__(self):
        return self.num_entries

    def __getitem__(self, idx):
        self.file.seek(idx * self.bytes_per_entry, 0)
        raw_data = self.file.read(self.bytes_per_entry)
        array = np.frombuffer(raw_data, dtype=np.int32)
        tensor = torch.from_numpy(array).view(-1, self.tot_fea)
        return _transform_features(x_int_batch=tensor[:, 1:14],
                                   x_cat_batch=tensor[:, 14:],
                                   y_batch=tensor[:, 0],
                                   max_ind_range=self.max_ind_range,
                                   flag_input_torch_tensor=True)

    def __del__(self):
        self.file.close()


def _preprocess(args):
    train_files = [
        '{}_{}_reordered.npz'.format(args.input_data_prefix, day)
        for day in range(0, 23)
    ]

    test_valid_file = args.input_data_prefix + '_23_reordered.npz'

    os.makedirs(args.output_directory, exist_ok=True)
    for split in ['train', 'val', 'test']:
        print('Running preprocessing for split =', split)

        output_file = os.path.join(args.output_directory,
                                   '{}_data.bin'.format(split))

        input_files = train_files if split == 'train' else [test_valid_file]
        numpy_to_binary(input_files=input_files,
                        output_file_path=output_file,
                        split=split)


#def _test_bin():
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_directory', required=True)
    #parser.add_argument('--input_data_prefix', required=True)
    parser.add_argument('--split',
                        choices=['train', 'test', 'val'],
                        required=True)
    args = parser.parse_args()

    #_preprocess(args)

    binary_data_file = os.path.join(args.output_directory,
                                    '{}_data.bin'.format(args.split))

    counts_file = os.path.join(args.output_directory, 'day_fea_count.npz')
    dataset_binary = CriteoBinDataset(
        data_file=binary_data_file,
        counts_file=counts_file,
        batch_size=1,
    )
    #from dlrm_data_pytorch import CriteoDataset
    #from dlrm_data_pytorch import collate_wrapper_criteo_offset as collate_wrapper_criteo

    binary_loader = torch.utils.data.DataLoader(
        dataset_binary,
        batch_size=None,
        shuffle=False,
        num_workers=0,
        collate_fn=None,
        pin_memory=False,
        drop_last=False,
    )

    #original_dataset = CriteoDataset(
    #    dataset='terabyte',
    #    max_ind_range=10 * 1000 * 1000,
    #    sub_sample_rate=1,
    #    randomize=True,
    #    split=args.split,
    #    raw_path=args.input_data_prefix,
    #    pro_data='dummy_string',
    #    memory_map=True
    #)

    #original_loader = torch.utils.data.DataLoader(
    #    original_dataset,
    #    batch_size=2048,
    #    shuffle=False,
    #    num_workers=0,
    #    collate_fn=collate_wrapper_criteo,
    #    pin_memory=False,
    #    drop_last=False,
    #)

    #assert len(dataset_binary) == len(original_loader)
    #for i, (old_batch, new_batch) in tqdm(enumerate(zip(original_loader,
    #                                                    binary_loader)),
    #                                      total=len(dataset_binary)):

    #    for j in range(len(new_batch)):
    #        if not np.array_equal(old_batch[j], new_batch[j]):
    #            raise ValueError('FAILED: Datasets not equal')
    #    if i > len(dataset_binary):
    #        break
    #print('PASSED')
    train_iter = iter(binary_loader)
    element = next(train_iter)
    print(element)
    print(element[0].shape)
    #for i in range(50):
    #    train_features, train_labels = next(train_iter)
    #    print(f"train_features : {train_features}, train_labels: {train_labels}")

#if __name__=="__main__":
#    _test_bin()
