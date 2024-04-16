import torch

# if we have both datasets the split is:
# 60% for nusc | 40% for lyft

# for training: 46778 samples (28130 + 18648)
# for validation: 10051 samples (6019 + 4032)

class MultiDataset(torch.utils.data.Dataset):

    def __init__(self, datasets, dataset_config, is_train):
        
        self.check_datasets(datasets)

        self.is_train = is_train

        self.dataset_config = dataset_config.clone()

        if self.has_nuscenes:
            from .nuscenes_dataloader import NuscData

            params = {'nusc': datasets['nuscenes']['nusc'],
                      'nusc_map': datasets['nuscenes']['nusc_map'],
                      'is_train': is_train,
                      'dataset_conf': dataset_config}

            self.nusc = NuscData(**params)

        if self.has_lyft:
            from .lyft_dataloader import LyftData

            params = {'lyft': datasets['lyft'],
                      'is_train': is_train,
                      'dataset_conf': dataset_config}
            
            self.lyft = LyftData(**params)
        
        self.set_start_idx()


    def check_datasets(self, dataset_obj):
        self.has_nuscenes = 'nuscenes' in dataset_obj.keys()
        self.has_lyft = 'lyft' in dataset_obj.keys()

        assert self.has_nuscenes or self.has_lyft, "There isn't a nuScenes or Lyft dataset object available"

        self.is_multidataset = self.has_lyft and self.has_nuscenes


    def set_start_idx(self):
        if self.has_nuscenes and self.has_lyft:
            self.nusc_idx = 0
            self.lyft_idx = len(self.nusc)

    def get_dataset(self, index):
        if self.is_multidataset:
            if index >= self.lyft_idx:
                index = index - self.lyft_idx
                dataset = self.lyft

            else:
                dataset = self.nusc

        elif self.has_nuscenes:
            dataset = self.nusc

        elif self.has_lyft:
            dataset = self.lyft

        return dataset, index


    def __len__(self):
        total = 0

        if self.has_nuscenes:
            total += len(self.nusc)
        if self.has_lyft:
            total += len(self.lyft)        

        return total