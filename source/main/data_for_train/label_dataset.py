from torch.utils.data import Dataset

voc = None
MAX_LENGTH = 100
NUM_WORKERS = 0
ROOT = '/source/'


class LabelDataset(Dataset):
    def __init__(self, list_data, label):
        super(LabelDataset, self).__init__()
        self.mentions = list(list_data)
        self.label = label

    def __len__(self):
        return len(self.mentions)

    def __getitem__(self, idx):
        return self.mentions[idx], self.label
