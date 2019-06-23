from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader


@dataclass(eq=False)
class FizBuzDataset(Dataset):
    """ Dataset class sublcassed from torch's dataset utility
    Args:
            input_size (int): input size of a datapoint or in this case, the binary size
            start (int): whole number from where the dataset starts counting, exclusive
            end (int): whole number till where the dataset keeps counting, inclusive
    """
    input_size: int = 10
    start: int = 0
    end: int = 1000

    def encoder(self, num):
        ret = [int(i) for i in '{0:b}'.format(num)]
        return [0] * (self.input_size - len(ret)) + ret

    def __getitem__(self, idx):
        idx += self.start
        x = self.encoder(idx)
        if idx % 15 == 0:
            y = [1, 0, 0, 0]
        elif idx % 5 == 0:
            y = [0, 1, 0, 0]
        elif idx % 3 == 0:
            y = [0, 0, 1, 0]
        else:
            y = [0, 0, 0, 1]
        return x, y

    def __len__(self):
        """ setting the length to a limit. Theoretically fizbuz dataset can have
        infinitily long dataset but dataloaders fetches len(dataset) to loop decide
        what's the length. Returning any number from this function sets that as the
        length of the dataset
        """
        return self.end - self.start


if __name__ == '__main__':
    dataset = FizBuzDataset()
    for i in range(len(dataset)):
        x, y = dataset[i]

    dataloader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=4)
    for batch in dataloader:
        print(batch)
