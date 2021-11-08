from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader

class CIFAR10IndexPseudoLabelEnsemble(Dataset):
    def __init__(self, root='', transform=None, download=False, train=True,
                 pseudoLabel_002=None,
                 pseudoLabel_010=None,
                 pseudoLabel_050=None,
                 pseudoLabel_100=None,
                 pseudoLabel_500=None,
                 ):
        self.cifar10 = datasets.CIFAR10(root=root,
                                        download=download,
                                        train=train,
                                        transform=transform)

        self.pseudo_label_002 = pseudoLabel_002
        self.pseudo_label_010 = pseudoLabel_010
        self.pseudo_label_050 = pseudoLabel_050
        self.pseudo_label_100 = pseudoLabel_100
        self.pseudo_label_500 = pseudoLabel_500

    def __getitem__(self, index):
        data, target = self.cifar10[index]
        label_p_002 = self.pseudo_label_002[index]
        label_p_010 = self.pseudo_label_010[index]
        label_p_050 = self.pseudo_label_050[index]
        label_p_100 = self.pseudo_label_100[index]
        label_p_500 = self.pseudo_label_500[index]

        label_p = (label_p_002,
                   label_p_010,
                   label_p_050,
                   label_p_100,
                   label_p_500)
        return data, target, label_p, index

    def __len__(self):
        return len(self.cifar10)


