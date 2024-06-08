import torch.utils
import torch.utils.data
from torchvision import transforms, datasets
import torch

data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
    "val": transforms.Compose([transforms.Resize(256),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}


class Flowers():
    def __init__(self, data_dir, batch_size, num_workers, aug=False):
        self.dataset = datasets.ImageFolder(root=data_dir, transform=data_transform["train" if aug else "val"])
        self.loader = torch.utils.data.DataLoader(dataset=self.dataset,
                                                  batch_size=batch_size,
                                                  shuffle=aug,
                                                  num_workers=num_workers)