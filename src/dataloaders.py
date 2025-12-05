from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from PIL import Image
from torchvision.transforms import Compose
from torchvision import transforms

class ImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.img_pth = list(self.root.rglob("*.jpg"))
        self.transform = transform
        
    def __len__(self):
        return len(self.img_pth)
    
    def __getitem__(self, index):
        img = Image.open(self.img_pth[index].convert("RGB"))
        if self.transform:
            img = self.transform(img)
        return img
               
transform = Compose([
    transforms.CenterCrop(256),
    transforms.ToTensor(),
])

def get_dataloaders(batch_size:int=16,
                   num_workers:int=4,
                   pin_memory:bool=True,
                   prefetch_factor:int=2,
                   persistent_workers:int=2):
    
    content_dataset = ImageDataset(root="data/content",
                                   transform=transform)
    style_dataset = ImageDataset(root="data/style",
                                 transform=transform)
    
    content_loader = DataLoader(dataset=content_dataset,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                pin_memory=pin_memory,
                                persistent_workers=persistent_workers,
                                prefetch_factor=prefetch_factor,
                                shuffle=True)
    style_loader = DataLoader(dataset=style_dataset,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              pin_memory=pin_memory,
                              persistent_workers=persistent_workers,
                              prefetch_factor=prefetch_factor,
                              shuffle=True)
    
    return content_loader, style_loader