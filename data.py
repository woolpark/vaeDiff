from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

IMAGE_SIZE = 128
BATCH_SIZE = 8

IMAGE_PATH = './CUB_200_2011/images'

class VAEDataLoader:
    def __init__(self):
        self.ds = ImageFolder(
            IMAGE_PATH,
            T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Resize(IMAGE_SIZE),
                T.CenterCrop(IMAGE_SIZE),
                T.ToTensor()
            ])
        )
        self.dl = DataLoader(self.ds, BATCH_SIZE)
