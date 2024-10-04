from PIL import Image
from torch.utils.data import Dataset
from utils.data_utils import make_dataset
from utils.class_registry import ClassRegistry


datasets_registry = ClassRegistry()


@datasets_registry.add_to_registry(name="base_dataset")
class BaseDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.paths = make_dataset(root)
        self.transforms = transforms

    def __getitem__(self, ind):
        path = self.paths[ind]
        image = Image.open(path).convert("RGB")

        if self.transforms:
            image = self.transforms(image)

        return {"images": image}

    def __len__(self):
        return len(self.paths)

