from utils.class_registry import ClassRegistry
from torch.optim import Adam


optimizers_registry = ClassRegistry()


@optimizers_registry.add_to_registry(name="adam")
class Adam_(Adam):
    pass