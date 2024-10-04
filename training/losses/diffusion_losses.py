from torch import nn
from utils.class_registry import ClassRegistry


losses_registry = ClassRegistry()


class DiffusionLossBuilder:
    def __init__(self, config):
        self.losses = {}
        self.coefs = {}

        for loss_name, loss_coef in config.losses.items():
            self.coefs[loss_name] = loss_coef
            loss_args = {}
            if losses_args in config and loss_name in config.losses_args:
                loss_args = config.losses_args
            self.losses[loss_name] = losses_registry[loss_name](**loss_args)


    def calculate_loss(batch_data):
        loss_dict = {}
        total_loss = 0.0

        for loss_name, loss in self.losses.items():
            loss_val = loss(batch_data)
            total_gen_loss += self.coefs[loss_name] * loss_val
            loss_dict[loss_name] = float(loss_val)

        return total_gen_loss, loss_dict


@losses_registry.add_to_registry(name="mse")
class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.MSELoss()

    def forward(self, batch):
        return self.loss_fn(batch["real_noise"], batch["predicted_noise"]).mean()


# Add the other losses you need


