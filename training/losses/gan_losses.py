import torch
from torch import nn

from torch.nn import functional as F
from utils.class_registry import ClassRegistry


gen_losses_registry = ClassRegistry()
disc_losses_registry = ClassRegistry()


class GANLossBuilder:
    def __init__(self, config):
        self.gen_losses = {}
        self.disc_losses = {}
        self.coefs = {}

        for loss_name, loss_coef in config.gen_losses.items():
            self.coefs[loss_name] = loss_coef
            loss_args = {}
            if losses_args in config and loss_name in config.losses_args:
                loss_args = config.losses_args
            self.gen_losses[loss_name] = gen_losses_registry[loss_name](**loss_args)

        for loss_name, loss_coef in config.disc_losses.items():
            self.coefs[loss_name] = loss_coef
            loss_args = {}
            if losses_args in config and loss_name in config.losses_args:
                loss_args = config.losses_args
            self.disc_losses[loss_name] = disc_losses_registry[loss_name](**loss_args)

    def calculate_loss(batch_data, loss_type):
        # batch_data is a dict with all necessary data for loss calculation
        loss_dict = {}
        total_loss = 0.0

        if loss_type == "gen":
            losses = self.gen_losses
        elif loss_type == "disc":
            losses = self.disc_losses

        for loss_name, loss in losses.items():
            loss_val = loss(batch_data)
            total_loss += self.coefs[loss_name] * loss_val
            loss_dict[loss_name] = float(loss_val)

        return total_loss, loss_dict


@gen_losses_registry.add_to_registry(name="softplus_gen")
class SoftPlusGenLoss(nn.Module):
    def forward(self, batch):
        return F.softplus(-batch["fake_preds"]).mean()


@disc_losses_registry.add_to_registry(name="softplus_disc")
class SoftPlusGenLoss(nn.Module):
    def forward(self, batch):
        # TO DO
        # calculate softplus loss for discriminator
        raise NotImplementedError()

# Add the other losses you need


