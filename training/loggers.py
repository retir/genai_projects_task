import wandb
import torch
from PIL import Image
from collections import defaultdict

class WandbLogger:
    def __init__(self, config):
        wandb.login(key=os.environ['WANDB_KEY'].strip())
        if config.train.checkpoint_path != "":
            # TO DO
            # resume training run from checkpoint
            raise NotImplementedError()
        else:
            # TO DO
            # create new wandb run and save args, config and etc.
            # self.wandb_args = {
            #     "id": wandb.util.generate_id(),
            #     "project": ...,
            #     "name": ...,
            #     "config": ...,
            # }
            raise NotImplementedError()

        wandb.init(**self.wandb_args, resume="allow")


    @staticmethod
    def log_values(values_dict: dict, step: int):
        # TO DO 
        # log values to wandb
        raise NotImplementedError()

    @staticmethod
    def log_images(images: dict, step: int):
        # TO DO
        # log images
        raise NotImplementedError()


class TrainingLogger:
    def __init__(self, config):
        self.logger = WandbLogger(config)
        self.losses_memory = defaultdict(list)


    def log_train_losses(self, step: int):
        # avarage losses in losses_memory
        # log them and clear losses_memory
        raise NotImplementedError()


    def log_val_metrics(self, val_metrics: dict, step: int):
        # TO DO
        raise NotImplementedError()


    def log_batch_of_images(self, batch: torch.Tensor, step: int, images_type: str = ""):
        # TO DO
        raise NotImplementedError()


    def update_losses(self, losses_dict):
        # it is useful to average losses over a number of steps rather than track them at each step
        # this makes training curves smoother
        for loss_name, loss_val in losses_dict.items():
            self.losses_memory[loss_name].append(loss_val)





