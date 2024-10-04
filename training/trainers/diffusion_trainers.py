from utils.class_registry import ClassRegistry
from training.trainers.base_trainer import BaseTrainer

from models.diffusion_models import diffusion_models_registry
from training.optimizers import optimizers_registry
from training.losses.diffusion_losses import DiffusionLossBuilder



diffusion_trainers_registry = ClassRegistry()


@diffusion_trainers_registry.add_to_registry(name="base_diffusion_trainer")
class BaseDiffusionTrainer(BaseTrainer):
    def setup_models(self):
        # TO DO
        # self.unet = ...
        # self.encoder == ... # if needed
        # self.noise_scheduler = ...
        # do not forget to load state from checkpoints if provided
        raise NotImplementedError()


    def setup_optimizers(self):
        # TO DO
        # self.optimizer = ...
        # do not forget to load state from checkpoints if provided
        raise NotImplementedError()


    def setup_losses(self):
        # TO DO
        # self.loss_builder = ...
        raise NotImplementedError()


    def to_train(self):
        # TO DO
        # all trainable modules to .train()
        raise NotImplementedError()


    def to_eval(self):
        # TO DO
        # all trainable modules to .eval()
        raise NotImplementedError()


    def train_step(self):
        # TO DO
        # batch = next(self.train_dataloader)
        # timesteps = ...
        # add noise to images according self.noise_scheduler
        # predict noise via self.unet
        # calculate losses, make oprimizer step
        # return dict of losses to log
        raise NotImplementedError()


    def save_checkpoint(self):
        # TO DO
        # save all necessary parts of your pipeline
        raise NotImplementedError()


    def synthesize_images(self):
        # TO DO
        # synthesize images and save to self.experiment_dir/images
        # synthesized additional batch of images to log
        # return batch_of_images, path_to_saved_pics, 
        raise NotImplementedError()