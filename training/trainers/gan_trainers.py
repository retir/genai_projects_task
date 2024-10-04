from utils.class_registry import ClassRegistry
from utils.model_utils import requires_grad
from training.trainers.base_trainer import BaseTrainer

from models.gan_models import gens_registry, discs_registry
from training.optimizers import optimizers_registry
from training.losses.gan_losses import GANLossBuilder


gan_trainers_registry = ClassRegistry()


@gan_trainers_registry.add_to_registry(name="base_gan_trainer")
class BaseGANTrainer(BaseTrainer):
    def setup_models(self):
        # TO DO
        # self.generator = ...
        # self.dicriminator = ...
        # do not forget to load state from checkpoints if provided
        raise NotImplementedError()


    def setup_optimizers(self):
        # TO DO
        # self.generator_optimizer = ...
        # self.dicriminator_optimizer = ...
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
        # synthesize images via generator
        # calculate disc losses, make discriminator step
        # calculate gen losses, make generator step
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
        # return batch_of_images,path_to_saved_pics, 
        raise NotImplementedError()