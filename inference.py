from omegaconf import OmegaConf
from utils.model_utils import setup_seed


def load_config():
    conf_cli = OmegaConf.from_cli()

    config_path = conf_cli.exp.config_path
    conf_file = OmegaConf.load(config_path)

    config = OmegaConf.merge(conf_file, conf_cli)
    return config


if __name__ == "__main__":
    config = load_config()
    setup_seed(config.exp.seed)

    if config.exp.model_type == "gan":
        from training.trainers.gan_trainers import gan_trainers_registry
        trainer = gan_trainers_registry[config.train.trainer](config)
    elif config.exp.model_type == "diffusion":
        from training.trainers.diffusion_trainers import diffusion_trainers_registry
        trainer = diffusion_trainers_registry[config.train.trainer](config)
    else:
        raise ValueError(f"Unknown model type {config.exp.model_type}")

    trainer.setup_inference()
    trainer.inference()