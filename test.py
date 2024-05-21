import uuid
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import omegaconf
import random


@hydra.main(config_path="conf", config_name="config", version_base="1.5")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    print(cfg.model.n_hid)
    print(cfg.epochs)
    # log config
    unique_id = uuid.uuid4()
    name = cfg.model.name + "_hid:" + str(cfg.model.n_hid) + "_epoch:" + str(cfg.epochs) + "_" + str(unique_id)

    config = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    wandb.init(project=cfg.wandb.project, settings=wandb.Settings(start_method="thread"), name=name, config=config)

    # # wandb.config is unitialized
    # wandb.config = OmegaConf.to_container(
    #     cfg, resolve=True, throw_on_missing=True
    # )
    # wandb.init(project=cfg.wandb.project, settings=wandb.Settings(start_method="thread"), name=name)
    # cfg = OmegaConf.merge(cfg, OmegaConf.create(dict(wandb.config)))
    # wandb.config = dict(cfg)
    n = random.random()
    wandb.log({"test": n})


if __name__ == "__main__":
    main()
