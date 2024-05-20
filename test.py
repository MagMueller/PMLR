import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="conf", config_name="config", version_base="1.5")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # print config model nclass
    print(cfg.model.nclass)
    print(cfg.epochs)

    # name of env
    print(cfg.env.name)


if __name__ == "__main__":
    main()
