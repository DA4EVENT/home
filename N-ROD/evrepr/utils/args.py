import os
from omegaconf import OmegaConf

from evrepr.utils.logger import setup_logger
logger = setup_logger(__name__)


def parse_args():
    # Retrieve the configs path
    conf_path = os.path.join(os.path.dirname(__file__), '../configs')

    # Retrieve the default config
    args = OmegaConf.load(os.path.join(conf_path, "default.yaml"))

    # Read the cli args
    cli_args = OmegaConf.from_cli()

    # Optional configs
    if cli_args.data.config:
        data_conf = cli_args.data.config
        if not data_conf.endswith(".yaml") and not data_conf.endswith(".yml"):
            logger.info("Read from evrepr's {}.yaml".format(data_conf))
            data_conf = os.path.join(conf_path, "data", data_conf + ".yaml")

        data_args = OmegaConf.load(data_conf)
        args = OmegaConf.merge(args, data_args)

    if cli_args.repr.config:
        repr_conf = cli_args.repr.config
        if not repr_conf.endswith(".yaml") and not repr_conf.endswith(".yml"):
            logger.info("Read from evrepr's {}.yaml".format(repr_conf))
            repr_conf = os.path.join(conf_path, "repr", repr_conf + ".yaml")

        repr_args = OmegaConf.load(repr_conf)
        args = OmegaConf.merge(args, repr_args)

    if cli_args.config:
        conf_args = OmegaConf.load(cli_args.config)
        args = OmegaConf.merge(args, conf_args)

    # Merge cli args into config ones
    args = OmegaConf.merge(args, cli_args)

    return args
