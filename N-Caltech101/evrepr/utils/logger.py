import logging
import coloredlogs


def setup_logger(name):

    logger = logging.getLogger(name)
    coloredlogs.install(
        level='DEBUG', logger=logger,
        fmt='%(asctime)s %(name)s %(levelname)s %(message)s')

    return logger
