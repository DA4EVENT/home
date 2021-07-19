import os
from urllib import request

from evrepr.utils.logger import setup_logger
logger = setup_logger(__name__)


def open_url(url, mode="rb", cache_dir=".cache"):

    os.makedirs(cache_dir, exist_ok=True)
    filename = url.split("/")[-1]
    assert len(filename), "Cannot obtain filename from url {}".format(url)

    cached = os.path.join(cache_dir, filename)

    if os.path.isfile(cached):
        logger.info("File {} already exists! Using cached file".format(filename))
        return open(cached, mode)

    logger.info("Downloading {}".format(url))

    request.urlretrieve(url, cached)
    return open(cached, mode)
