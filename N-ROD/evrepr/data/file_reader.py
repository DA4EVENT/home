import os
from glob import glob
from abc import ABCMeta, abstractmethod


class FileReader(metaclass=ABCMeta):
    ext = None

    def __init__(self, **kwargs):
        pass

    @classmethod
    def glob(cls, root_path):
        paths = glob(
            os.path.join(root_path, "**/*" + cls.ext),
            recursive=True
        )
        return paths

    @abstractmethod
    def read_example(self, filename, start=0, count=-1, *args):
        pass
