from abc import ABC, abstractmethod


class Action(ABC):
    @property
    @abstractmethod
    def START_CHAR(self):
        pass

    @property
    @abstractmethod
    def END_CHAR(self):
        pass
