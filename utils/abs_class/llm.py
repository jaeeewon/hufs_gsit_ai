from abc import ABC, abstractmethod


class AbsCallLLM(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def call(self, prompt: str) -> str:
        raise NotImplementedError()
