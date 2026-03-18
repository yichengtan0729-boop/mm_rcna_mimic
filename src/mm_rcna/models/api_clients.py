from __future__ import annotations


class NullAPIClient:
    def __init__(self, name: str = 'null-client') -> None:
        self.name = name

    def generate(self, prompt: str) -> str:
        return prompt[:256]
