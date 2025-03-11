import dataclasses
import utils
from flax import nnx


@dataclasses.dataclass
class TransformerConfig:
    layers: int
    vocab_size: int


@dataclasses.dataclass
class Transformer(utils.SaveableModule):
    def __init__(self, config: TransformerConfig, rngs):
        super().__init__(config)  # Save config
        self.layers = [
            nnx.Linear(config.vocab_size, config.vocab_size, rngs=rngs)
            for _ in range(config.layers)
        ]
