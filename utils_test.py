import transformer
import utils
from flax import nnx

rngs = nnx.Rngs(0)
model = transformer.Transformer(
    transformer.TransformerConfig(layers=12, vocab_size=10), rngs
)

utils.save(model, "test_save")

model2 = utils.load("test_save")

print(model2)
