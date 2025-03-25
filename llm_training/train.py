from absl import app, flags
from gemma import transformer
from jaxtyping import Int, Array
import nnjax
import jax
import optax
import dataclasses

import jax
import numpy as np
import jax.numpy as jnp
from jax import sharding as shd
from functools import partial
import tqdm
import wandb

FLAGS = flags.FLAGS
flags.DEFINE_integer('steps', 600_000, 'Number of steps to train for')
flags.DEFINE_integer('batch_size', 128, 'Batch size')

@nnjax.pytree_dataclass
class TrainState:
    model: transformer.Gemma
    tx: optax.GradientTransformation = nnjax.static_field()
    opt_state: optax.OptState

    @property
    def params(self):
        return nnjax.asdict(self.model)

    def update(self, grads):
        updates, new_opt_state = self.tx.update(grads, self.opt_state, params=self.params)
        new_params = optax.apply_updates(self.params, updates)
        return dataclasses.replace(self, model=nnjax.merge(self.model, new_params), opt_state=new_opt_state)


def loss_fn(model: transformer.Gemma, params: dict, tokens: Int[Array, "b l"], input_mask: Int[Array, "b l"]) -> tuple[Array, dict[str, Array]]:
    model = nnjax.merge(model, params)

    def masked_mean(x: Array, mask: Array, axis= None) -> Array:
        mask = jnp.broadcast_to(mask, x.shape)
        return (x * mask).sum(axis=axis) / jnp.clip(mask.sum(axis=axis), a_min=1)

    input_tokens = tokens[:, :-1]
    target_tokens = tokens[:, 1:]
    loss_mask = input_mask[:, 1:]

    logits, _ = model(input_tokens)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    target_probs = jax.nn.one_hot(target_tokens, logits.shape[-1])
    loss = (target_probs * -log_probs).sum(axis=-1)
    loss = masked_mean(loss, loss_mask)
    return loss, {
        "loss": loss,
        "accuracy": masked_mean(logits.argmax(axis=-1) == target_tokens, loss_mask),
    }



def main(_):
    jax.distributed.initialize()
    mesh = jax.make_mesh(axis_shapes=(jax.device_count(),), axis_names=('DP',))
    P = lambda *specs: shd.NamedSharding(mesh, shd.PartitionSpec(*specs))

    lr_schedule = optax.warmup_cosine_decay_schedule(0, 6e-4, min(2_000, FLAGS.steps - 1), FLAGS.steps, 6e-5)
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(lr_schedule, weight_decay=0.1, mask=lambda params: jax.tree_util.tree_map_with_path(lambda path, _: '/w' in jax.tree_util.keystr(path, separator='/'), params)),
    )

    @partial(jax.jit, out_shardings=P())
    def init():
        config = transformer.GemmaConfig.gemma_gpt2_small(dtype=jnp.bfloat16)
        config.vocab_size = 50_257
        model = transformer.Gemma.create(jax.random.key(0), config)
        params = nnjax.asdict(model)
        return TrainState(model, tx, tx.init(params))
    
    @partial(jax.jit, in_shardings=(P(), P('DP')), out_shardings=P())
    def update(state: TrainState, tokens: Int[Array, "b l"]):
        print(f"{tokens.shape=}")
        input_mask = jnp.ones_like(tokens)
        (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True, argnums=1)(state.model, state.params, tokens, input_mask)
        state = state.update(grads)
        return state, info
    
    @partial(jax.jit, in_shardings=(P(), P('DP')), out_shardings=P())
    def metrics(state: TrainState, tokens: Int[Array, "b l"]):
        input_mask = jnp.ones_like(tokens)
        loss, info = loss_fn(state.model, state.params, tokens, input_mask)
        return info
    
    state = init()

    train_data = np.memmap('/home/dibya/train.bin', dtype=np.uint16, mode='r')
    val_data = np.memmap('/home/dibya/val.bin', dtype=np.uint16, mode='r')
    n_processes = jax.process_count()
    _ds = P('DP')
    def sample_from(data, shape):
        idx = np.random.randint(0, len(data) - shape[1], (shape[0],))
        return jax.make_array_from_process_local_data(_ds, np.stack([data[i:i+shape[1]] for i in idx]))

    if jax.process_index() == 0:
        wandb.init(project='gpt2')

    for step in (ranger := tqdm.trange(FLAGS.steps)):
        tokens = sample_from(train_data, (FLAGS.batch_size // n_processes, 1024))
        state, info = update(state, tokens)
        if step % 100 == 0:
            val_tokens = sample_from(val_data, (FLAGS.batch_size // n_processes, 1024))
            val_info = metrics(state, val_tokens)
            info, val_info = jax.device_get((info, val_info))   
            ranger.write(f"{info} {val_info}")
            if jax.process_index() == 0:
                wandb.log({
                    'train': info,
                    'val': val_info
                }, step=step)

if __name__ == "__main__":
    app.run(main)
