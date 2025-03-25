from absl import app, flags
from gemma import dumb_implementation
from jaxtyping import Int, Array, Float
import nnjax
import jax
import optax
import dataclasses
import einops
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
    model: dumb_implementation.NewGemma
    tx: optax.GradientTransformation = nnjax.static_field()
    opt_state: optax.OptState
    step: int

    @property
    def params(self):
        return nnjax.asdict(self.model)

    def update(self, grads):
        updates, new_opt_state = self.tx.update(grads, self.opt_state, params=self.params)
        new_params = optax.apply_updates(self.params, updates)
        return dataclasses.replace(self, model=nnjax.merge(self.model, new_params), opt_state=new_opt_state, step=self.step + 1)


def loss_fn(model: dumb_implementation.NewGemma, params: dict, tokens: Int[Array, "b 2 l"]) -> tuple[Array, dict[str, Array]]:
    model = nnjax.merge(model, params)
    input_tokens = tokens[..., :-1]
    target_tokens = tokens[..., 1:]
    logits, _ = model(input_tokens)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    target_probs = jax.nn.one_hot(target_tokens, logits.shape[-1])
    loss = (target_probs * -log_probs).sum(axis=-1).mean()
    return loss, {
        "loss": loss,
        "accuracy": (logits.argmax(axis=-1) == target_probs.argmax(axis=-1)).mean(),
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
        config = dumb_implementation.GemmaConfig.gemma_gpt2_small(dtype=jnp.bfloat16)
        config.vocab_size = 50_257
        model = dumb_implementation.NewGemma.create(jax.random.key(0), config)
        params = nnjax.asdict(model)
        return TrainState(model, tx, tx.init(params), 0)
    
    @partial(jax.jit, in_shardings=(P(), P('DP')), out_shardings=P())
    def update(state: TrainState, tokens: Int[Array, "b l"]):
        print(f"{tokens.shape=}")
        tokens = einops.rearrange(tokens, "(b k) l -> b k l", k=2)
        (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True, argnums=1)(state.model, state.params, tokens)
        state = state.update(grads)
        return state, info
    
    @partial(jax.jit, in_shardings=(P(), P('DP')), out_shardings=P())
    def metrics(state: TrainState, tokens: Int[Array, "b l"]):
        tokens = einops.repeat(tokens, "b l -> b k l", k=2)
        loss, info = loss_fn(state.model, state.params, tokens)
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
        wandb.run.log_code(str(nnjax.repo_path()))

    for step in (ranger := tqdm.trange(FLAGS.steps)):
        tokens = sample_from(train_data, (2 * FLAGS.batch_size // n_processes, 1024))
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
