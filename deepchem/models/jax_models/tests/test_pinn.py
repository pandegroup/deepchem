import pytest
import numpy as np
import functools
try:
  import jax
  import jax.numpy as jnp
  import haiku as hk
  import optax
  from deepchem.models import PINN_Model
  from deepchem.data import NumpyDataset
  from jax import jacrev
  has_haiku_and_optax = True
except:
  has_haiku_and_optax = False


@pytest.mark.jax
def test_sine_x():
  """
    Here we are solving the diffrential equation- f`(x) = -sin(x) and f(0) = 1
    We give initial for the neural network at x_init --> np.linspace(-1 * np.pi, 1 * np.pi, 5)
    And we try to approximate the function for domain (-np.pi, np.pi)
    """

  def create_eval_fn(forward_fn, params):
    """
      Calls the function to evaluate the model
      """

    @jax.jit
    def eval_model(x, rng=None):

      bu = forward_fn(params, rng, x)
      return jnp.squeeze(bu)

    return eval_model

  def gradient_fn(forward_fn, loss_outputs, initial_data):
    """
        This function calls the gradient function, to implement the backpropogation
        """
    boundary_data = initial_data['X0']
    boundary_target = initial_data['u0']

    @jax.jit
    def model_loss(params, target, weights, rng, x_train):

      @functools.partial(jax.vmap, in_axes=(None, 0))
      def periodic_loss(params, x):
        x = jnp.expand_dims(x, 0)
        u_x = jacrev(forward_fn, argnums=(2))(params, rng, x)
        return u_x + jnp.sin(x)

      u_pred = forward_fn(params, rng, boundary_data)
      loss_u = jnp.mean((u_pred - boundary_target)**2)

      f_pred = periodic_loss(params, x_train)
      loss_f = jnp.mean((f_pred**2))

      return loss_u + loss_f

    return model_loss

  def f(x):
    net = hk.nets.MLP(output_sizes=[256, 128, 1], activation=jax.nn.softplus)
    val = net(x)
    return val

  init_params, forward_fn = hk.transform(f)
  rng = jax.random.PRNGKey(500)
  params = init_params(rng, np.random.rand(1000, 1))

  opt = optax.chain(
      optax.clip_by_global_norm(1.00), optax.adam(1e-2, b1=0.9, b2=0.99))

  in_array = np.linspace(-1 * np.pi, 1 * np.pi, 5)
  out_array = np.cos(in_array)
  boundary_data = {
      'X0': jnp.expand_dims(in_array, 1),
      'u0': jnp.expand_dims(out_array, 1)
  }

  j_m = PINN_Model(
      forward_fn=forward_fn,
      params=params,
      boundary_data=boundary_data,
      batch_size=1000,
      optimizer=opt,
      grad_fn=gradient_fn,
      eval_fn=create_eval_fn,
      deterministic=True,
      log_frequency=1000)

  X_f = np.expand_dims(np.linspace(-1 * np.pi, 1 * np.pi, 100), 1)
  dataset = NumpyDataset(X_f)
  _ = j_m.fit(dataset, nb_epochs=1000)

  test = np.expand_dims(np.linspace(-1 * np.pi, 1 * np.pi, 1000), 1)
  dataset_test = NumpyDataset(test)
  ans = j_m.predict(dataset_test)
  out_array = np.cos(test).squeeze()

  assert np.allclose(out_array, ans, atol=1e-01)
