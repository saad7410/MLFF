import jax
import jax.numpy as jnp

print(f"JAX version: {jax.__version__}")
print(f"Available devices: {jax.devices()}")
print(f"GPU devices: {jax.devices('gpu')}")
print(f"Device count: {jax.device_count()}")

if jax.devices('gpu'):
    # Test computation on GPU
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (1000, 1000))
    y = jax.random.normal(key, (1000, 1000))
    z = jnp.dot(x, y)
    print(f"GPU computation test passed! Result shape: {z.shape}")
else:
    print("❌ GPU not available for JAX")