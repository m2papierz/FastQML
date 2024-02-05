try:
    from jax import config as j_cfg
except ImportError:
    raise ImportError(
        "JAX is not installed. Please install JAX before using FastQML."
    )

j_cfg.update("jax_enable_x64", True)
