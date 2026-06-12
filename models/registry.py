from typing import Callable, Type, Any, Dict

from utils.logger import get_logger

logger = get_logger(__name__)

MODEL_REGISTRY: Dict[str, Type] = {}


def register_model(name: str) -> Callable[[Type], Type]:
    """
    Decorator to register a model class in the MODEL_REGISTRY.

    Args:
        name: Unique identifier for the model. Must be non-empty string.

    Returns:
        Decorator function that registers the class.

    Raises:
        TypeError: If name is not a string.
        ValueError: If name is empty or already registered.
    """
    if not isinstance(name, str):
        raise TypeError(f"Model name must be a string, got {type(name).__name__}")

    name = name.strip()
    if not name:
        raise ValueError("Model name cannot be empty or whitespace-only")

    if name in MODEL_REGISTRY:
        existing = MODEL_REGISTRY[name].__name__
        logger.warning(
            f"Overwriting existing model registration: '{name}' "
            f"(previously: {existing}, new: pending)"
        )

    def decorator(cls: Type) -> Type:
        if not isinstance(cls, type):
            raise TypeError(
                f"@register_model('{name}') must be applied to a class, "
                f"got {type(cls).__name__}"
            )

        if name in MODEL_REGISTRY and MODEL_REGISTRY[name] is not cls:
            old_cls = MODEL_REGISTRY[name].__name__
            logger.warning(f"Replacing model '{name}': {old_cls} -> {cls.__name__}")

        MODEL_REGISTRY[name] = cls
        logger.info(f"Registered model '{name}' -> {cls.__module__}.{cls.__name__}")

        cls._model_name = name
        cls._model_registered = True

        return cls

    return decorator


def get_model(name: str, **kwargs: Any) -> Any:
    """
    Instantiate a registered model by name.

    Args:
        name: Registered model identifier.
        **kwargs: Arguments passed to model constructor.

    Returns:
        Instance of the requested model.

    Raises:
        TypeError: If name is not a string.
        ValueError: If model not found or instantiation fails.
        RuntimeError: If model class is invalid.
    """
    if not isinstance(name, str):
        raise TypeError(f"Model name must be a string, got {type(name).__name__}")

    name = name.strip()
    if not name:
        raise ValueError("Model name cannot be empty")

    logger.debug(
        f"Requesting model instantiation: '{name}' with kwargs: {list(kwargs.keys())}"
    )

    if name not in MODEL_REGISTRY:
        available = sorted(MODEL_REGISTRY.keys())
        available_str = ", ".join(f"'{k}'" for k in available) if available else "none"
        logger.error(f"Model '{name}' not found. Available: {available_str}")
        raise ValueError(
            f"Model '{name}' not found in registry. "
            f"Available models: {available_str}"
        )

    model_cls = MODEL_REGISTRY[name]

    if not isinstance(model_cls, type):
        logger.critical(
            f"Registry corruption detected: '{name}' maps to non-class "
            f"object {type(model_cls).__name__}"
        )
        raise RuntimeError(f"Registry entry for '{name}' is corrupted")

    if not getattr(model_cls, "_model_registered", False):
        logger.warning(
            f"Model '{name}' ({model_cls.__name__}) lacks registration marker. "
            f"Possible manual registry manipulation."
        )

    try:
        instance = model_cls(**kwargs)
    except TypeError as e:
        logger.error(
            f"Failed to instantiate model '{name}' ({model_cls.__name__}): {e}"
        )
        raise ValueError(
            f"Failed to instantiate model '{name}': {e}. "
            f"Check that constructor arguments match provided kwargs: {list(kwargs.keys())}"
        ) from e
    except Exception as e:
        logger.exception(f"Unexpected error instantiating model '{name}'")
        raise RuntimeError(
            f"Unexpected error creating model '{name}': {type(e).__name__}: {e}"
        ) from e

    logger.info(f"Successfully instantiated model '{name}' ({model_cls.__name__})")
    return instance


def list_models() -> Dict[str, str]:
    """Return mapping of registered model names to their class names."""
    return {
        name: cls.__name__
        for name, cls in MODEL_REGISTRY.items()
        if isinstance(cls, type)
    }


def unregister_model(name: str) -> bool:
    """
    Remove a model from registry. Returns True if removed, False if not found.
    """
    if name in MODEL_REGISTRY:
        cls_name = MODEL_REGISTRY[name].__name__
        del MODEL_REGISTRY[name]
        logger.info(f"Unregistered model '{name}' ({cls_name})")
        return True
    logger.debug(f"Attempted to unregister non-existent model '{name}'")
    return False


def clear_registry() -> None:
    """Clear all registered models. Use with caution."""
    count = len(MODEL_REGISTRY)
    MODEL_REGISTRY.clear()
    logger.warning(f"Cleared model registry ({count} models removed)")
