"""AdaptiveRL Studio desktop application.

The GUI is optional and deliberately depends on the public AdaptiveRL APIs rather
than implementing a second training or evaluation stack.
"""

__all__ = ["launch_studio"]


def launch_studio(*args: object, **kwargs: object) -> int:
    """Launch Studio lazily so importing :mod:`adaptive_rl` stays lightweight."""
    from adaptive_rl.studio.app import launch_studio as _launch_studio

    return _launch_studio(*args, **kwargs)
