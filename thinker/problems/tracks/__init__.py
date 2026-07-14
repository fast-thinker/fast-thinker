"""Built-in problem tracks."""


def register_default_tracks() -> tuple[str, ...]:
    """Import and register the default deterministic tracks.

    Keeping these imports inside a function lets downstream tools import one
    track module without also importing the others.
    """
    from thinker.problems.tracks import constructive, depth_control, olympiad

    _ = (constructive, depth_control, olympiad)
    return ("constructive", "depth_control", "olympiad")

__all__ = [
    "register_default_tracks",
]
