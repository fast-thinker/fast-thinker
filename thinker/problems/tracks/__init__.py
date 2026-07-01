"""Built-in problem tracks."""


def register_default_tracks() -> tuple[str, ...]:
    """Import and register the default deterministic tracks.

    Keeping these imports inside a function lets downstream tools
    import one track module without also importing every dataset dependency used
    by the others.
    """
    from thinker.problems.tracks import constructive, depth_control, olympiad, procedural

    _ = (constructive, depth_control, olympiad, procedural)
    return ("constructive", "depth_control", "olympiad", "procedural")

__all__ = [
    "register_default_tracks",
]
