"""Shared geometry used by data export and model inputs."""


def dimensions(size: tuple[int, int], side: int) -> tuple[int, int]:
    scale = side / max(size)
    return tuple(max(16, round(value * scale / 16) * 16) for value in size)
