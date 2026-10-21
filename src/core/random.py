"""Sample identity determines the seed independently of iteration order."""

import hashlib


def sample_seed(seed: int, sample_id: str) -> int:
    digest = hashlib.sha256(sample_id.encode()).digest()
    return (seed + int.from_bytes(digest[:4], "little")) % (2**32)
