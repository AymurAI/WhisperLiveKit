import numpy as np
from typing import Dict, Iterable, Union

def embedding_sign_hash(
    embedding: Union[Iterable[float], np.ndarray], bits: int = 256
) -> str:
    """
    Deterministic binary signature for an embedding.

    Uses sign bits of the (optionally truncated) normalized embedding so
    Hamming distance approximates cosine similarity. Padding ensures a
    byte-aligned hex string.
    """
    vec = np.asarray(embedding, dtype=np.float32).flatten()
    if vec.size == 0:
        raise ValueError("Cannot hash empty embedding")

    norm = np.linalg.norm(vec)
    if norm == 0:
        raise ValueError("Cannot hash zero-norm embedding")

    vec = vec / norm
    usable = min(bits, vec.size) if bits else vec.size
    signs = vec[:usable] >= 0

    pad = (-usable) % 8
    if pad:
        signs = np.pad(signs, (0, pad), mode="constant", constant_values=False)

    packed = np.packbits(signs.astype(np.uint8))
    return packed.tobytes().hex()


def build_speaker_hashes(
    embeddings: Dict[Union[int, str], Iterable[float]], bits: int = 256
) -> Dict[int, str]:
    """Map speaker index -> deterministic hex hash."""
    hashes: Dict[int, str] = {}
    for spk, emb in embeddings.items():
        try:
            hashes[int(spk)] = embedding_sign_hash(emb, bits=bits)
        except Exception:
            continue
    return hashes
