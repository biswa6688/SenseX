"""3D-Speaker (SDPN/ECAPA-TDNN, VoxCeleb-trained English speaker
verification) embedding adapter — see DECISIONS.md #17.

Lazily loaded only when boundary_refinement.py actually needs a second
opinion (community-1's own embedding declined to split an uncertain
turn) — not resident by default, so the common case's RAM footprint is
unchanged. Verified on real audio: found a genuine speaker change
(t=104.85s, similarity 0.297 vs a 0.6-0.97 baseline) inside a turn
community-1's own embedding couldn't split — both halves matched
DIFFERENT known speakers (0.71 vs 0.65, and 0.74 vs 0.63) — exactly the
independent signal reusing community-1's own embedding space can't
provide (see DECISIONS.md #15/#16 for why that reuse alone wasn't
enough on this same test case).

Measured cost: ~560MB resident once loaded (checkpoint is ~343MB on
disk). Not loaded until refine_diarization_turns() actually needs it.
"""

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)

MODEL_ID = "iic/speech_sdpn_ecapa_tdnn_sv_en_voxceleb_16k"

_pipeline_cache = None
# A load failure (missing deps, no network on first download, etc.) is
# NOT cached as permanent — mirrors diarize.get_diarization_pipeline()'s
# same policy, so a transient failure doesn't require a process restart.


def _load_with_weights_only_false(load_fn):
    """modelscope's SDPN checkpoint loader calls torch.load() without
    weights_only=False; PyTorch 2.6+ defaults that to True, which breaks
    on this checkpoint's pickled numpy scalar globals (verified: even
    torch.serialization.add_safe_globals([numpy.core.multiarray.scalar])
    doesn't fix it — the pickle encodes the pre-numpy-2.x module path
    string, which doesn't match how torch resolves the allowlisted
    object's __module__). Scoped monkeypatch, restored immediately after
    — this checkpoint is from an official ModelScope org account (iic /
    Alibaba DAMO), a reasonable trust level for disabling weights_only
    specifically for this one load."""
    original_load = torch.load

    def patched_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = patched_load
    try:
        return load_fn()
    finally:
        torch.load = original_load


def _get_sv_pipeline():
    global _pipeline_cache
    if _pipeline_cache is not None:
        return _pipeline_cache

    try:
        from modelscope.pipelines import pipeline

        _pipeline_cache = _load_with_weights_only_false(
            lambda: pipeline(task="speaker-verification", model=MODEL_ID)
        )
    except Exception:
        logger.exception("Failed to load 3D-Speaker embedding model")
        return None
    return _pipeline_cache


class ThreeDSpeakerEmbedding:
    """Adapts the SDPN pipeline to the same call signature
    boundary_refinement.py already uses for pyannote's own embedding
    model: __call__(waveforms: torch.Tensor shaped (batch, 1, samples))
    -> np.ndarray shaped (batch, dim), plus a `sample_rate` attribute —
    lets boundary_refinement.py use either embedding source unmodified."""

    sample_rate = 16000

    def __init__(self, sv_pipeline):
        self._model = sv_pipeline.model

    def __call__(self, waveforms: torch.Tensor) -> np.ndarray:
        results = []
        with torch.inference_mode():
            for i in range(waveforms.shape[0]):
                audio = waveforms[i]  # (1, samples) — this model's expected input shape
                embedding = self._model(audio)
                results.append(embedding.squeeze(0).cpu().numpy())
        return np.stack(results)


def get_embedding_model():
    sv_pipeline = _get_sv_pipeline()
    if sv_pipeline is None:
        return None
    return ThreeDSpeakerEmbedding(sv_pipeline)
