"""Train-time, per-modality augmentation (§4.1).

Pure functions / lightweight transforms, never stateful modules — they slot
directly into dataset pipelines. Submodules are imported directly to avoid
pulling every modality's dependencies at once.
"""
