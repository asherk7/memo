"""Train-time, per-modality augmentation.

Pure functions / lightweight transforms that slot directly into dataset
pipelines. Submodules are imported directly to avoid pulling every modality's
dependencies at once.
"""
