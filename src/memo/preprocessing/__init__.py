"""Deterministic, stateless preprocessing for face, text, and audio.

Submodules are imported directly (`from memo.preprocessing.audio import ...`)
so a caller that only needs one modality never pays the import cost of the
others (MediaPipe, transformers, librosa are each heavy).
"""
