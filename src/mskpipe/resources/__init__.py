# SPDX-License-Identifier: Apache-2.0
"""Data the pipeline needs besides the code: per-user folder, model weights,
TotalSegmentator licence key and the attachment atlas (``mskpipe setup ...``).

Nothing here imports torch or the segmentation tools; work that needs them (downloads)
runs in a child process.
"""
