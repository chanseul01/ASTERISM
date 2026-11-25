"""Assemble the ASTERISM items of the sources that are released as identifiers.

MileBench, VLM2-Bench, M4Bench, MMLongBench-Doc, MaRVL, Q-Bench2 and EMMA carry non-commercial terms or
no stated license, so ASTERISM releases their identifiers and our own annotations.
`python -m build.assemble` reads each item from the public release of its source, applies the
changes recorded in the ASTERISM record and writes the full item with its images.
"""
