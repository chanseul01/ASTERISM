"""Assemble the ASTERISM-Train items that are released as identifiers.

Sources with non-commercial terms or without a stated license are released as item identifiers. For items of
public datasets and instruction collections, the `build` field of a record writes the training text as our own
text and slices of the source record; for synthetic items, the record holds our text and the `images` entries
name the source images and the operations applied to them. `python -m build.train.assemble` reads the source
records and images from the public releases and writes the full items.
"""
