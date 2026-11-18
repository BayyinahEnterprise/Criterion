"""Single source of truth for the Criterion version.

Every artifact that states a version (records, SARIF, capture metadata)
reads it from here, so a record can never claim a version other than the
code that wrote it.
"""

__version__ = "0.5.0"
