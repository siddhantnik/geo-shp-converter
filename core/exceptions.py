"""
core/exceptions.py
Custom exception hierarchy for the GeoJSON → Shapefile converter.
"""

from typing import Optional


class ConverterError(Exception):
    """Base exception for all converter errors."""
    pass


class UnreachableURLError(ConverterError):
    """
    Raised when a URL cannot be reached or returns a non-200 HTTP status.
    Includes the original URL and the HTTP status code when available.
    """

    def __init__(self, message: str, url: str = "", status_code: Optional[int] = None):
        super().__init__(message)
        self.url = url
        self.status_code = status_code


class UnrecognizedShapeError(ConverterError):
    """
    Raised when the parsed JSON doesn't match any of the three known input shapes:
    GEOJSON_DIRECT, METADATA_REDIRECT, or TABULAR_POINTS.
    """
    pass


class LatLonDetectionError(ConverterError):
    """
    Raised when tabular (flat JSON) data is detected but lat/lon fields
    cannot be auto-detected from common field name variants.

    Carries `available_columns` so the UI can present dropdowns for the user
    to manually identify which column is latitude and which is longitude.
    """

    def __init__(
        self,
        message: str,
        available_columns: Optional[list] = None,
        detected_lat: Optional[str] = None,
        detected_lon: Optional[str] = None,
    ):
        super().__init__(message)
        self.available_columns = available_columns or []
        self.detected_lat = detected_lat    # partially-detected lat field (may be None)
        self.detected_lon = detected_lon    # partially-detected lon field (may be None)


class ColumnCollisionWarning(UserWarning):
    """
    Issued (as a warning, not an error) when truncating column names to the
    10-character shapefile limit produces a name collision. The converter
    handles the collision automatically, but the UI should surface this to
    the user so they are aware of the rename.
    """
    pass
