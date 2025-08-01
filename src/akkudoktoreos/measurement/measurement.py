"""Measurement module to provide and store measurements.

This module provides a `Measurement` class to manage and update a sequence of
data records for measurements.

The measurements can be added programmatically or imported from a file or JSON string.
"""

from typing import Any, ClassVar, List, Optional

import numpy as np
from loguru import logger
from numpydantic import NDArray, Shape
from pendulum import DateTime, Duration
from pydantic import Field, computed_field

from akkudoktoreos.config.configabc import SettingsBaseModel
from akkudoktoreos.core.coreabc import SingletonMixin
from akkudoktoreos.core.dataabc import DataImportMixin, DataRecord, DataSequence
from akkudoktoreos.utils.datetimeutil import to_duration


class MeasurementCommonSettings(SettingsBaseModel):
    """Measurement Configuration."""

    load0_name: Optional[str] = Field(
        default=None, description="Name of the load0 source", examples=["Household", "Heat Pump"]
    )
    load1_name: Optional[str] = Field(
        default=None, description="Name of the load1 source", examples=[None]
    )
    load2_name: Optional[str] = Field(
        default=None, description="Name of the load2 source", examples=[None]
    )
    load3_name: Optional[str] = Field(
        default=None, description="Name of the load3 source", examples=[None]
    )
    load4_name: Optional[str] = Field(
        default=None, description="Name of the load4 source", examples=[None]
    )


class MeasurementDataRecord(DataRecord):
    """Represents a measurement data record containing various measurements at a specific datetime.

    Attributes:
        date_time (Optional[DateTime]): The datetime of the record.
    """

    # Single loads, to be aggregated to total load
    load0_mr: Optional[float] = Field(
        default=None, ge=0, description="Load0 meter reading [kWh]", examples=[40421]
    )
    load1_mr: Optional[float] = Field(
        default=None, ge=0, description="Load1 meter reading [kWh]", examples=[None]
    )
    load2_mr: Optional[float] = Field(
        default=None, ge=0, description="Load2 meter reading [kWh]", examples=[None]
    )
    load3_mr: Optional[float] = Field(
        default=None, ge=0, description="Load3 meter reading [kWh]", examples=[None]
    )
    load4_mr: Optional[float] = Field(
        default=None, ge=0, description="Load4 meter reading [kWh]", examples=[None]
    )

    max_loads: ClassVar[int] = 5  # Maximum number of loads that can be set

    grid_export_mr: Optional[float] = Field(
        default=None, ge=0, description="Export to grid meter reading [kWh]", examples=[1000]
    )

    grid_import_mr: Optional[float] = Field(
        default=None, ge=0, description="Import from grid meter reading [kWh]", examples=[1000]
    )

    # Computed fields
    @computed_field  # type: ignore[prop-decorator]
    @property
    def loads(self) -> List[str]:
        """Compute a list of active loads."""
        active_loads = []

        # Loop through loadx
        for i in range(self.max_loads):
            load_attr = f"load{i}_mr"

            # Check if either attribute is set and add to active loads
            if getattr(self, load_attr, None):
                active_loads.append(load_attr)

        return active_loads


class Measurement(SingletonMixin, DataImportMixin, DataSequence):
    """Singleton class that holds measurement data records.

    Measurements can be provided programmatically or read from JSON string or file.
    """

    records: List[MeasurementDataRecord] = Field(
        default_factory=list, description="List of measurement data records"
    )

    topics: ClassVar[List[str]] = [
        "load",
    ]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if hasattr(self, "_initialized"):
            return
        super().__init__(*args, **kwargs)

    def _interval_count(
        self, start_datetime: DateTime, end_datetime: DateTime, interval: Duration
    ) -> int:
        """Calculate number of intervals between two datetimes.

        Args:
            start_datetime: Starting datetime
            end_datetime: Ending datetime
            interval: Time duration for each interval

        Returns:
            Number of intervals as integer

        Raises:
            ValueError: If end_datetime is before start_datetime
            ValueError: If interval is zero or negative
        """
        if end_datetime < start_datetime:
            raise ValueError("end_datetime must be after start_datetime")

        if interval.total_seconds() <= 0:
            raise ValueError("interval must be positive")

        # Calculate difference in seconds
        diff_seconds = end_datetime.diff(start_datetime).total_seconds()
        interval_seconds = interval.total_seconds()

        # Return ceiling of division to include partial intervals
        return int(np.ceil(diff_seconds / interval_seconds))

    def name_to_key(self, name: str, topic: str) -> Optional[str]:
        """Provides measurement key for given name and topic."""
        topic = topic.lower()

        if topic not in self.topics:
            return None

        topic_keys = [
            key for key in self.config.measurement.model_fields.keys() if key.startswith(topic)
        ]
        key = None
        if topic == "load":
            for config_key in topic_keys:
                if (
                    config_key.endswith("_name")
                    and getattr(self.config.measurement, config_key) == name
                ):
                    key = topic + config_key[len(topic) : len(topic) + 1] + "_mr"
                    break

        if key is not None and key not in self.record_keys:
            # Should never happen
            error_msg = f"Key '{key}' not available."
            logger.error(error_msg)
            raise KeyError(error_msg)

        return key

    def _energy_from_meter_readings(
        self,
        key: str,
        start_datetime: DateTime,
        end_datetime: DateTime,
        interval: Duration,
    ) -> NDArray[Shape["*"], Any]:
        """Calculate an  energy values array indexed by fixed time intervals from energy metering data within an optional date range.

        Args:
            key: Key for energy meter readings.
            start_datetime (datetime): The start date for filtering the energy data (inclusive).
            end_datetime (datetime): The end date for filtering the energy data (exclusive).
            interval (duration): The fixed time interval.

        Returns:
            np.ndarray: A NumPy Array of the energy [kWh] per interval values calculated from
                        the meter readings.
        """
        # Add one interval to end_datetime to assure we have a energy value interval for all
        # datetimes from start_datetime (inclusive) to end_datetime (exclusive)
        end_datetime += interval
        size = self._interval_count(start_datetime, end_datetime, interval)

        energy_mr_array = self.key_to_array(
            key=key, start_datetime=start_datetime, end_datetime=end_datetime, interval=interval
        )
        if energy_mr_array.size != size:
            logging_msg = (
                f"'{key}' meter reading array size: {energy_mr_array.size}"
                f" does not fit to expected size: {size}, {energy_mr_array}"
            )
            if energy_mr_array.size != 0:
                logger.error(logging_msg)
                raise ValueError(logging_msg)
            logger.debug(logging_msg)
            energy_array = np.zeros(size - 1)
        elif np.any(energy_mr_array == None):
            # 'key_to_array()' creates None values array if no data records are available.
            # Array contains None value -> ignore
            debug_msg = f"'{key}' meter reading None: {energy_mr_array}"
            logger.debug(debug_msg)
            energy_array = np.zeros(size - 1)
        else:
            # Calculate load per interval
            debug_msg = f"'{key}' meter reading: {energy_mr_array}"
            logger.debug(debug_msg)
            energy_array = np.diff(energy_mr_array)
            debug_msg = f"'{key}' energy calculation: {energy_array}"
            logger.debug(debug_msg)
        return energy_array

    def load_total(
        self,
        start_datetime: Optional[DateTime] = None,
        end_datetime: Optional[DateTime] = None,
        interval: Optional[Duration] = None,
    ) -> NDArray[Shape["*"], Any]:
        """Calculate a total load energy values array indexed by fixed time intervals from load metering data within an optional date range.

        Args:
            start_datetime (datetime, optional): The start date for filtering the load data (inclusive).
            end_datetime (datetime, optional): The end date for filtering the load data (exclusive).
            interval (duration, optional): The fixed time interval. Defaults to 1 hour.

        Returns:
            np.ndarray: A NumPy Array of the total load energy [kWh] per interval values calculated from
                        the load meter readings.
        """
        if len(self) < 1:
            # No data available
            if start_datetime is None or end_datetime is None:
                size = 0
            else:
                size = self._interval_count(start_datetime, end_datetime, interval)
            return np.zeros(size)
        if interval is None:
            interval = to_duration("1 hour")
        if start_datetime is None:
            start_datetime = self[0].date_time
        if end_datetime is None:
            end_datetime = self[-1].date_time
        size = self._interval_count(start_datetime, end_datetime, interval)
        load_total_array = np.zeros(size)
        # Loop through load<x>_mr
        for i in range(self.record_class().max_loads):
            key = f"load{i}_mr"
            # Calculate load per interval
            load_array = self._energy_from_meter_readings(
                key=key, start_datetime=start_datetime, end_datetime=end_datetime, interval=interval
            )
            # Add calculated load to total load
            load_total_array += load_array
            debug_msg = f"Total load '{key}' calculation: {load_total_array}"
            logger.debug(debug_msg)

        return load_total_array


def get_measurement() -> Measurement:
    """Gets the EOS measurement data."""
    return Measurement()
