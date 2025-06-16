"""Retrieves and processes electricity price forecast data from Tibber API.

This module provides classes and mappings to manage electricity price data obtained from the
Tibber API. The data is mapped to the `ElecPriceDataRecord`
format, enabling consistent access to forecasted and historical electricity price attributes.
"""

from typing import Any, List, Optional, Union

import time 

import numpy as np
import pandas as pd
from datetime import datetime
import requests
from loguru import logger
from pydantic import ValidationError
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from akkudoktoreos.core.cache import cache_in_file
from akkudoktoreos.core.pydantic import PydanticBaseModel
from akkudoktoreos.prediction.elecpriceabc import ElecPriceProvider
from akkudoktoreos.utils.datetimeutil import to_datetime, to_duration

# TODO: add to config
TIBBER_API_KEY = "<putYourOwnApiKey>"
TIBBER_HOME_INDEX = 0
TIBBER_HEADERS = {"Authorization": f"Bearer {TIBBER_API_KEY}"}
TIBBER_PRICE_QUERY = """{ viewer 
  { 
    homes { 
      currentSubscription { 
        priceInfo {
          range (resolution: HOURLY, last: 720){
            nodes {
              total startsAt
            }
          }
          today { 
            total startsAt } 
          tomorrow { 
            total startsAt } 
        } 
      } 
    }
  }
}"""

# class TibberElecPriceMeta(PydanticBaseModel):
#     start_timestamp: str
#     end_timestamp: str
#     start: str
#     end: str


# class TibberElecPriceValue(PydanticBaseModel):
#     start_timestamp: int
#     end_timestamp: int
#     start: str
#     end: str
#     marketprice: float
#     unit: str
#     marketpriceEurocentPerKWh: float

class TibberElecPriceValue(PydanticBaseModel):
    start_timestamp: int
    marketpriceEuroPerKWh: float

class TibberElecPrice(PydanticBaseModel):
    values: List[TibberElecPriceValue]


class ElecPriceTibber(ElecPriceProvider):
    """Fetch and process electricity price from Tibber.

    ElecPriceTibber is a singleton-based class that retrieves electricity price forecast data
    from the Tibber API and maps it to `ElecPriceDataRecord` fields, applying
    any necessary scaling or unit corrections. It manages the forecast over a range
    of hours into the future and retains historical data.

    Attributes:
        hours (int, optional): Number of hours in the future for the forecast.
        historic_hours (int, optional): Number of past hours for retaining data.
        start_datetime (datetime, optional): Start datetime for forecasts, defaults to the current datetime.
        end_datetime (datetime, computed): The forecast's end datetime, computed based on `start_datetime` and `hours`.
        keep_datetime (datetime, computed): The datetime to retain historical data, computed from `start_datetime` and `historic_hours`.

    Methods:
        provider_id(): Returns a unique identifier for the provider.
        _request_forecast(): Fetches the forecast from the Tibber API.
        _update_data(): Processes and updates forecast data from Tibber in ElecPriceDataRecord format.
    """

    @classmethod
    def provider_id(cls) -> str:
        """Return the unique identifier for the Tibber provider."""
        return "ElecPriceTibber"

    @classmethod
    def _validate_data(cls, json: Union[bytes, Any]) -> TibberElecPrice:
        """Validate Tibber Electricity Price forecast data."""
        try:
            tibber_data = TibberElecPrice.model_validate(json)
        except ValidationError as e:
            error_msg = ""
            for error in e.errors():
                field = " -> ".join(str(x) for x in error["loc"])
                message = error["msg"]
                error_type = error["type"]
                error_msg += f"Field: {field}\nError: {message}\nType: {error_type}\n"
            logger.error(f"Tibber schema change: {error_msg}")
            raise ValueError(error_msg)
        return tibber_data

    @cache_in_file(with_ttl="1 hour")
    def _request_forecast(self) -> TibberElecPrice:
        """Fetch electricity price forecast data from Tibber.

        This method sends a request to Tibber's API to retrieve forecast data for a specified
        date range. The response data is parsed and returned as JSON for further processing.

        Returns:
            dict: The parsed JSON response from Tibber API containing forecast data.

        Raises:
            ValueError: If the API response does not include expected `electricity price` data.

        Todo:
            - add the file cache again.
        """
        source = "https://api.tibber.com/"

        url = f"{source}v1-beta/gql" 
        response = requests.post(url, headers=TIBBER_HEADERS, json={"query": TIBBER_PRICE_QUERY}, timeout=10)
        logger.debug(f"Response from {url}: {response}")
        response.raise_for_status()  # Raise an error for bad responses
        
        json_data = response.json()
        # encapsel data
        history = json_data['data']['viewer']['homes'][TIBBER_HOME_INDEX]['currentSubscription']['priceInfo']['range']['nodes']  # past prices 
        today =  history + json_data['data']['viewer']['homes'][TIBBER_HOME_INDEX]['currentSubscription']['priceInfo']['today']  # today prices 
        prices = today + json_data['data']['viewer']['homes'][TIBBER_HOME_INDEX]['currentSubscription']['priceInfo']['tomorrow'] # tomorrow prices
        tibber_data = [ TibberElecPriceValue(start_timestamp=int(time.mktime(datetime.fromisoformat(p['startsAt']).timetuple())), marketpriceEuroPerKWh=p['total']) for p in prices ]

        tibber_prices = self._validate_data(TibberElecPrice(values=tibber_data))
        
        # We are working on fresh data (no cache), report update time
        self.update_datetime = to_datetime(in_timezone=self.config.general.timezone)
        return tibber_prices

    def _cap_outliers(self, data: np.ndarray, sigma: int = 2) -> np.ndarray:
        mean = data.mean()
        std = data.std()
        lower_bound = mean - sigma * std
        upper_bound = mean + sigma * std
        capped_data = data.clip(min=lower_bound, max=upper_bound)
        return capped_data

    def _predict_ets(self, history: np.ndarray, seasonal_periods: int, hours: int) -> np.ndarray:
        clean_history = self._cap_outliers(history)
        model = ExponentialSmoothing(
            clean_history, seasonal="add", seasonal_periods=seasonal_periods
        ).fit()
        return model.forecast(hours)

    def _predict_median(self, history: np.ndarray, hours: int) -> np.ndarray:
        clean_history = self._cap_outliers(history)
        return np.full(hours, np.median(clean_history))

    def _update_data(
        self, force_update: Optional[bool] = False
    ) -> None:  # tuple[np.ndarray, np.ndarray]
        """Update forecast data in the ElecPriceDataRecord format.

        Retrieves data from Akkudoktor, maps each Akkudoktor field to the corresponding
        `ElecPriceDataRecord` and applies any necessary scaling.

        The final mapped and processed data is inserted into the sequence as `ElecPriceDataRecord`.
        """
        # Get Akkudoktor electricity price data
        tibber_data = self._request_forecast(force_update=force_update)  # type: ignore
        if not self.start_datetime:
            raise ValueError(f"Start DateTime not set: {self.start_datetime}")

        # Assumption that all lists are the same length and are ordered chronologically
        # in ascending order and have the same timestamps.

        # Get charges_kwh in wh
        
        #charges_wh = (self.config.elecprice.charges_kwh or 0) / 1000

        highest_orig_datetime = None  # newest datetime from the api after that we want to update.
        series_data = pd.Series(dtype=float)  # Initialize an empty series

        for value in tibber_data.values:
            orig_datetime = to_datetime(value.start_timestamp, in_timezone=self.config.general.timezone)
            if highest_orig_datetime is None or orig_datetime > highest_orig_datetime:
                highest_orig_datetime = orig_datetime

            price_wh = value.marketpriceEuroPerKWh / 1000

            # Collect all values into the Pandas Series
            series_data.at[orig_datetime] = price_wh

        # Update values using key_from_series
        self.key_from_series("elecprice_marketprice_wh", series_data)

        # Generate history array for prediction
        history = self.key_to_array(
            key="elecprice_marketprice_wh", end_datetime=highest_orig_datetime, fill_method="linear"
        )

        amount_datasets = len(self.records)
        if not highest_orig_datetime:  # mypy fix
            error_msg = f"Highest original datetime not available: {highest_orig_datetime}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # some of our data is already in the future, so we need to predict less. If we got less data we increase the prediction hours
        needed_hours = int(
            self.config.prediction.hours
            - ((highest_orig_datetime - self.start_datetime).total_seconds() // 3600)
        )

        if needed_hours <= 0:
            logger.warning(
                f"No prediction needed. needed_hours={needed_hours}, hours={self.config.prediction.hours},highest_orig_datetime {highest_orig_datetime}, start_datetime {self.start_datetime}"
            )  # this might keep data longer than self.start_datetime + self.config.prediction.hours in the records
            return

        if amount_datasets > 800:  # we do the full ets with seasons of 1 week
            prediction = self._predict_ets(history, seasonal_periods=168, hours=needed_hours)
        elif amount_datasets > 168:  # not enough data to do seasons of 1 week, but enough for 1 day
            prediction = self._predict_ets(history, seasonal_periods=24, hours=needed_hours)
        elif amount_datasets > 0:  # not enough data for ets, do median
            prediction = self._predict_median(history, hours=needed_hours)
        else:
            logger.error("No data available for prediction")
            raise ValueError("No data available")

        # write predictions into the records, update if exist.
        prediction_series = pd.Series(
            data=prediction,
            index=[
                highest_orig_datetime + to_duration(f"{i + 1} hours")
                for i in range(len(prediction))
            ],
        )
        self.key_from_series("elecprice_marketprice_wh", prediction_series)

        # history2 = self.key_to_array(key="elecprice_marketprice_wh", fill_method="linear") + 0.0002
        return None #history, prediction  # for debug main


"""
def visualize_predictions(
    history: np.ndarray[Any, Any],
    history2: np.ndarray[Any, Any],
    predictions: np.ndarray[Any, Any],
) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(28, 14))
    plt.plot(range(len(history)), history, label="History", color="green")
    plt.plot(range(len(history2)), history2, label="History_new", color="blue")
    plt.plot(
        range(len(history), len(history) + len(predictions)),
        predictions,
        label="Predictions",
        color="red",
    )
    plt.title("Predictions ets")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.savefig("predictions_vs_true.png")
    plt.close()


def main() -> None:
    # Initialize ElecPriceAkkudoktor with required parameters
    elec_price_akkudoktor = ElecPriceAkkudoktor()
    history, history2, predictions = elec_price_akkudoktor._update_data()

    visualize_predictions(history, history2, predictions)
    # print(history, history2, predictions)


if __name__ == "__main__":
    main()
"""
