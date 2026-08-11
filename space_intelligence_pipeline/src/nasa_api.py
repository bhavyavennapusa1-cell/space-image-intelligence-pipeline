"""
NASA Open API Client module.
Fetches APOD (Astronomy Picture of the Day) and NeoWs (Near Earth Object Web Service) telemetry.
"""

from typing import Dict, Any, Optional
import requests
from src.config import NASA_API_KEY, NASA_APOD_URL, NASA_NEOWS_URL

class NASADataClient:
    """
    Client interface interacting with NASA Open APIs.
    """
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or NASA_API_KEY

    def get_apod(self) -> Dict[str, Any]:
        """
        Fetches the latest Astronomy Picture of the Day data.
        """
        params = {"api_key": self.api_key}
        try:
            response = requests.get(NASA_APOD_URL, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as err:
            return {"error": f"Failed to fetch APOD: {str(err)}"}

    def get_near_earth_objects(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Fetches Near-Earth Asteroid trajectory metrics within specified date window.
        """
        params = {
            "start_date": start_date,
            "end_date": end_date,
            "api_key": self.api_key
        }
        try:
            response = requests.get(NASA_NEOWS_URL, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as err:
            return {"error": f"Failed to fetch NeoWs data: {str(err)}"}
