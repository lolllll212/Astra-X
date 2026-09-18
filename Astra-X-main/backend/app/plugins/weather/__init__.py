"""Weather Plugin — provides current weather and forecast data.

Supports multiple weather providers with configurable API keys.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from app.plugins.base import Plugin, PluginContext
from app.tools.base import Tool
from app.tools.context import ToolContext
from app.tools.models import ToolParameter, ToolSchema
from app.tools.result import ToolResult


class WeatherTool(Tool):
    """Weather tool with support for multiple providers."""

    def __init__(self, api_key: str | None = None, provider: str = "openweather") -> None:
        self._api_key = api_key or os.getenv("OPENWEATHER_API_KEY")
        self._provider = provider
        self._base_urls = {
            "openweather": "https://api.openweathermap.org/data/2.5",
            "weatherapi": "https://api.weatherapi.com/v1",
        }

    @property
    def name(self) -> str:
        return "weather"

    @property
    def description(self) -> str:
        return "Get current weather or forecast for a given location. Supports OpenWeatherMap and WeatherAPI providers."

    @property
    def capabilities(self) -> list[str]:
        return ["get_weather", "get_forecast"]

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="location",
                    type_="string",
                    description="City name, coordinates (lat,lon), or ZIP code",
                    required=True,
                ),
                ToolParameter(
                    name="type",
                    type_="string",
                    description="Type of data: 'current' or 'forecast'",
                    required=False,
                    default="current",
                    enum=["current", "forecast"],
                ),
                ToolParameter(
                    name="days",
                    type_="integer",
                    description="Number of forecast days (1-5, only for forecast type)",
                    required=False,
                    default=3,
                ),
                ToolParameter(
                    name="units",
                    type_="string",
                    description="Temperature units: 'metric', 'imperial', or 'standard'",
                    required=False,
                    default="metric",
                    enum=["metric", "imperial", "standard"],
                ),
            ],
            capabilities=self.capabilities,
        )

    async def _execute(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        location: str = kwargs.get("location", "")
        data_type: str = kwargs.get("type", "current")
        days: int = int(kwargs.get("days", 3))
        units: str = kwargs.get("units", "metric")

        if not location:
            return ToolResult(success=False, error="location is required")

        if not self._api_key:
            return await self._get_mock_weather(location, data_type, units)

        try:
            if self._provider == "openweather":
                return await self._get_openweather(location, data_type, days, units)
            elif self._provider == "weatherapi":
                return await self._get_weatherapi(location, data_type, days, units)
            else:
                return ToolResult(success=False, error=f"Unknown provider: {self._provider}")
        except httpx.HTTPStatusError as exc:
            return ToolResult(success=False, error=f"API error: {exc.response.status_code}")
        except httpx.RequestError as exc:
            return ToolResult(success=False, error=f"Network error: {exc}")
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))

    async def _get_openweather(self, location: str, data_type: str, days: int, units: str) -> ToolResult:
        async with httpx.AsyncClient(timeout=15.0) as client:
            if data_type == "current":
                resp = await client.get(
                    f"{self._base_urls['openweather']}/weather",
                    params={"q": location, "appid": self._api_key, "units": units},
                )
                resp.raise_for_status()
                data = resp.json()

                temp = data["main"]["temp"]
                feels_like = data["main"]["feels_like"]
                humidity = data["main"]["humidity"]
                description = data["weather"][0]["description"]
                wind_speed = data["wind"]["speed"]
                city = data["name"]
                country = data["sys"]["country"]

                unit_symbol = "°C" if units == "metric" else "°F" if units == "imperial" else "K"
                wind_unit = "m/s" if units == "metric" else "mph"

                output = (
                    f"Weather in {city}, {country}: {temp}{unit_symbol} (feels like {feels_like}{unit_symbol}), "
                    f"{description}, humidity {humidity}%, wind {wind_speed} {wind_unit}."
                )
                return ToolResult(success=True, output=output, metadata=data)

            else:
                resp = await client.get(
                    f"{self._base_urls['openweather']}/forecast",
                    params={"q": location, "appid": self._api_key, "units": units, "cnt": days * 8},
                )
                resp.raise_for_status()
                data = resp.json()

                forecasts = []
                for item in data["list"][:days]:
                    dt = item["dt_txt"]
                    temp = item["main"]["temp"]
                    desc = item["weather"][0]["description"]
                    unit_symbol = "°C" if units == "metric" else "°F" if units == "imperial" else "K"
                    forecasts.append(f"{dt}: {temp}{unit_symbol}, {desc}")

                output = f"Forecast for {data['city']['name']}, {data['city']['country']}:\n" + "\n".join(forecasts)
                return ToolResult(success=True, output=output, metadata=data)

    async def _get_weatherapi(self, location: str, data_type: str, days: int, units: str) -> ToolResult:
        async with httpx.AsyncClient(timeout=15.0) as client:
            endpoint = "forecast.json" if data_type == "forecast" else "current.json"
            resp = await client.get(
                f"{self._base_urls['weatherapi']}/{endpoint}",
                params={"key": self._api_key, "q": location, "days": days, "aqi": "no"},
            )
            resp.raise_for_status()
            data = resp.json()

            if data_type == "current":
                current = data["current"]
                location_info = data["location"]
                temp_c = current["temp_c"]
                temp_f = current["temp_f"]
                condition = current["condition"]["text"]
                humidity = current["humidity"]
                wind_kph = current["wind_kph"]
                wind_mph = current["wind_mph"]

                if units == "metric":
                    temp = f"{temp_c}°C"
                    wind = f"{wind_kph} km/h"
                elif units == "imperial":
                    temp = f"{temp_f}°F"
                    wind = f"{wind_mph} mph"
                else:
                    temp = f"{temp_c + 273.15:.1f}K"
                    wind = f"{wind_kph} km/h"

                output = (
                    f"Weather in {location_info['name']}, {location_info['country']}: {temp}, "
                    f"{condition}, humidity {humidity}%, wind {wind}."
                )
                return ToolResult(success=True, output=output, metadata=data)
            else:
                location_info = data["location"]
                forecasts = []
                for day in data["forecast"]["forecastday"][:days]:
                    date = day["date"]
                    day_data = day["day"]
                    condition = day_data["condition"]["text"]
                    max_temp = day_data["maxtemp_c"] if units == "metric" else day_data["maxtemp_f"]
                    min_temp = day_data["mintemp_c"] if units == "metric" else day_data["mintemp_f"]
                    unit_symbol = "°C" if units == "metric" else "°F"
                    forecasts.append(f"{date}: {condition}, high {max_temp}{unit_symbol}, low {min_temp}{unit_symbol}")

                output = f"Forecast for {location_info['name']}, {location_info['country']}:\n" + "\n".join(forecasts)
                return ToolResult(success=True, output=output, metadata=data)

    async def _get_mock_weather(self, location: str, data_type: str, units: str) -> ToolResult:
        unit_symbol = "°C" if units == "metric" else "°F" if units == "imperial" else "K"
        if data_type == "current":
            output = f"Weather in {location}: 22{unit_symbol}, partly cloudy, humidity 45%, wind 12 km/h. (Mock data - set API key for real data)"
        else:
            output = f"Forecast for {location}:\nDay 1: 22{unit_symbol}, partly cloudy\nDay 2: 20{unit_symbol}, rain\nDay 3: 18{unit_symbol}, sunny. (Mock data)"
        return ToolResult(success=True, output=output, metadata={"mock": True})


class WeatherPlugin(Plugin):
    """Plugin that registers WeatherTool on load and unregisters on unload."""

    def __init__(self, api_key: str | None = None, provider: str = "openweather") -> None:
        self._tool: WeatherTool | None = None
        self._context: PluginContext | None = None
        self._api_key = api_key
        self._provider = provider

    @property
    def name(self) -> str:
        return "weather"

    async def on_load(self, context: PluginContext) -> None:
        self._context = context
        self._tool = WeatherTool(api_key=self._api_key, provider=self._provider)
        tr = context.registries.tool
        if tr:
            tr.register(self._tool)
        cr = context.registries.capability
        if cr:
            cr.rebuild()

    async def on_unload(self) -> None:
        if self._tool is not None and self._context is not None:
            tr = self._context.registries.tool
            if tr and tr.exists(self._tool.name):
                tr.unregister(self._tool.name)
            cr = self._context.registries.capability
            if cr:
                cr.rebuild()
        self._tool = None
        self._context = None