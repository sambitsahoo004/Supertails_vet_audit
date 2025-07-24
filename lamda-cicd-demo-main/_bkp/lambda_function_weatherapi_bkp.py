import json
import requests
from datetime import datetime, timedelta


def lambda_handler(event, context):
    url = "https://api.weather.gov/gridpoints/MPX/107,70/forecast"
    response = requests.get(url)

    if response.status_code != 200:
        return {
            "statusCode": response.status_code,
            "body": json.dumps({"error": "Failed to fetch weather data"}),
        }

    data = response.json()
    periods = data.get("properties", {}).get("periods", [])

    tomorrow_date = (datetime.utcnow() + timedelta(days=1)).date()

    tomorrow_forecasts = []
    for period in periods:
        start_time = datetime.fromisoformat(period["startTime"].replace("Z", "+00:00"))
        if start_time.date() == tomorrow_date:
            tomorrow_forecasts.append(period)

    for forecast in tomorrow_forecasts:
        print("=============================")
        print("Forecast for:", forecast.get("name", "No name available"))
        print("Tomorrow's date:", tomorrow_date)
        print(forecast.get("detailedForecast", "No detailed forecast available."))
        print("=============================")
    if not tomorrow_forecasts:
        return {
            "statusCode": 200,
            "body": json.dumps({"message": "No forecast data found for tomorrow."}),
        }

    return {"statusCode": 200, "body": json.dumps(tomorrow_forecasts)}
