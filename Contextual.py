from datetime import datetime
import geocoder
import requests
import os

# Get API key from environment variable (or hardcoded)
WEATHER_API_KEY = "0341599fc3fc9da59ce4bdc61fba34fd"

def get_device_info():
    """Collect device information including time, date, location and weather"""
    # Initialize with default values
    data = ["N/A"] * 4
    
    try:
        # Get time and date
        now = datetime.now()
        data[0] = now.strftime("%H:%M:%S")
        data[1] = now.strftime("%Y-%m-%d")
        
        # Get location
        try:
            g = geocoder.ip('me')
            if not g.ok:
                raise ValueError("Geocoder request failed")
                
            location_parts = []
            if g.city: location_parts.append(g.city)
            if g.state: location_parts.append(g.state)
            if g.country: location_parts.append(g.country)
            
            data[2] = ", ".join(location_parts) if location_parts else "Location unavailable"

            # Get weather if location found and coordinates available
            if location_parts and hasattr(g, 'latlng') and g.latlng:
                try:
                    if not WEATHER_API_KEY:
                        raise ValueError("Weather API key not set")
                    
                    lat, lon = g.latlng[:2]
                    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric"
                    
                    # Get weather data with timeout
                    response = requests.get(url, timeout=5)
                    response.raise_for_status()
                    weather_data = response.json()
                    
                    # Parse weather response
                    if 'main' in weather_data and 'temp' in weather_data['main']:
                        temp = weather_data['main']['temp']
                        description = weather_data['weather'][0]['description'] if weather_data.get('weather') else "N/A"
                        data[3] = f"{temp}°C, {description}"
                    else:
                        data[3] = "Weather data incomplete"
                        
                except requests.exceptions.RequestException:
                    pass  # Fail silently
                except ValueError:
                    pass  # Fail silently
        except Exception:
            data[2] = "Location unavailable"
                
    except Exception:
        pass  # Fail silently
    
    return data

if __name__ == "__main__":
    device_data = get_device_info()
    print(device_data)