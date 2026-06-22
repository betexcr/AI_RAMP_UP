from pydantic import BaseModel

class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]


class Step(BaseModel):
    explanation: str
    output: str


class MathReasoning(BaseModel):
    steps: list[Step]
    final_answer: str

class Weather(BaseModel):
    location: str
    latitude: float
    longitude: float
    geocoder: str
    units: str
    temperature: float
    description: str
    timestamp: str
    weather_code: int
    wind_speed: float
    wind_direction: str
    wind_gust: float
    wind_gust_direction: str
    wind_gust_speed: float
