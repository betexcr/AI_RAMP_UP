from pydantic import BaseModel, Field


class TextRequest(BaseModel):
    text: str = Field(min_length=1)


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1)


class SearchRequest(BaseModel):
    search_query: str = Field(min_length=1)


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
