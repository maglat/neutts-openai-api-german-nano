from pydantic import BaseModel, Field
from typing import Optional, Literal

AudioFormat = Literal["mp3", "wav", "flac", "pcm"]

class OpenAISpeechRequest(BaseModel):
    input: str = Field(..., description="Text input to synthesize.")
    voice: str = Field("default", description="Voice ID mapped to reference samples.")
    response_format: AudioFormat = Field("mp3", description="Output audio format.")

    # Optional extensions (non-OpenAI standard) – bleiben optional:
    speed: Optional[float] = Field(1.0, ge=0.5, le=2.0)
    stream: Optional[bool] = Field(False, description="If true, use chunked streaming response.")
