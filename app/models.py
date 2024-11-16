from pydantic import BaseModel
from typing import Any, List, Optional, Optional
from enum import Enum


class Item(BaseModel):
    name: str
    description: str = None
    price: float
    tax: float = None


class Summary(BaseModel):
    data: str


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    messages: List[Message]


class Chat(BaseModel):
    user_p: str
    id_: str


class Name(BaseModel):
    id_: str


class RequestWithSessionid(BaseModel):
    session_id: str
    user_input: str


class Data(BaseModel):
    sender: str
    timestamp: int
    message: str


class ChatRequest(BaseModel):
    type: str
    platform: str
    id_: str
    data: List[Message]


class Platforms(str, Enum):
    LINKEDIN_RECRUITER_LITE = "LINKEDIN_RECRUITER_LITE"
    INDEED = "INDEED"


class EventTypes(str, Enum):
    SEND_MESSAGE_REPLY = "SEND_MESSAGE_REPLY"


class Message(BaseModel):
    sender: str
    timestamp: int
    message: str = ""


class SendMessageEvent(BaseModel):
    id_: str
    etype: str = EventTypes.SEND_MESSAGE_REPLY
    platform: str = Platforms.LINKEDIN_RECRUITER_LITE
    data: List[Message] = []
    timestamp: int

    # def __init__(self, id_, etype, platform, data, timestamp):
    #     super().__init__()
    #     self.id_=id_
    #     self.etype=etype
    #     self.platform=platform
    #     self.data=data
    #     self.timestamp=timestamp/


class CSP(BaseModel):
    id_: str
    jobdescription: str


class IdealCandidateProfile(BaseModel):
    titles: List[str] = []
    skills: List[str] = []
    locations: List[str] = []
    companies: List[str] = []
    industries: List[str] = []
    keywords: List[str] = []
    summary: str = ""
