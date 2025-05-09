from typing import List
from pydantic import BaseModel

class Page(BaseModel):
    content: List[str]

class Document(BaseModel):
    pages: List[Page]
