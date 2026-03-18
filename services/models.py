from dataclasses import dataclass
from pydantic import BaseModel, Field


@dataclass
class SummaryResult:
    website_text: str
    company_summary: str


class CompanyBriefOut(BaseModel):
    company_summary: str = Field(..., description="German company summary")


class EmailOut(BaseModel):
    subject: str = Field(..., description="German subject line")
    email_body: str = Field(..., description="German email body")
