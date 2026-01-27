from datasets import Features, Pdf, Value
from pydantic import BaseModel, Field


extraction_prompt = """You are an expert CV parser. Extract the information from the following CV text into a structured JSON object.

###################
## CV Text
###################

{cv_text}

###################
## Instructions
###################

- Strictly follow the original text when extracting values.
- Return only a valid JSON object (no extra commentary).
- Use null for fields that are missing and cannot be reasonably inferred.
- All dates must use the YYYY-MM-DD format. If only a year is present, normalize to YYYY-01-01. If year and month are present, normalize to YYYY-MM-01.
- If the birth date is not provided, you may infer a plausible birth date based on the candidate's education history and typical local schooling ages/system; state an inferred value or null if inference is not reasonable.
- If country/nationality is not provided, you may infer nationality or country of residence from education, institutions, or other clear clues; otherwise use null.
"""


class CVData(BaseModel):
    # Basic Info
    name: str = Field(..., description="The full name of the person")
    country: str | None = Field(..., description="The content of nationality or country of residence")
    birthday: str | None = Field(
        ..., description="Birthday in YYYY-MM-DD format if available", pattern=r"\d{4}-\d{2}-\d{2}"
    )
    phone_number: str | None = Field(
        ...,
        description="Contact phone number",
        pattern=r"(?:\+?(\d{1,3}))?([-. (]*(\d{3})[-. )]*)?((\d{3})[-. ]*(\d{2,4})(?:[-.x ]*(\d+))?)",
    )
    email: str | None = Field(
        ..., description="Email address", pattern=r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    )

    # Education Background
    highest_level_degree: str | None = Field(
        ..., description="The highest degree obtained", pattern=r"(Bachelor|Master|PhD)"
    )
    university: str | None = Field(..., description="Name of the university for the highest degree")
    department: str | None = Field(..., description="Department name for the highest degree")
    major: str | None = Field(..., description="Major of study")
    start_date: str | None = Field(..., description="Start date of the degree", pattern=r"\d{4}-\d{2}-\d{2}")
    end_date: str | None = Field(..., description="End date or expected graduation date", pattern=r"\d{4}-\d{2}-\d{2}")

    # Profile
    homepage_url: str | None = Field(..., description="Personal homepage URL")
    twitter_url: str | None = Field(
        ..., description="Twitter/X profile URL", pattern=r"https?:\/\/(www\.)?(x|twitter)\.com\/[a-zA-Z0-9_]+\/?"
    )
    github_url: str | None = Field(
        ...,
        description="GitHub profile URL",
        pattern=r"https?:\/\/(www\.)?github\.com\/[a-zA-Z0-9-]+(\/[a-zA-Z0-9-]+)*\/?",
    )
    google_scholar_url: str | None = Field(
        ...,
        description="Google Scholar profile URL",
        pattern=r"https?:\/\/(scholar\.google\.com\/citations\?user=[a-zA-Z0-9-]+)",
    )


cv_data_schema = CVData.model_json_schema()


def get_hf_features() -> Features:
    """
    Returns the Hugging Face Dataset Features corresponding to the flattened schema.
    """
    return Features(
        {
            "pdf": Pdf(),
            "file_name": Value("string"),
            "extracted_text": Value("string"),
            # Basic
            "name": Value("string"),
            "country": Value("string"),
            "birthday": Value("string"),
            "phone_number": Value("string"),
            "email": Value("string"),
            # Education
            "highest_level_degree": Value("string"),
            "university": Value("string"),
            "department": Value("string"),
            "major": Value("string"),
            "start_date": Value("string"),
            "end_date": Value("string"),
            # Profile
            "homepage_url": Value("string"),
            "twitter_url": Value("string"),
            "github_url": Value("string"),
            "google_scholar_url": Value("string"),
        }
    )
