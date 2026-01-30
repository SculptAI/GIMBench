from gimkit import guide as g
from pydantic import BaseModel, Field


CV_FIELDS = [
    "name",
    "country",
    "birthday",
    "phone_number",
    "email",
    "highest_level_degree",
    "university",
    "department",
    "major",
    "start_date",
    "end_date",
    "homepage_url",
    "github_url",
]

SHARED_PROMPT_PREFIX = """You are an expert on extracting key information from a CV document.

## CV Content

{cv_content}
"""


date_regex = r"(?:\d{4}-\d{2}-\d{2})?"
phone_regex = r"(?:\+?(\d{1,3}))?([-. (]*(\d{3})[-. )]*)?((\d{3})[-. ]*(\d{2,4})(?:[-.x ]*(\d+))?)?"
email_regex = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
degree_regex = r"(Bachelor|Master|PhD)?"
url_regex = r"(?:https?:\/\/(www\.)?[a-zA-Z0-9-]+\.[a-zA-Z]{2,}(\/[a-zA-Z0-9-]+)*\/?)?"


GIMKIT_TEMPLATE = f"""
## Fields

### Basic Info

- Name: {g(name="name")}
- Country: {g(name="country", desc="If country/nationality is not provided, you may infer nationality or country of residence from education, institutions, or other clear clues.")}
- Birthday: {g(name="birthday", desc="If not given, try to infer from context. If only a year is present, normalize to YYYY-01-01. If year and month are present, normalize to YYYY-MM-01.", regex=date_regex)}
- Phone Number: {g(name="phone_number", regex=phone_regex)}
- Email: {g(name="email", regex=email_regex)}

### Highest Level Degree

- Highest Level Degree: {g(name="highest_level_degree", regex=degree_regex)}
- University: {g(name="university")}
- Department: {g(name="department")}
- Major: {g(name="major")}
- Start Date: {g(name="start_date", regex=date_regex)}
- End Date: {g(name="end_date", regex=date_regex)}

### Profile

- Homepage URL: {g(name="homepage_url", regex=url_regex)}
- GitHub URL: {g(name="github_url", regex=url_regex)}
"""


OUTLINES_TEMPLATE = """
## Note

- Extract the following fields from the CV content: Name, Country, Birthday, Phone Number, Email, Highest Level Degree, University, Department, Major, Start Date, End Date, Homepage URL, GitHub URL.
- If country/nationality is not provided, you may infer nationality or country of residence from education, institutions, or other clear clues.
- For Birthday, if not given, try to infer from context. If only a year is present, normalize to YYYY-01-01. If year and month are present, normalize to YYYY-MM-01.
"""


class CVData(BaseModel):
    # Basic Info
    name: str = Field(...)
    country: str | None = Field(...)
    birthday: str | None = Field(..., pattern=date_regex)
    phone_number: str | None = Field(..., pattern=phone_regex)
    email: str | None = Field(..., pattern=email_regex)

    # Education Background
    highest_level_degree: str | None = Field(..., pattern=degree_regex)
    university: str | None = Field(...)
    department: str | None = Field(...)
    major: str | None = Field(...)
    start_date: str | None = Field(..., pattern=date_regex)
    end_date: str | None = Field(..., pattern=date_regex)
    # Profile
    homepage_url: str | None = Field(..., pattern=url_regex)
    github_url: str | None = Field(..., pattern=url_regex)


OUTLINES_JSON_SCHEMA = CVData.model_json_schema()
