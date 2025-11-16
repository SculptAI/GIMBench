import random

from functools import partial

from datasets import Dataset
from openai import OpenAI


KEYWORDS = [
    "Biophotonics",
    "Urban foraging",
    "Quantum alloys",
    "Cryptoeconomics",
    "Microfiction",
    "Agroecology",
    "Lunar mining",
    "Ritual anthropology",
    "Aquifer recharge",
    "Nanotoxicology",
    "Mythic symbolism",
    "Generative grammar",
    "Bioacoustics",
    "Disaster logistics",
    "Narrative therapy",
    "Blockchain governance",
    "Coral bleaching",
    "Drone choreography",
    "Cyberethics",
    "Ornamental horticulture",
    "Cloud seeding",
    "Behavioral biometrics",
    "Exoplanet climatology",
    "Maritime archaeology",
    "Fermentation science",
    "Industrial robotics",
    "Tea ceremony",
    "Volcanic minerals",
    "Microfluidics",
    "Dark matter mapping",
    "Cultural semiotics",
    "Literary minimalism",
    "Speech pathology",
    "Edge computing",
    "Synthetic biology",
    "Radio ecology",
    "Algorithmic trading",
    "Folk medicine",
    "Toxic leadership",
    "Paleo cuisine",
    "Ecofeminism",
    "Antimicrobial coatings",
    "Ocean governance",
    "Interactive fiction",
    "Market microstructure",
    "Dream interpretation",
    "Biogeographical modeling",
    "Sports analytics",
    "Cyberpunk aesthetics",
    "Food sovereignty",
    "Space law",
    "Emotion regulation",
    "High-frequency welding",
    "Desert hydrology",
    "Cognitive mapping",
    "Precision viticulture",
    "Zen gardening",
    "Atmospheric chemistry",
    "Biodegradable plastics",
    "Digital paleography",
    "Wildfire modeling",
    "Soundscape ecology",
    "Quantum cryptography",
    "Ceramic engineering",
    "Affective computing",
    "Environmental justice",
    "Vision restoration",
    "Woodblock printing",
    "Neural architecture search",
    "Behavioral finance",
    "Folk astronomy",
    "Ultrasonic sensing",
    "Marine virology",
    "Ergonomic design",
    "Deep-sea drilling",
    "Symbolic logic",
    "Immersive theater",
    "Memory consolidation",
    "Terahertz imaging",
    "Narrative worldbuilding",
    "Ocean acidification",
    "Disaster psychology",
    "Textile conservation",
    "Structural seismology",
    "Viral genomics",
    "Computational linguistics",
    "Cultural diplomacy",
    "Wind tunnel testing",
    "Herbal distillation",
    "Autonomous navigation",
    "Epigenetic drift",
    "Glacial geomorphology",
    "Ceramic pigments",
    "Participatory budgeting",
    "Robotic prosthetics",
    "Infrared astronomy",
    "Soil carbon flux",
    "Aquaponic systems",
    "Mythmaking traditions",
    "Immune profiling",
]

TYPES = [
    "three sentences",
    "a one-paragraph story",
    "20 lines of code",
    "a poem with no more than 12 lines",
    "a TOML configuration file with no more than 15 lines",
    "a JSON datapoint with no more than 15 lines",
    "a summary with no more than 100 words",
    "a mathematical description with no more than 200 words",
    "a tech article with no more than 500 words",
    "a XML document with no more than 15 lines",
    "a Python script with no more than 20 lines",
    "a C++ program with no more than 20 lines",
]

LANGUAGES = [
    "Spanish",
    "French",
    "German",
    "Italian",
    "Japanese",
    "Korean",
    "Russian",
    "Portuguese",
] + [
    "Chinese",
    "English",
] * 12

PROMPT_TEMPLATE = """Please generate {type} about {keywords_list} in {language}."""

SEED = 42
BASE_URL = "https://api.siliconflow.cn/v1"
MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"
MAX_TOKENS = 2048
TEMPERATURE = 0.7
COUNT = 2000
DATASET_NAME = "Sculpt-AI/random-articles"


def random_config() -> tuple[str, list[str], str]:
    article_type = random.choice(TYPES)
    keywords = random.sample(KEYWORDS, k=random.randint(1, 3))
    language = random.choice(LANGUAGES)
    return article_type, keywords, language


def format_prompt(article_type: str, keywords: list[str], language: str) -> str:
    article_type = random.choice(TYPES)
    keywords = random.sample(KEYWORDS, k=random.randint(1, 3))
    language = random.choice(LANGUAGES)

    keywords_list = (
        ", ".join(keywords[:-1]) + ", and " + keywords[-1] if len(keywords) > 1 else keywords[0]
    )
    return PROMPT_TEMPLATE.format(
        type=article_type,
        keywords_list=keywords_list,
        language=language,
    )


def model_request(client: OpenAI, prompt: str) -> str:
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
    except Exception as e:
        print(f"Error during model request: {e}")
        return ""

    response_content = response.choices[0].message.content
    return response_content or ""


def article_generator(count: int):
    client = OpenAI(base_url=BASE_URL)
    for _ in range(count):
        article_type, keywords, language = random_config()
        prompt = format_prompt(article_type, keywords, language)
        article = model_request(client, prompt)
        yield {
            "type": article_type,
            "keywords": keywords,
            "language": language,
            "prompt": prompt,
            "article": article,
        }


if __name__ == "__main__":
    random.seed(SEED)
    _generator = partial(article_generator, count=COUNT)
    ds = Dataset.from_generator(_generator)
    ds = ds.filter(lambda x: len(x["article"].strip()) > 0)
    ds.save_to_disk(DATASET_NAME.replace("/", "_"))
    ds.push_to_hub(DATASET_NAME)
