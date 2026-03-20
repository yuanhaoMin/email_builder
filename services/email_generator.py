from pathlib import Path

from services.llm_client import generate_structured_output
from services.models import EmailOut

PROMPTS_DIR = Path("prompts")
DEFAULT_PROMPT_PATH = PROMPTS_DIR / "it4you_verband.txt"


def list_prompt_files() -> list[Path]:
    if not PROMPTS_DIR.exists():
        return []

    return sorted(
        [p for p in PROMPTS_DIR.glob("*.txt") if p.is_file()],
        key=lambda p: p.name.lower(),
    )


def load_prompt_file(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Prompt file is empty: {path}")

    return text


def safe_replace_prompt(
    template: str,
    company_summary: str,
    first_name: str,
    last_name: str,
    job_title: str,
) -> str:
    company_summary = (company_summary or "").strip()
    if len(company_summary) > 4000:
        company_summary = company_summary[:4000]

    first_name = (first_name or "Johann").strip()
    last_name = (last_name or "Min").strip()
    job_title = (job_title or "").strip()

    prompt = template
    prompt = prompt.replace("{company_summary}", company_summary)
    prompt = prompt.replace("{first_name}", first_name)
    prompt = prompt.replace("{last_name}", last_name)
    prompt = prompt.replace("{job_title}", job_title)
    return prompt


async def build_email(
    company_summary: str,
    prompt_template: str,
    first_name: str = "Johann",
    last_name: str = "Min",
    job_title: str = "CEO",
    model: str = "gpt-5.1",
):
    if not prompt_template or not prompt_template.strip():
        raise ValueError("Prompt template is empty.")

    if not model or not model.strip():
        raise ValueError("Model must not be empty.")

    prompt = safe_replace_prompt(
        template=prompt_template,
        company_summary=company_summary,
        first_name=first_name,
        last_name=last_name,
        job_title=job_title,
    )

    reasoning_effort = "low" if model.strip() == "gpt-5" else None

    parsed, raw_text = await generate_structured_output(
        prompt=prompt,
        model=model.strip(),
        output_model=EmailOut,
        reasoning_effort=reasoning_effort,
    )

    return parsed, raw_text
