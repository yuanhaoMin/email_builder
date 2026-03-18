from services.llm_client import generate_structured_output
from services.models import CompanyBriefOut, SummaryResult


def clean_website_text(text: str) -> str:
    if not text:
        return ""

    lines = [line.strip() for line in text.splitlines()]
    cleaned = []

    for line in lines:
        if not line:
            continue
        if len(line) <= 2:
            continue
        cleaned.append(line)

    return "\n".join(cleaned).strip()


async def summarize_company_from_text(website_text: str) -> SummaryResult:
    cleaned_text = clean_website_text(website_text or "")

    if len(cleaned_text) < 120:
        raise ValueError(
            "The pasted website text is too short. Please provide more content."
        )

    snippet = cleaned_text[:6000]

    prompt = f"""
    Du erhältst kopierten Website-Text einer Firma.

    Aufgabe:
    Schreibe eine präzise, neutrale Unternehmenszusammenfassung auf Deutsch in 6-8 Sätzen.

    Wichtig:
    - Konzentriere dich auf Geschäftstätigkeit, Produkte, Dienstleistungen, Zielkunden, Branchen, Technologien und Positionierung.
    - Verwende konkrete Business-Begriffe nur dann, wenn sie im Text erkennbar sind.
    - Keine Spekulationen.
    - Keine Erwähnung von Cookie-Bannern, Datenschutz-Hinweisen, Navigationselementen oder irrelevanten Website-Fragmenten.
    - Keine Sicherheitslücken, Angriffsszenarien oder Missbrauchshinweise ableiten.

    Website-Text:
    {snippet}
    """.strip()

    parsed, _raw = await generate_structured_output(
        prompt=prompt,
        model="gpt-5-mini",
        output_model=CompanyBriefOut,
    )

    company_summary = ""
    if parsed:
        company_summary = (parsed.company_summary or "").strip()

    return SummaryResult(
        website_text=cleaned_text,
        company_summary=company_summary,
    )
