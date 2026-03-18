import asyncio
from pathlib import Path

import streamlit as st
from openai import APIConnectionError, APIError, APITimeoutError, RateLimitError

from services.email_generator import (
    DEFAULT_PROMPT_PATH,
    build_email,
    list_prompt_files,
    load_prompt_file,
)
from services.summarizer import summarize_company_from_text

TRANSIENT_ERRORS = (APIConnectionError, APITimeoutError, RateLimitError, APIError)


async def run_summary(website_text: str):
    return await summarize_company_from_text(
        website_text=website_text,
    )


async def run_email(
    company_summary: str,
    prompt_template: str,
    first_name: str,
    last_name: str,
    job_title: str,
    model: str,
):
    return await build_email(
        company_summary=company_summary,
        prompt_template=prompt_template,
        first_name=first_name,
        last_name=last_name,
        job_title=job_title,
        model=model,
    )


def init_state():
    defaults = {
        "website_text": "",
        "company_summary": "",
        "email_subject": "",
        "email_body": "",
        "selected_prompt_name": "",
        "prompt_editor_text": "",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def load_selected_prompt_into_editor(prompt_files: list[Path], prompt_name: str):
    selected = next((p for p in prompt_files if p.name == prompt_name), None)
    if selected is None:
        raise ValueError(f"Prompt not found: {prompt_name}")

    st.session_state.selected_prompt_name = prompt_name
    st.session_state.prompt_editor_text = load_prompt_file(selected)


st.set_page_config(
    page_title="Summary + Email Generator",
    layout="wide",
)

init_state()

st.title("Summary + Email Generator")

prompt_files = list_prompt_files()
prompt_names = [p.name for p in prompt_files]

if prompt_files:
    if not st.session_state.selected_prompt_name:
        default_name = (
            DEFAULT_PROMPT_PATH.name
            if DEFAULT_PROMPT_PATH.name in prompt_names
            else prompt_names[0]
        )
        st.session_state.selected_prompt_name = default_name

    if not st.session_state.prompt_editor_text.strip():
        load_selected_prompt_into_editor(
            prompt_files=prompt_files,
            prompt_name=st.session_state.selected_prompt_name,
        )

# =========================
# Top: Company Summary
# =========================
st.subheader("Company Summary")

website_text = st.text_area(
    "Website Text",
    value=st.session_state.website_text,
    height=220,
    placeholder="Paste website text here...",
)

summary_action_col1, summary_action_col2 = st.columns([1, 3])

with summary_action_col1:
    if st.button("Generate Summary", use_container_width=True):
        st.session_state.website_text = website_text
        st.session_state.email_subject = ""
        st.session_state.email_body = ""

        if not website_text.strip():
            st.error("Please paste website text first.")
        else:
            try:
                with st.spinner("Generating summary..."):
                    result = asyncio.run(
                        run_summary(
                            website_text=website_text,
                        )
                    )

                st.session_state.company_summary = result.company_summary or ""

                if st.session_state.company_summary:
                    st.success("Summary generated.")
                else:
                    st.warning("The model returned an empty summary.")

            except ValueError as exc:
                st.error(str(exc))
            except TRANSIENT_ERRORS as exc:
                st.error(f"Temporary API error: {type(exc).__name__}: {exc}")
            except Exception as exc:
                st.error(f"Error: {type(exc).__name__}: {exc}")

st.text_area(
    "Generated Company Summary",
    key="company_summary",
    height=240,
    placeholder="Generated company summary will appear here...",
)

st.divider()

# =========================
# Bottom: Left Prompt / Right Email
# =========================
left, right = st.columns(2, gap="large")

with left:
    st.subheader("Prompt")

    if prompt_files:
        current_index = prompt_names.index(st.session_state.selected_prompt_name)

        selected_prompt_name = st.selectbox(
            "Choose Existing Prompt",
            options=prompt_names,
            index=current_index,
        )

        prompt_btn_col1, prompt_btn_col2 = st.columns(2)

        with prompt_btn_col1:
            if st.button("Load Selected Prompt", use_container_width=True):
                try:
                    load_selected_prompt_into_editor(
                        prompt_files=prompt_files,
                        prompt_name=selected_prompt_name,
                    )
                    st.success(f"Loaded prompt: {selected_prompt_name}")
                except Exception as exc:
                    st.error(f"Failed to load prompt: {type(exc).__name__}: {exc}")

        with prompt_btn_col2:
            if st.button("Clear Prompt", use_container_width=True):
                st.session_state.prompt_editor_text = ""

    else:
        st.info(
            "No prompt files found in prompts/. You can write a prompt manually below."
        )

    st.text_area(
        "Prompt Editor",
        key="prompt_editor_text",
        height=420,
        placeholder=(
            "Write or edit your prompt here...\n\n"
            "Available placeholders:\n"
            "- {first_name}\n"
            "- {last_name}\n"
            "- {company_summary}\n"
            "- {job_title}"
        ),
    )

with right:
    st.subheader("Generate Email")

    name_col1, name_col2 = st.columns(2)

    with name_col1:
        first_name = st.text_input("First Name", value="Johann")

    with name_col2:
        last_name = st.text_input("Last Name", value="Min")

    job_title = st.text_input(
        "Job Title",
        value="CFO",
        placeholder="Enter any job title...",
    )

    email_model = st.radio(
        "Email Model",
        options=["gpt-5.1", "gpt-5"],
        index=0,
        help="gpt-5.1 is faster. gpt-5 thinks a bit more.",
    )

    if email_model == "gpt-5.1":
        st.caption("gpt-5.1: faster")
    else:
        st.caption("gpt-5: slightly slower, with low thinking effort enabled")

    if st.button(
        "Generate Email",
        use_container_width=True,
        disabled=not bool(st.session_state.company_summary.strip()),
    ):
        try:
            if not first_name.strip():
                st.error("First Name is required.")
            elif not last_name.strip():
                st.error("Last Name is required.")
            elif not job_title.strip():
                st.error("Job Title is required.")
            elif not st.session_state.prompt_editor_text.strip():
                st.error("Prompt cannot be empty.")
            elif not st.session_state.company_summary.strip():
                st.error("Company Summary cannot be empty.")
            else:
                with st.spinner("Generating email..."):
                    parsed, _raw = asyncio.run(
                        run_email(
                            company_summary=st.session_state.company_summary,
                            prompt_template=st.session_state.prompt_editor_text,
                            first_name=first_name,
                            last_name=last_name,
                            job_title=job_title,
                            model=email_model,
                        )
                    )

                if parsed is None:
                    st.error("The model did not return valid structured output.")
                else:
                    st.session_state.email_subject = parsed.subject or ""
                    st.session_state.email_body = parsed.email_body or ""
                    st.success("Email generated.")

        except ValueError as exc:
            st.error(str(exc))
        except TRANSIENT_ERRORS as exc:
            st.error(f"Temporary API error: {type(exc).__name__}: {exc}")
        except Exception as exc:
            st.error(f"Error: {type(exc).__name__}: {exc}")

    st.text_input(
        "Subject",
        key="email_subject",
    )

    st.text_area(
        "Email Body",
        key="email_body",
        height=360,
    )
