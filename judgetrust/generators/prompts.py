"""Answer-generator prompts. Task text comes from config (prompt A vs B)."""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

GENERATOR_PREFACE = """This is EVAL TOOLING, not medical practice. Produce an answer that
will be used only as an evaluation input. Do not give personalized medical advice
or dosing. Stick to general, well-known safety facts. If a clinician should be
involved, say so plainly.
"""

GENERATOR_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "{preface}\n\n{task_prompt}"),
        ("human", "{question}"),
    ]
)
