"""
rag.py — Retrieve relevant chunks + prompt GPT-4o with strict grounding
"""

from typing import List, Dict, Tuple

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


SYSTEM_PROMPT = """You are a precise supply-chain assistant for Meridian Components Pvt. Ltd.
Answer ONLY using the context provided below.
If the context does not contain the answer, reply exactly with this sentence:
"The information is not available in the uploaded documents."
Never invent numbers, clause numbers, policy rules, or supplier details.
When you use information, prefer quoting the key figure or clause.

Context:
{context}
"""


def format_docs(docs) -> str:
    """Format retrieved documents with clear source labels."""
    formatted = []
    for d in docs:
        source = d.metadata.get("source", "unknown")
        # PyPDFLoader pages are 0-indexed → show human page (1-based)
        page = d.metadata.get("page", 0) + 1
        formatted.append(
            f"[Source: {source} | Page: {page}]\n{d.page_content}"
        )
    return "\n\n---\n\n".join(formatted)


def ask_question(
    vectorstore,
    question: str,
    top_k: int = 6,
) -> Tuple[str, List[Dict]]:
    """
    Retrieve top_k chunks, send them + question to GPT-4o,
    and return (answer, list of unique sources).
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.1,          # low temperature = more factual
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = chain.invoke(question)

    # Collect sources for display
    retrieved_docs = retriever.invoke(question)
    sources = []
    seen = set()
    for d in retrieved_docs:
        source = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", 0) + 1
        key = (source, page)
        if key not in seen:
            seen.add(key)
            sources.append({"file": source, "page": page})

    return answer, sources
