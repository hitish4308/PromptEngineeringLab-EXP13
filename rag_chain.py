"""
Experiment 13 - RAG Chain with LCEL

Builds a full RAG chain:
  1. Index a sample text (reuses EXP_12 approach)
  2. Retrieve top-k chunks for a question
  3. Format a prompt with context + question
  4. Generate an answer grounded in the retrieved text

Uses NVIDIA NIM for both embeddings and generation.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------- Setup ----------
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

api_key = os.environ.get("NVIDIA_API_KEY")
if not api_key:
    raise SystemExit("NVIDIA_API_KEY not found. Create a .env file.")

EMBED_MODEL = "nvidia/nemotron-3-embed-1b"
LLM_MODEL = "openai/gpt-oss-20b"


# ---------- Sample Text ----------
SAMPLE_TEXT = """
Climate change refers to long-term shifts in temperatures and weather
patterns. These shifts may be natural, but since the 1800s, human activities
have been the main driver of climate change, primarily due to the burning of
fossil fuels like coal, oil, and gas, which produces heat-trapping gases.

Rising global temperatures cause sea levels to rise due to melting polar ice
caps and thermal expansion of seawater. Coastal cities face flooding risks
and saltwater intrusion into freshwater supplies.

More frequent and intense storms, droughts, and heatwaves are also linked to
climate change. These extremes stress agriculture, damage infrastructure,
and displace communities, with vulnerable populations hit hardest.

Scientists agree that reducing greenhouse gas emissions is essential.
Renewable energy, energy efficiency, and sustainable land use are key
mitigation strategies.

Deforestation is another major contributor to climate change. Trees absorb
carbon dioxide, and when forests are cleared, that stored carbon is released
back into the atmosphere, accelerating the greenhouse effect.

Ocean acidification, caused by absorbed carbon dioxide, threatens marine life
including coral reefs and shellfish, disrupting entire food chains.
"""


# ---------- Test Questions ----------
QUESTIONS = [
    "What causes climate change?",
    "What are the effects of climate change?",
    "How can climate change be mitigated?",
]


# ---------- Build the pipeline ----------
def build_retriever():
    """Chunk text, embed it, and return a retriever."""
    print("Building vector store...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=30,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    documents = splitter.create_documents([SAMPLE_TEXT])
    print(f"  Chunks: {len(documents)}")

    embeddings = NVIDIAEmbeddings(model=EMBED_MODEL, model_type="passage")
    vectorstore = FAISS.from_documents(documents, embeddings)
    print(f"  FAISS store ready.")

    return vectorstore.as_retriever(search_kwargs={"k": 2})


def format_docs(docs):
    """Join retrieved chunks into a single context string."""
    return "\n\n".join(d.page_content.strip() for d in docs)


# ---------- Main ----------
if __name__ == "__main__":
    print("=" * 72)
    print("  EXPERIMENT 13 - RAG CHAIN (LCEL)")
    print("=" * 72)
    print(f"Embedding model: {EMBED_MODEL}")
    print(f"LLM            : {LLM_MODEL}")
    print()

    retriever = build_retriever()
    print()

    # LLM
    llm = ChatNVIDIA(model=LLM_MODEL, temperature=0.2, max_tokens=300)
    parser = StrOutputParser()

    # RAG prompt
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant. Use only the provided context to "
         "answer the question. If the answer is not in the context, "
         "reply exactly: \"I don't know.\""),
        ("user",
         "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"),
    ])

    # LCEL RAG chain
    # The dict at the front maps the input question to two sub-computations:
    #   - retriever runs on the question -> returns chunks
    #   - RunnablePassthrough forwards the question unchanged
    # format_docs turns the chunk list into a single string.
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | rag_prompt
        | llm
        | parser
    )

    # ---------- Run questions ----------
    for i, question in enumerate(QUESTIONS, start=1):
        print("=" * 72)
        print(f"  QUESTION {i}: {question}")
        print("=" * 72)

        # Retrieve chunks separately so we can show them
        retrieved = retriever.invoke(question)

        # Generate answer
        try:
            answer = rag_chain.invoke(question)
        except Exception as e:
            answer = f"[ERROR] {type(e).__name__}: {e}"

        print(f"\nANSWER:\n  {answer}\n")
        print(f"RETRIEVED CHUNKS ({len(retrieved)}):")
        for j, doc in enumerate(retrieved, start=1):
            snippet = doc.page_content.strip().replace("\n", " ")
            print(f"  [{j}] {snippet[:180]}...")
        print()

    # ---------- Summary ----------
    print("=" * 72)
    print("  SUMMARY - RAG CHAIN CONCEPTS")
    print("=" * 72)
    print("""
The LCEL RAG chain:

    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | parser

How it works:

  1. The dict at the front creates two parallel branches from one input:
       - "context" branch runs the retriever on the question, then
         flattens the retrieved documents to a single string
       - "question" branch passes the original question through unchanged

  2. Both branches feed a formatted prompt.

  3. The LLM generates an answer using the retrieved context.

  4. The parser extracts the plain string.

Why this matters for RAG:

  - The answer is grounded in retrieved text, not in model memory
  - If the context does not contain the answer, the prompt directs
    the model to say "I don't know"
  - Adding more chunks (higher k) gives broader coverage but risks
    diluting relevance
  - Quality depends on both retrieval (chunking + embeddings) and
    generation (prompt + LLM)

This is the full RAG loop: retrieve, augment, generate.
""")
