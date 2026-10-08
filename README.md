# Prompt Engineering Lab - Experiment 13

## RAG Chain with LCEL

Builds a full Retrieval-Augmented Generation pipeline: retrieve relevant
chunks, format a grounded prompt, generate an answer with an LLM.

## Pipeline Stages

| Stage | Component | Input | Output |
|-------|-----------|-------|--------|
| Index | RecursiveCharacterTextSplitter + NVIDIAEmbeddings + FAISS | Sample text | Vector store |
| Retrieve | FAISS retriever (k=2) | Question | 2 Document objects |
| Format | Custom join function | Documents | Single context string |
| Prompt | ChatPromptTemplate | context + question | Formatted messages |
| Generate | ChatNVIDIA (openai/gpt-oss-20b) | Messages | AIMessage |
| Parse | StrOutputParser | AIMessage | Plain string |

## The LCEL Chain

    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | parser

The dictionary at the front creates two branches from one input:

- **context branch** - runs the retriever on the question, then joins
  the retrieved documents into a single string
- **question branch** - passes the original question through unchanged

Both feed the prompt template, which then goes to the LLM.

## Model and Parameters

- Embedding model: nvidia/nemotron-3-embed-1b
- LLM: openai/gpt-oss-20b
- Temperature: 0.2
- Retrieval: k=2

## Why NVIDIA, not OpenAI

This environment uses NVIDIA NIM. We use:

- ChatNVIDIA from langchain-nvidia-ai-endpoints
- NVIDIAEmbeddings from the same package

Both are official LangChain integrations and work with the same
NVIDIA_API_KEY already in .env.

## Setup

Reuse the environment from Experiment 1:

    Copy-Item ..\EXP_1\.env .
    python -m pip install -r requirements.txt

## Run

    python rag_chain.py

## Expected Output

For each of three test questions:

- The question
- The generated answer (grounded in the retrieved chunks)
- The retrieved chunks (for transparency)

Followed by a summary of RAG chain concepts.

## Key Findings

1. **Retrieval grounds the answer** - the LLM answers from the retrieved
   chunks, not from general knowledge. This reduces hallucination.

2. **Prompt design matters** - instructing the model to reply
   "I don't know" when the answer is not in the context prevents
   fabrication.

3. **Top-k is a tradeoff** - too few chunks may miss relevant info;
   too many dilute the prompt with off-topic content.

4. **Quality depends on both halves** - good retrieval without good
   generation gives ungrounded answers; good generation without good
   retrieval gives confident mistakes.

## Troubleshooting

### Error code: 410 - Gone

The embedding or LLM model was retired. Check the live list:

    curl -s -H "Authorization: Bearer $env:NVIDIA_API_KEY" https://integrate.api.nvidia.com/v1/models

Look for entries containing "embed" (for embeddings) or "instruct"
(for generation), and swap the model ID.

### Answer is "I don't know" for every question

Retrieval is failing - the top-k chunks don't contain the answer. Try:

- Increasing k to 3 or 4
- Reducing chunk size to 200 chars
- Ensuring the sample text actually contains the information

### ImportError: langchain_nvidia_ai_endpoints

    python -m pip install langchain-nvidia-ai-endpoints

## Security

- Never commit .env
- Never paste API keys in chat, logs, or screenshots

## License

For educational / lab use only.
