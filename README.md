# AI Engineering

Notebooks and small projects from an AI engineering course, covering prompting, conversation
management, tool calling, RAG, and agentic loops using the OpenAI API.

## AI Engineering Part 1

| File | What it covers |
|---|---|
| `hello-world.ipynb` | First OpenAI API call |
| `system-vs-user-prompt.ipynb` | System vs. user prompt roles |
| `conversation-history.ipynb` | Managing multi-turn conversation history |
| `caching.ipynb` | Prompt caching |
| `my-post-generator.ipynb` | LLM-based content generator |
| `ai-wars.ipynb` | Two LLMs conversing with each other |
| `gradio.ipynb` | Building a chat UI with Gradio |
| `tool-calling.ipynb` | OpenAI function/tool calling (dice roll + Pushover notification tools) |
| `RAG.ipynb` | Chunking, embeddings, t-SNE/KMeans visualization, and ChromaDB retrieval over multiple Netflix documents |
| `digital-twin-arch1-dynamic-context.ipynb` | Digital twin agent, v1: dynamic context injection |
| `digital-twin-arch2-basic-tool-calling.ipynb` | Digital twin agent, v2: basic tool calling |
| `digital-twin-arch3-proper-tool-calling.ipynb` | Digital twin agent, v3: proper tool-calling loop |
| `digital-twin-arch4-full-RAG.ipynb` | Digital twin agent, v4: full RAG pipeline |
| `digital-twin.ipynb` | Digital twin agent combining RAG retrieval and tool calling, served through Gradio |
| `digital-twin/` | Deployable Gradio Space version of the digital twin (`app.py`, `requirements.txt`) |
| `my-first-space/` | Minimal Gradio Space starter app |

## AI Engineering Part 2

| File | What it covers |
|---|---|
| `dice-roll-agent.ipynb` | A simple agentic loop: the model decides each turn whether to roll again or stop, via plain-text parsing rather than structured tool calls |

## Setup

```bash
python -m venv ai_env
source ai_env/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the repo root:

```
OPENAI_API_KEY=...
PUSHOVER_USER=...
PUSHOVER_TOKEN=...
```

### Private content

My personal text and the system prompt are not in this repo. They live in `private_documents.py`, which is gitignored. To run your own twin, create that file next to `app.py` and define these strings:

```python
document_overview = """..."""
document_education = """..."""
document_professional_experience = """..."""
document_skills = """..."""
document_projects = """..."""
document_interests = """..."""
system_message = """..."""
```

## Run

```bash
python app.py
```

The app opens in your browser on port 7860 (set `PORT` to change it).

## Files

| File | Purpose |
|---|---|
| `app.py` | Chunking, embedding, retrieval, tools and the Gradio app |
| `private_documents.py` | Personal documents and system prompt (gitignored, not included) |
| `requirements.txt` | Python dependencies |

`.env`, `private_documents.py` and generated `chroma-db*` folders are gitignored and never committed.
