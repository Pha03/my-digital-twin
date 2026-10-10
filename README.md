# My Digital Twin

A chatbot that answers questions about me, in my voice. It is a retrieval-augmented (RAG) assistant
built for the SuperDataScience 6-week AI Engineering challenge. Ask it about my education,
experience, skills, projects, or interests.

## How it works

1. **Chunk:** my personal documents (overview, education, experience, skills, projects, interests)
   are split into overlapping chunks that end on natural boundaries (paragraph, line, sentence).
2. **Embed:** each chunk is embedded with OpenAI `text-embedding-3-small`.
3. **Store:** the vectors are saved in a local ChromaDB collection (`chroma_db_twin`). The
   collection is cleared and rebuilt each time the app starts.
4. **Retrieve:** for every question, the top 5 most similar chunks are fetched and added to the
   system prompt as context.
5. **Generate:** `gpt-4.1-mini` answers using that context and can call tools when needed.
6. **Serve:** the chat interface is built with Gradio.

### Tools

| Tool | What it does |
|---|---|
| `send_notification` | Sends me a Pushover push notification when someone wants to get in touch, or when the twin can't answer a question |
| `dice_roll` | Rolls a six-sided die |

## Project structure

| File | Purpose |
|---|---|
| `app.py` | Chunking, embedding, ChromaDB storage, retrieval, tools, and the Gradio app |
| `private_documents.py` | My personal documents and system prompt (gitignored, not included) |
| `requirements.txt` | Python dependencies |

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

`OPENAI_API_KEY` is required. The two Pushover values are optional; without them, notifications
are skipped.

### Private content

My personal text and the system prompt are not in this repo. They live in `private_documents.py`,
which is gitignored. To run your own twin, create that file next to `app.py` and define these strings:

```python
document_overview = """..."""
document_education = """..."""
document_professional_experience = """..."""
document_skills = """..."""
document_projects = """..."""
document_interests = """..."""
system_message = """..."""
```

Optionally, put an avatar image named `Pooja.JPG` next to `app.py` to show it in the chat.

## Run

```bash
python app.py
```

The app opens in your browser on port 7860 (set `PORT` to change it). Because the documents are
re-embedded at startup, restart the app after editing `private_documents.py`.

`.env`, `private_documents.py`, `ai_env/` and generated `chroma-db*` folders are gitignored and
never committed.
