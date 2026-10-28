# AI Customer Ticket Response System

A local-first customer support ticket application. Agents can create, review, update, and deactivate tickets. A local scikit-learn classifier suggests a short solution, semantic search retrieves relevant local help articles, and a locally running Ollama model can draft a customer response for review.

## Architecture and request flow

1. The browser UI in `static/` submits requests to the HTTP entry point in `server.py`.
2. `server.py` routes ticket and dynamic-field API requests, while `database/db.py` opens SQLite connections and creates the existing schema.
3. `services/field_service.py` validates dynamic values. `services/ml_service.py` trains the local TF-IDF plus Logistic Regression classifier from `data/training_tickets.json`.
4. For ticket creation, the API predicts a solution and combines it with the ticket subject and description to build a retrieval query.
5. `rag.py` embeds the local Markdown documents and query with `sentence-transformers/all-MiniLM-L6-v2`. FAISS inner-product search over normalized embeddings returns cosine similarity rankings. The top two results are filtered by the configured score threshold and relative cutoff.
6. When local Ollama is available, the app sends the ticket, predicted solution, and retrieved article text to its loopback HTTP endpoint. The prompt tells the model to ground claims in the articles and avoid promising unverified actions or outcomes.
7. The resulting response and ticket are saved in SQLite. Agents can review and edit the response in the ticket detail view.

The model, vector index, and document mapping are cached in the running process. The index is rebuilt when the knowledge-base Markdown files change. The knowledge base is read from `knowledge_base/*.md`.

## Requirements and installation

- Windows 10/11 and Python 3.10 or newer.
- Network access when installing Python packages.
- Disk space for the Python packages, including PyTorch and FAISS.
- Ollama and the `llama3.2:3b` model for live LLM response generation. Ollama is optional for starting the app; if it is unavailable, the app gives a safe support-contact fallback.

The Python standard library provides the HTTP server, SQLite access, JSON handling, and other basic utilities. Third-party packages listed in `requirements.txt` provide NumPy, scikit-learn, FAISS CPU, and Sentence Transformers. Sentence Transformers also installs its required runtime packages, including PyTorch.

### Start with the Windows helper

From PowerShell in the project folder:

```powershell
.\run.ps1
```

The helper creates `.venv` if needed, installs `requirements.txt`, and starts `server.py`. Package installation may take time and use substantial disk space. It does **not** pull either AI model.

### Manual Python setup

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe server.py
```

Open <http://127.0.0.1:8000>. Keep the terminal open while using the app; press Ctrl+C to stop it.

### Embedding model setup

The embedding model is `sentence-transformers/all-MiniLM-L6-v2`. On the first retrieval, Sentence Transformers may download its model files from Hugging Face if they are not already cached. This is model setup, not hosted inference. It requires network access and additional storage. To download it deliberately before using the app:

```powershell
.\.venv\Scripts\python.exe -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
```

After the model is cached, embedding and retrieval run on the local machine. The app does not call a hosted embedding API.

### Local Ollama setup

Install Ollama separately, then download the model explicitly:

```powershell
ollama pull llama3.2:3b
```

Ensure Ollama is running locally, then start the app. The defaults are `OLLAMA_MODEL=llama3.2:3b` and `OLLAMA_URL=http://127.0.0.1:11434/api/generate`. You can set these before startup if needed:

```powershell
$env:OLLAMA_MODEL = 'llama3.2:3b'
$env:OLLAMA_URL = 'http://127.0.0.1:11434/api/generate'
.\.venv\Scripts\python.exe server.py
```

The server accepts only loopback HTTP URLs for Ollama. It does not send ticket content to a cloud LLM. If Ollama is unavailable, the ticket is still saved and the response explains that a detailed response could not be generated. If retrieval fails, the API reports a separate retrieval fallback; if retrieval finds no matching article, it recommends contacting support.

## Database and dynamic fields

`database/db.py` creates `data/tickets.db` and its existing `tickets` and `field_definitions` tables when the app starts. `server.py` keeps the startup behavior and environment configuration used by `run.ps1`. Ticket rows store the predicted solution, response, status, timestamps, and dynamic field values as JSON. New field definitions do not require a new ticket-table column. The database is local and excluded by `.gitignore`; do not submit customer data.

The field API supports text, dropdown, number, date, and checkbox definitions. The frontend loads definitions from `GET /api/fields`, renders them dynamically, and includes values in ticket create and update requests. The backend validates required values, dropdown options, finite numbers, ISO dates, and checkbox booleans.

## Dataset and known limitations

`data/training_tickets.json` contains 12 sample subject/description/solution examples. `Predictor` fits a local `TfidfVectorizer` and multiclass `LogisticRegression` classifier when a `Predictor` is created. Each sample currently has a distinct solution label, with one example per label. The current tests check representative outputs and fallbacks, but there is no held-out accuracy evaluation; do not claim a measured accuracy.

The knowledge base contains four short documents for accounts, application troubleshooting, billing, and orders. Retrieval indexes whole Markdown files rather than smaller passages. The score threshold and relative cutoff are small-project settings, not values validated on a large evaluation set. The model can recommend the wrong example solution, so generated responses must remain constrained by retrieved content and reviewed by an agent.

Live Ollama generation must be tested with Ollama installed, running, and the model pulled. Automated tests mock the Ollama response and separately test the unavailable-service fallback; a mock is not evidence of live model output quality.

## Tests and retrieval diagnostic

Run the automated suite:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Print unfiltered semantic scores for all knowledge-base files for four example questions:

```powershell
.\.venv\Scripts\python.exe check_scores.py
```

Expected top-ranked sources are `app_and_files.md` for application crashes, `accounts.md` for password recovery, `billing.md` for duplicate payment, and `orders.md` for delayed delivery. The diagnostic loads the embedding model once per run and shows scores before production filtering.

## Project files

- `server.py` - application entry point, HTTP handler, ticket and field API routes, and startup configuration.
- `database/db.py` - SQLite path, connection setup, and initialization of the existing schema and default fields.
- `services/ml_service.py` - local TF-IDF and Logistic Regression solution predictor.
- `services/field_service.py` - validation for text, dropdown, number, date, and checkbox values.
- `rag.py` - local Markdown loading, embeddings, FAISS retrieval, semantic scoring, and Ollama response generation.
- `static/index.html`, `static/app.js`, `static/styles.css` - browser interface.
- `data/training_tickets.json` - sample labeled tickets and solutions.
- `knowledge_base/*.md` - local customer-support guidance.
- `tests/test_system.py` - automated API, retrieval, and validation tests.
- `test_rag_llm.py` - local retrieval and Ollama smoke script.
- `check_scores.py` - unfiltered retrieval diagnostic.
- `run.ps1` - Windows environment setup and startup helper.

## Local API routes

- `GET /api/tickets` — list active tickets.
- `GET /api/tickets/{id}` — read a ticket.
- `POST /api/tickets` — validate, predict, retrieve, generate/fallback, and save a ticket.
- `PUT /api/tickets/{id}` — update ticket fields, dynamic values, status, or response.
- `DELETE /api/tickets/{id}` — deactivate a ticket.
- `GET /api/fields` and `POST /api/fields` — list and add dynamic field definitions.
