# Document Q&A Service

### Quick Start
```
# 1. install everything into a poetry virtual-env
poetry install

# 2. set the environment variables
cp .env.example .env # Replace the fake values with the real API keys

# 3. start the backend API (defaults to http://localhost:8000)
poetry run uvicorn backend_api.api.app:app

# 4. in another terminal, ask questions about a PDF
poetry run chatbot ~/Desktop/document.pdf
```

### Implementation details
[Backend Implementation](backend/README.md)
