# Document Q&A Service

### Quick Start

Prerequisites: [**Poetry**](https://python-poetry.org/docs/)

1. First, set the environment variables
    ```
    cp .env.example .env # Replace the OPENAI_API_KEY with the real key
    ```

2. Add `vertex_user.json` to the root of the project (Google API credentials)

3. Build and run the app:

    ```
    # 1. install everything into a poetry virtual-env
    poetry install

    # 2. start the backend API (defaults to http://localhost:8000)
    poetry run uvicorn backend_api.api.app:app

    # 3. in another terminal, ask questions about a PDF
    poetry run chatbot ~/Desktop/document.pdf
    ```

### Features
- Lightweight CLI tool with streaming support
- Event-driven backend REST API built with FastAPI
- Langchain for PDF processing and chunking
- ChromaDB for document vectorization and storage
- Gemini 2.5 Flash for doc-enhanced chat responses

### Codebase quality
The backend API is fully typed using Python type hints. When working on production-level code, typing can help make development faster, find bugs easier, and make it much easier to collaborate in a team environment (when different engineers are working on different pieces of the code). IMO untyped code should never be shipped to production unless there's extensive test coverage and every engineer is **fully familiar** with all code. 

### Approach
#### Document processing
- Preprocessing using simple regex string cleanup and normalizing all text as unicode characters
- Document chunking using LangChain's [RecursiveCharacterTextSplitter](https://python.langchain.com/api_reference/text_splitters/character/langchain_text_splitters.character.RecursiveCharacterTextSplitter.html)

#### RAG vs Large-Context Input
While RAG generally works well, it struggles on tasks where context from the entire document is required, e.g. "Summarize the document" (here is where analysis on the entire document could help). However, RAG excels for factual Q&A in structured documents.

I considered simply giving the entire document to a large-context model, such as Gemini 2.5 Pro / Flash as this would be an extremely simple solution to the problem. However, the cost and latency of large-context inference are much higher than embedding the document and retrieving relevant results. Therefore, I decided to move forward with the RAG implementation.

#### Embeddings and Vector Store
- For embeddings, I used OpenAI's embeddings model (text-embeddings-small) as it is heavily used across the industry. An alternative model would be the Gemini Vertex embeddings model.
- For the vector store, I used ChromaDB as it is open source and easy to run locally.

#### LLM Choice
I was debating between a few different models:
- Claude 3.7 Sonnet
- GPT 4o
- GPT o4-mini-high
- Gemini 2.5 Pro
- Gemini 2.5 Flash

However I decided to use Gemini 2.5 Flash as it offered some of the **quickest response times**, **cheapest cost**, and **large context window** (the large context window could be even more useful in the future if we decide to implement a hybrid approach, where both RAG and direct prompt context are used to provide better insights on data)

### CLI implementation
The CLI tool is just a small script that POSTs data to the backend for document upload, and listens for server-sent events in a loop to print to stdout (until the user closes the program or types `exit`)

### Backend implementation details
[Backend Implementation](backend/README.md)

### Future considerations
- Improve standardization of the API, expose shared types between backend APIs and frontend consumers
- Use `rich` plugin to print prettified output to the terminal, we should be able to print markdown as well.
- Implement hybrid approach of full context injection + RAG for more accurate results
- Handle multiple document types, PNGs, videos, etc (the model can already do this, but we would need to explore an embedding strategy if the content is too large)
- Run evals against multiple models to determine the most **accurate** model (as well as the fastest)
- Implement the AG-UI protocol for a standard event-driven interface: https://github.com/ag-ui-protocol/ag-ui
    - UI comes for free with CopilotKit: https://docs.copilotkit.ai/
- Include chat history as messages for a multi-turn conversation with context
