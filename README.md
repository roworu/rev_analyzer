# Review Analyzer Automation

A FastAPI-based service that implements a hybrid inference pipeline for review classification, combining lightweight domain models with external LLM calls.

## Project Structure

```
client (curl / web UI)
  └─> FastAPI service
      └─>
        Hybrid inference pipeline
          ├─> lightweight domain model (review classifier)
          ├─> external LLM call (if domain model score is lower that provided in request)
          └─> hybrid logic combiner (threshold + review message length threshold)
        ↓
        response + structured metadata (what llm provider used, confidence level)
```

## Quick Start

### Prerequisites

- Docker and Docker Compose
- OpenAI API key or Ollama instance

### Environment Setup

Fill in ENV options to docker-compose.yml file, or to src/.env/your_env_name.py
If you decided to state ENV options as .py file, select is as config like that: `CONFIG=your_env_name`

Recomennded setup is to have 2 config files, one for dev layout and one for production environment.

### Running with Docker Compose

1. Start the services:
   ```bash
   docker-compose up -d
   ```

2. The API will be available at `http://localhost:8000` or `http://127.0.0.1:8000/docs`