# Product reviews and user's sentiments analyze automation

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
        response + structured metadata (what llm provider used, confidence level and extracted tags)
```

## Quick Start

### Prerequisites

- Docker and Docker Compose
- OpenAI API key (optional, if no key provided - local LLM will be used)

### Running with Docker Compose

1. Start the services:
   ```bash
   docker-compose up -d
   ```

2. The API UI will be available at `http://127.0.0.1:8000/docs`