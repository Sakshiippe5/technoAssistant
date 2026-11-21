
import logging

from threading import Lock

from fastapi import FastAPI, HTTPException
from groq import APIError

from backend.schemas import (
    TroubleshootRequest,
    TroubleshootResponse
)

from backend.rag.pipeline import run_pipeline

from backend.database.vector_store import (
    get_client,
    COLLECTION_NAME
)


logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

pipeline_lock = Lock()


app = FastAPI(
    title="technoAssistant",
    description="AI-powered IT troubleshooting assistant",
    version="1.0.0"
)


@app.get("/")
def root():

    return {
        "name": "technoAssistant",
        "message": "AI Troubleshooting Assistant API",
        "version": "1.0.0"
    }


@app.get("/health")
def health_check():

    try:

        with pipeline_lock:

            client = get_client()

            if not client.collection_exists(
                COLLECTION_NAME
            ):
                raise HTTPException(
                    status_code=503,
                    detail="Knowledge base is not indexed."
                )

            count = client.count(
                collection_name=COLLECTION_NAME,
                exact=True
            ).count

            if count == 0:

                raise HTTPException(
                    status_code=503,
                    detail="Knowledge base is empty."
                )

        return {
            "status": "healthy",
            "knowledge_base": "available",
            "indexed_documents": count
        }

    except HTTPException:
        raise

    except Exception:

        logger.exception("Health check failed")

        raise HTTPException(
            status_code=503,
            detail="Knowledge base unavailable."
        )


@app.post(
    "/api/troubleshoot",
    response_model=TroubleshootResponse
)
def troubleshoot(request: TroubleshootRequest):

    query = request.query.strip()

    if not query:

        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty."
        )

    try:

        with pipeline_lock:

            result = run_pipeline(query)

        return result

    except ValueError:

        logger.exception("Pipeline validation failed")

        raise HTTPException(
            status_code=422,
            detail="Unable to process the query."
        )

    except APIError:

        logger.exception("Groq API request failed")

        raise HTTPException(
            status_code=502,
            detail="LLM service is temporarily unavailable."
        )

    except Exception:

        logger.exception("Pipeline execution failed")

        raise HTTPException(
            status_code=500,
            detail="An internal server error occurred."
        )
