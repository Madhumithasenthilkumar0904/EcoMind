
import os
import chromadb

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "solid_waste_rules_2026"


def retrieve_relevant_context(query, n_results=4):
    """
    Retrieve relevant passages from the indexed
    Solid Waste Management Rules, 2026.
    """

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    try:
        collection = client.get_collection(
            name=COLLECTION_NAME
        )
    except Exception:
        return []

    count = collection.count()

    if count == 0:
        return []

    result = collection.query(
        query_texts=[query],
        n_results=min(n_results, count)
    )

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]

    retrieved = []

    for document, metadata in zip(documents, metadatas):
        retrieved.append({
            "text": document,
            "source": metadata.get(
                "source",
                "Solid Waste Management Rules, 2026"
            ),
            "page": metadata.get("pdf_page", "Unknown")
        })

    return retrieved
