
import os
from pypdf import PdfReader
import chromadb

# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(
    BASE_DIR, "documents", "Solid_Waste_Management_Rules_2026.pdf"
)
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

# English section: PDF pages 94 to 144
START_PAGE = 78
END_PAGE = 144

SOURCE_NAME = "Solid Waste Management Rules, 2026"

def split_into_chunks(text, chunk_size=1100, overlap=150):
    text = " ".join(text.split())
    chunks = []

    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks


def main():
    if not os.path.exists(PDF_PATH):
        print("PDF not found:", PDF_PATH)
        return

    print("Reading PDF...")
    reader = PdfReader(PDF_PATH)

    all_chunks = []
    all_ids = []
    all_metadata = []

    for page_number in range(START_PAGE, END_PAGE + 1):
        page_text = reader.pages[page_number - 1].extract_text() or ""

        if not page_text.strip():
            continue

        chunks = split_into_chunks(page_text)

        for chunk_number, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"page_{page_number}_chunk_{chunk_number}")
            all_metadata.append({
                "source": SOURCE_NAME,
                "pdf_page": page_number
            })

    if not all_chunks:
        print("No text was extracted. Please check the PDF.")
        return

    print("Extracted chunks:", len(all_chunks))
    print("Creating ChromaDB knowledge base...")

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    collection = client.get_or_create_collection(
        name="solid_waste_rules_2026"
    )

    # Remove older copies of this source before rebuilding
    existing = collection.get(where={"source": SOURCE_NAME})
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    collection.add(
        ids=all_ids,
        documents=all_chunks,
        metadatas=all_metadata
    )

    print("\nKnowledge base created successfully!")
    print("Total chunks stored:", len(all_chunks))
    print("Source:", SOURCE_NAME)
    print("PDF pages used:", START_PAGE, "to", END_PAGE)


if __name__ == "__main__":
    main()
