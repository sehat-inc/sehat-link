import os
import itertools
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv
from pypdf import PdfReader

load_dotenv()

PDF_FILE = "data/Empanelled_Hospital_List.pdf"
INDEX_NAME = "eligibilty-agent-index"     
NAMESPACE = "eligibility-namespace"        
BATCH_SIZE = 100                        
EMBED_MODEL = "text-embedding-3-large"

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(INDEX_NAME)

def extract_pdf_pages(pdf_path):
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            pages.append({
                "page": i + 1,
                "text": text.strip()
            })
    return pages

def chunks(iterable, batch_size):
    it = iter(iterable)
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))

def embed_texts(texts):
    response = openai_client.embeddings.create(
        input=texts,
        model=EMBED_MODEL,
        dimensions=2048
    )
    return [d.embedding for d in response.data]

print("📖 Reading PDF and extracting pages...")
pages = extract_pdf_pages(PDF_FILE)
print(f"✅ Found {len(pages)} pages.")

for i, batch in enumerate(chunks(pages, BATCH_SIZE)):
    texts = [p["text"] for p in batch]
    ids = [f"page-{p['page']}" for p in batch]
    embeddings = embed_texts(texts)

    vectors = [
        {
            "id": id_,
            "values": emb,
            "metadata": {
                "page_number": p["page"],
                "program": "Sehat Sahulat Program",
                "type": "Sehat Sahulat Location Info",
                "text": p["text"]
            }
        }
        for id_, emb, p in zip(ids, embeddings, batch)
    ]

    print(f"⬆️  Upserting batch {i+1} ({len(vectors)} vectors)...")
    index.upsert(vectors=vectors, namespace=NAMESPACE)

print("✅ All pages embedded and upserted successfully!")