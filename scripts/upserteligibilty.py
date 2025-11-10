import os
import re
import itertools
from dotenv import load_dotenv
from pinecone.grpc import PineconeGRPC as Pinecone

load_dotenv()

# Initialize Pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_KEY"))
index = pc.Index(host="sehat-link-programs")

def chunks(iterable, batch_size=100):
    """Break an iterable into chunks of size batch_size."""
    it = iter(iterable)
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))

# --- Load Markdown file ---
with open("pbm.md", "r", encoding="utf-8") as f:
    md_text = f.read()

# --- Extract top-level ## sections (with all content below until next ##) ---
# This regex finds each '## Heading' and captures everything up to the next '## ' or end of file
pattern = r"(##\s+[^\n]+)([\s\S]*?)(?=\n##\s+|$)"
matches = re.findall(pattern, md_text)

documents = []
for i, (heading, content) in enumerate(matches):
    heading_clean = heading.replace("##", "").strip()
    text_chunk = f"{heading}\n{content.strip()}"

    # Make ID slug from heading
    doc_id = re.sub(r'[^a-zA-Z0-9]+', '_', heading_clean.lower()).strip("_")

    documents.append({
        "id": f"{doc_id}_{i}",
        "values": [],  # Pinecone will embed internally
        "metadata": {
            "title": heading_clean,
            "text": text_chunk
        }
    })

# --- Upsert in batches ---
for batch in chunks(documents, batch_size=10):
    index.upsert(vectors=batch, namespace="eligibility-agent")

print(f"✅ Uploaded {len(documents)} sections to namespace 'eligibility-agent'")
