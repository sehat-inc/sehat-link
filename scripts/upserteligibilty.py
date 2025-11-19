import os
import re
import itertools
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

# ---------- CONFIG ----------
MD_FILE = "pbm.md"                     
INDEX_NAME = "sehat-link-programs"     
NAMESPACE = "eligibility-agent"        
BATCH_SIZE = 10                        
EMBED_MODEL = "text-embedding-3-large" 
# ----------------------------
# Initialize Pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API"))
index = pc.Index(host="sehat-link-programs")

# initialize clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(INDEX_NAME)

# ---------- HELPER FUNCTIONS ----------

def chunk_by_headings(md_text):
    """Split Markdown into chunks starting at each '## ' heading until next one."""
    pattern = r"(^## .+?$)"
    lines = md_text.splitlines()
    chunks = []
    current_title = None
    buffer = []

    for line in lines:
        if re.match(pattern, line.strip()):
            if current_title and buffer:
                chunks.append({
                    "title": current_title.strip(),
                    "text": "\n".join(buffer).strip()
                })
                buffer = []
            current_title = line.strip()
        else:
            buffer.append(line)

    # last chunk
    if current_title and buffer:
        chunks.append({
            "title": current_title.strip(),
            "text": "\n".join(buffer).strip()
        })

    return chunks

def chunks(iterable, batch_size):
    """Yield successive batches."""
    it = iter(iterable)
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))


def embed_texts(texts):
    """Generate embeddings for a list of texts using OpenAI."""
    response = openai_client.embeddings.create(
        input=texts,
        model=EMBED_MODEL,
        dimensions=2048
    )
    return [d.embedding for d in response.data]


# ---------- MAIN LOGIC ----------

with open(MD_FILE, "r", encoding="utf-8") as f:
    md_text = f.read()

print("📖 Reading Markdown and chunking by headings...")
sections = chunk_by_headings(md_text)
print(f"✅ Found {len(sections)} chunks.")

# Generate embeddings and upsert in batches
for i, batch in enumerate(chunks(sections, BATCH_SIZE)):
    texts = [s["text"] for s in batch]
    ids = [f"chunk-{i}-{j}" for j in range(len(batch))]
    embeddings = embed_texts(texts)

    vectors = [
        {
            "id": id_,
            "values": emb,
            "metadata": {
                "title": s["title"],
                "text": s["text"]
            }
        }
        for id_, emb, s in zip(ids, embeddings, batch)
    ]

    print(f"⬆️  Upserting batch {i+1} ({len(vectors)} vectors)...")
    index.upsert(vectors=vectors, namespace=NAMESPACE)

print("✅ All chunks embedded and upserted successfully!")
