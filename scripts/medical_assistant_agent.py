import os
import itertools
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv
from datasets import load_dataset
from langchain_community.document_loaders import DataFrameLoader
from langchain_text_splitters import CharacterTextSplitter


load_dotenv()

# ---------- CONFIG ----------
INDEX_NAME = "sehat-link-programs"
NAMESPACE = "medical-kb"
BATCH_SIZE = 10
EMBED_MODEL = "text-embedding-3-large"
# ----------------------------

# initialize clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(INDEX_NAME)

def load_and_chunk_data():
    print("Loading dataset from Hugging Face...")
    data = load_dataset("keivalya/MedQuad-MedicalQnADataset", split='train')
    data_df = data.to_pandas()
    data_df = data_df.head(100) # Optional: limit size

    print("Loading data into LangChain Documents...")
    loader = DataFrameLoader(data_df, page_content_column="Answer")
    documents = loader.load()

    print("Chunking documents...")
    text_splitter = CharacterTextSplitter(
        chunk_size=1250,
        separator="\n",
        chunk_overlap=100
    )
    split_docs = text_splitter.split_documents(documents)
    return split_docs

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

# ---------- MAIN LOGIC ----------
print("📖 Starting data loading and chunking...")
split_documents = load_and_chunk_data()
print(f"✅ Found and split into {len(split_documents)} chunks.")

for i, batch in enumerate(chunks(split_documents, BATCH_SIZE)):
    texts = [doc.page_content for doc in batch]
    ids = [f"med-chunk-{i}-{j}" for j in range(len(batch))]
    embeddings = embed_texts(texts)

    vectors = []
    for id_, emb, doc in zip(ids, embeddings, batch):
        meta = doc.metadata.copy()
        meta["text"] = doc.page_content
        vectors.append({
            "id": id_,
            "values": emb,
            "metadata": meta
        })

    print(f"⬆️  Upserting batch {i+1} ({len(vectors)} vectors)...")
    index.upsert(vectors=vectors, namespace=NAMESPACE)

print("✅ All chunks embedded and upserted successfully!")