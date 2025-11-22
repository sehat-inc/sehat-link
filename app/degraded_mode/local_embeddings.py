"""
Local Embeddings using ONNX Runtime
Works completely offline
"""

import numpy as np
from typing import List, Optional
import pickle
from pathlib import Path
from core.logging import get_logger

logger = get_logger("MAKING EMBEDDINGS")

class LocalEmbeddings:
    """Local embedding model using ONNX for offline semantic search"""
    
    def __init__(self, model_path: str = None, tokenizer_path: str = None):
        """Initialize ONNX embeddings"""
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
            
            base_dir = Path(__file__).parent
            model_path = model_path or str(base_dir / "bge-small.onnx")
            tokenizer_path = tokenizer_path or str(base_dir / "tokenizer.json")
            
            self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            self.tokenizer = Tokenizer.from_file(tokenizer_path)
            self.dimension = 384
        except ImportError:
            raise RuntimeError("Install: pip install onnxruntime tokenizers")
    
    def embed_text(self, text: str) -> np.ndarray:
        """Embed single text"""
        encoded = self.tokenizer.encode(text)
        ids = encoded.ids
        seq_len = len(ids)
        
        input_ids = np.array([ids], dtype=np.int64)
        attention_mask = np.ones((1, seq_len), dtype=np.int64)
        token_type_ids = np.zeros((1, seq_len), dtype=np.int64)
        
        outputs = self.session.run(None, {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        })
        
        return outputs[0][0]
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed multiple texts"""
        embeddings = [self.embed_text(t) for t in texts]
        return np.vstack(embeddings)


class FAISSVectorStore:
    """Local vector store using FAISS"""
    
    def __init__(self, dimension: int = 384):
        try:
            import faiss
            self.index = faiss.IndexFlatL2(dimension)
            self.texts = []
            self.metadata = []
        except ImportError:
            raise RuntimeError(
                "faiss not installed. "
                "Install: pip install faiss-cpu"
            )
    
    def add(self, embeddings: np.ndarray, texts: List[str], metadata: List[dict]):
        """Add vectors to index"""
        import faiss
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)
        
        self.index.add(embeddings.astype('float32'))
        self.texts.extend(texts)
        self.metadata.extend(metadata)
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[dict]:
        """Search for similar vectors"""
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.texts):
                results.append({
                    "text": self.texts[idx],
                    "metadata": self.metadata[idx],
                    "score": float(dist)
                })
        
        return results
    
    def save(self, path: str):
        """Save index to disk"""
        import faiss
        faiss.write_index(self.index, f"{path}.index")
        with open(f"{path}.meta", 'wb') as f:
            pickle.dump({"texts": self.texts, "metadata": self.metadata}, f)
    
    def load(self, path: str):
        """Load index from disk"""
        import faiss
        self.index = faiss.read_index(f"{path}.index")
        with open(f"{path}.meta", 'rb') as f:
            data = pickle.load(f)
            self.texts = data["texts"]
            self.metadata = data["metadata"]


def build_knowledge_index(knowledge_base, save_path: Optional[str] = None):
    """
    Build FAISS index from knowledge base.
    Run once, then load from disk.
    """
    embedder = LocalEmbeddings()
    vector_store = FAISSVectorStore()
    
    all_texts = []
    all_metadata = []
    
    # Index medical data
    data = knowledge_base.get_all_data()
    
    for symptom in data.get("medical", {}).get("symptoms", []):
        text = f"{symptom['name']}: {symptom.get('description', '')}"
        all_texts.append(text)
        all_metadata.append({"type": "symptom", "data": symptom})
    
    for condition in data.get("medical", {}).get("conditions", []):
        text = f"{condition['name']}: {condition.get('description', '')}"
        all_texts.append(text)
        all_metadata.append({"type": "condition", "data": condition})
    
    for aid in data.get("medical", {}).get("first_aid", []):
        text = f"{aid['name']}: {aid.get('description', '')}"
        all_texts.append(text)
        all_metadata.append({"type": "first_aid", "data": aid})
    
    for med in data.get("medical", {}).get("medications", []):
        text = f"{med['name']}: {med.get('uses', '')}"
        all_texts.append(text)
        all_metadata.append({"type": "medication", "data": med})
    
    for program in data.get("programs", {}).get("programs", []):
        text = f"{program['name']}: {program.get('description', '')}"
        all_texts.append(text)
        all_metadata.append({"type": "program", "data": program})
    
    for city, hospitals in data.get("hospitals", {}).get("cities", {}).items():
        for hospital in hospitals:
            text = f"{hospital['name']} in {city}: {', '.join(hospital.get('specialties', []))}"
            all_texts.append(text)
            all_metadata.append({"type": "hospital", "data": hospital})
    
    # Embed all at once
    if all_texts:
        embeddings = embedder.embed_batch(all_texts)
        vector_store.add(embeddings, all_texts, all_metadata)
    logger.info("Index built successfully")
    logger.info(f"Index size: {len(all_texts)}")
    logger.info(f"Index dimension: {vector_store.index.d}")

    if save_path:
        vector_store.save(save_path)
    
    return embedder, vector_store
