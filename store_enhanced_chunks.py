"""
Store Enhanced Chunks in ChromaDB with Hybrid Search

Uses the enhanced chunker output and stores in ChromaDB
with Azure OpenAI embeddings + BM25 for hybrid search.
"""

import os
import json
import re
from collections import defaultdict
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma

# BM25 for keyword search
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    print("⚠️  rank_bm25 not installed. Run: pip install rank-bm25")

load_dotenv()

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION - Change these for different documents
# ═══════════════════════════════════════════════════════════════════════════════
DOC_ID = "DHL"                          # Document identifier (used in ChromaDB)
ENHANCED_CHUNKS_PATH = "ENHANCED_CHUNKS.json"  # Input: From enhanced_chunker.py
CHROMA_PERSIST_DIR = "./chroma_db_enhanced"
COLLECTION_NAME = "enhanced_ocr_documents"


def get_embeddings():
    """Get Azure OpenAI embeddings using dedicated embedding credentials."""
    # Use dedicated embedding credentials (separate from chat model)
    azure_api_key = os.getenv("OPENAI_EMBEDDING_API_KEY")
    azure_resource = os.getenv("OPENAI_EMBEDDING_RESOURCE")
    azure_api_version = os.getenv("OPENAI_EMBEDDING_VERSION", "2024-02-01")
    azure_embedding_deployment = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
    
    if azure_api_key and azure_resource:
        azure_endpoint = f"https://{azure_resource}.openai.azure.com/"
        return AzureOpenAIEmbeddings(
            azure_deployment=azure_embedding_deployment,
            azure_endpoint=azure_endpoint,
            api_key=azure_api_key.strip(),
            api_version=azure_api_version,
        )
    raise ValueError("Azure OpenAI Embedding credentials not found. Check OPENAI_EMBEDDING_* vars in .env")


def load_enhanced_chunks(file_path: str) -> list:
    """Load enhanced chunks from JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["chunks"]


def store_chunks(chunks: list, doc_id: str = "TAX_INVOICE"):
    """Store chunks in ChromaDB."""
    print(f"\n{'='*60}")
    print("🗄️  STORING ENHANCED CHUNKS IN CHROMADB")
    print(f"{'='*60}")
    print(f"Total chunks: {len(chunks)}")
    
    # Prepare data
    ids = []
    texts = []
    metadatas = []
    
    for i, chunk in enumerate(chunks):
        ids.append(f"{doc_id}_chunk_{i}")
        texts.append(chunk["text"])
        
        metadata = {
            "doc_id": doc_id,
            "chunk_index": i,
            "content_type": chunk["metadata"].get("content_type", "text"),
            "page_number": chunk["metadata"].get("page_number") or 0,
            "section": chunk["metadata"].get("section") or "",
        }
        
        # Add table-specific metadata
        if "table_index" in chunk["metadata"]:
            metadata["table_index"] = chunk["metadata"]["table_index"]
        if "row_index" in chunk["metadata"]:
            metadata["row_index"] = chunk["metadata"]["row_index"]
        if "headers" in chunk["metadata"]:
            metadata["headers"] = ", ".join(chunk["metadata"]["headers"])
        
        metadatas.append(metadata)
    
    # Get embeddings
    print("\n🔗 Initializing Azure OpenAI Embeddings...")
    embeddings = get_embeddings()
    
    # Create ChromaDB
    print("🗃️  Creating ChromaDB collection...")
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    
    # Clear existing
    try:
        existing = vectorstore.get(where={"doc_id": doc_id})
        if existing and existing['ids']:
            print(f"🗑️  Deleting {len(existing['ids'])} existing chunks...")
            vectorstore.delete(ids=existing['ids'])
    except:
        pass
    
    # Add chunks
    print(f"📤 Adding {len(texts)} chunks...")
    vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=ids)
    
    print(f"\n✅ Successfully stored {len(texts)} chunks!")
    return vectorstore


class HybridSearcher:
    """
    Hybrid Search: Combines BM25 (keyword) + Semantic (embedding) search.
    
    - BM25: Good for exact keyword matching
    - Semantic: Good for meaning/context
    - Combined: Best of both worlds
    """
    
    def __init__(self, chunks: list, vectorstore):
        self.chunks = chunks
        self.vectorstore = vectorstore
        self.texts = [c["text"] for c in chunks]
        
        # Build text-to-index map for reliable chunk matching
        self.text_to_idx = {text: i for i, text in enumerate(self.texts)}
        
        # Build BM25 index
        if BM25_AVAILABLE:
            tokenized = [self._tokenize(text) for text in self.texts]
            self.bm25 = BM25Okapi(tokenized)
            print("✅ BM25 index built for hybrid search")
        else:
            self.bm25 = None
    
    def _tokenize(self, text: str) -> list:
        """Simple tokenizer for BM25."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.split()
    
    def search(self, query: str, k: int = 10, alpha: float = 0.5) -> list:
        """
        Hybrid search combining BM25 and semantic scores.
        
        Args:
            query: Search query
            k: Number of results to return
            alpha: Weight for semantic (1-alpha for BM25)
                   0.5 = equal weight
                   0.7 = more semantic
                   0.3 = more keyword
        
        Returns:
            List of (doc, combined_score, bm25_score, semantic_score)
        """
        # Get semantic results (fetch more for reranking)
        semantic_results = self.vectorstore.similarity_search_with_score(query, k=min(k*3, len(self.texts)))
        
        # Build semantic scores dict
        semantic_scores = {}
        for doc, score in semantic_results:
            # ChromaDB returns distance (lower is better), convert to similarity
            similarity = 1 / (1 + score)  # Bounded transform: distance to similarity (safe for any distance)
            # Use text content to find index (fixes chunk_index mismatch bug)
            chunk_idx = self.text_to_idx.get(doc.page_content, -1)
            if chunk_idx >= 0:
                semantic_scores[chunk_idx] = similarity
        
        # Get BM25 scores
        bm25_scores = {}
        if self.bm25:
            tokenized_query = self._tokenize(query)
            scores = self.bm25.get_scores(tokenized_query)
            
            # Normalize BM25 scores to 0-1 range
            max_score = max(scores) if max(scores) > 0 else 1
            for i, score in enumerate(scores):
                bm25_scores[i] = score / max_score
        
        # Combine scores
        combined = {}
        all_indices = set(semantic_scores.keys()) | set(bm25_scores.keys())
        
        for idx in all_indices:
            sem_score = semantic_scores.get(idx, 0)
            bm25_score = bm25_scores.get(idx, 0)
            
            # Weighted combination
            combined_score = (alpha * sem_score) + ((1 - alpha) * bm25_score)
            combined[idx] = (combined_score, bm25_score, sem_score)
        
        # Sort by combined score (descending)
        sorted_results = sorted(combined.items(), key=lambda x: x[1][0], reverse=True)[:k]
        
        # Build result list
        results = []
        for idx, (combined_score, bm25_score, sem_score) in sorted_results:
            if 0 <= idx < len(self.chunks):
                chunk = self.chunks[idx]
                results.append({
                    "text": chunk["text"],
                    "metadata": chunk["metadata"],
                    "combined_score": combined_score,
                    "bm25_score": bm25_score,
                    "semantic_score": sem_score,
                })
        
        return results


def query_and_display(vectorstore, query: str, k: int = 10):
    """Query using basic semantic search and display results."""
    print(f"\n{'='*60}")
    print(f"🔍 Query: '{query}' (Semantic Only, k={k})")
    print(f"{'='*60}")
    
    results = vectorstore.similarity_search_with_score(query=query, k=k)
    
    for i, (doc, score) in enumerate(results, 1):
        metadata = doc.metadata
        print(f"\n[Result {i}] (Distance: {score:.4f})")
        print(f"📍 Type: {metadata.get('content_type')} | Page: {metadata.get('page_number')} | Section: {metadata.get('section')}")
        
        # Show content preview
        content = doc.page_content
        # Remove header for cleaner display
        if content.startswith("[Source:"):
            content = content.split("]\n\n", 1)[-1] if "]\n\n" in content else content
        
        preview = content[:200] + "..." if len(content) > 200 else content
        print(f"📄 Content: {preview}")


def hybrid_query_and_display(searcher: HybridSearcher, query: str, k: int = 10, alpha: float = 0.5):
    """Query using hybrid search (BM25 + Semantic) and display results."""
    print(f"\n{'='*60}")
    print(f"🔍 Query: '{query}' (HYBRID: α={alpha}, k={k})")
    print(f"{'='*60}")
    
    results = searcher.search(query, k=k, alpha=alpha)
    
    for i, result in enumerate(results, 1):
        metadata = result["metadata"]
        print(f"\n[Result {i}] Combined: {result['combined_score']:.3f} | BM25: {result['bm25_score']:.3f} | Semantic: {result['semantic_score']:.3f}")
        print(f"📍 Type: {metadata.get('content_type')} | Page: {metadata.get('page_number')} | Section: {metadata.get('section')}")
        
        # Show content preview
        content = result["text"]
        if content.startswith("[Source:"):
            content = content.split("]\n\n", 1)[-1] if "]\n\n" in content else content
        
        preview = content[:250] + "..." if len(content) > 250 else content
        print(f"📄 Content: {preview}")


def main():
    print(f"\n{'='*60}")
    print("🗄️  STORE ENHANCED CHUNKS")
    print(f"{'='*60}")
    print(f"   Document ID:  {DOC_ID}")
    print(f"   Input:        {ENHANCED_CHUNKS_PATH}")
    print(f"   ChromaDB:     {CHROMA_PERSIST_DIR}")
    print(f"{'='*60}")
    
    # Load chunks
    print("\n📂 Loading enhanced chunks...")
    chunks = load_enhanced_chunks(ENHANCED_CHUNKS_PATH)
    print(f"   Loaded {len(chunks)} chunks")
    
    # Store in ChromaDB with document ID
    vectorstore = store_chunks(chunks, doc_id=DOC_ID)
    
    print(f"\n{'='*60}")
    print("✅ Storage complete! Run 'python rag_chat.py' to query.")
    print(f"{'='*60}\n")


def interactive_mode():
    """Interactive query mode for testing retrieval."""
    print("📂 Loading enhanced chunks...")
    chunks = load_enhanced_chunks(ENHANCED_CHUNKS_PATH)
    print(f"   Loaded {len(chunks)} chunks")
    
    # Load existing vectorstore
    print("🔗 Connecting to ChromaDB...")
    embeddings = get_embeddings()
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    
    # Initialize hybrid searcher
    print("🔧 Initializing Hybrid Search...")
    hybrid_searcher = HybridSearcher(chunks, vectorstore)
    
    print(f"\n{'='*60}")
    print("🎯 INTERACTIVE HYBRID SEARCH MODE")
    print("='*60")
    print("Commands:")
    print("  q       - Quit")
    print("  k=N     - Set number of results (e.g., k=5)")
    print("  a=N     - Set alpha weight (e.g., a=0.7 for more semantic)")
    print("  sem     - Switch to semantic-only mode")
    print("  hyb     - Switch to hybrid mode (default)")
    print(f"{'='*60}")
    
    k = 10
    alpha = 0.5
    mode = "hybrid"
    
    while True:
        try:
            query = input(f"\n🔍 [{mode.upper()}] Enter query: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        
        if not query:
            continue
        
        # Commands
        if query.lower() == 'q':
            print("👋 Goodbye!")
            break
        elif query.lower().startswith('k='):
            try:
                k = int(query.split('=')[1])
                print(f"✅ Set k={k}")
            except:
                print("❌ Invalid k value")
            continue
        elif query.lower().startswith('a='):
            try:
                alpha = float(query.split('=')[1])
                print(f"✅ Set alpha={alpha}")
            except:
                print("❌ Invalid alpha value")
            continue
        elif query.lower() == 'sem':
            mode = "semantic"
            print("✅ Switched to SEMANTIC-only mode")
            continue
        elif query.lower() == 'hyb':
            mode = "hybrid"
            print("✅ Switched to HYBRID mode")
            continue
        
        # Execute query
        if mode == "hybrid":
            hybrid_query_and_display(hybrid_searcher, query, k=k, alpha=alpha)
        else:
            query_and_display(vectorstore, query, k=k)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_mode()
    else:
        main()

