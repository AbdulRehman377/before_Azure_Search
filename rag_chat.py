"""
RAG Chat - Retrieval Augmented Generation with Azure OpenAI

Beautiful terminal interface for querying documents using:
- Hybrid Search (BM25 + Semantic)
- Azure OpenAI GPT for answer generation
"""

import os
import sys
import json
import re
from dotenv import load_dotenv
from openai import AzureOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma

# BM25 for keyword search
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

load_dotenv()

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

ENHANCED_CHUNKS_PATH = "ENHANCED_CHUNKS.json"
CHROMA_PERSIST_DIR = "./chroma_db_enhanced"
COLLECTION_NAME = "enhanced_ocr_documents"

# Search settings
DEFAULT_K = 30  # Number of chunks to retrieve (balanced for quality)
DEFAULT_ALPHA = 0.7  # 0.8 = more weight on semantic, 0.2 = more weight on BM25
DISPLAY_ALL_SOURCES = False  # Show all retrieved sources (not just top 3)

# Output settings
RETRIEVED_CHUNKS_PATH = "RETRIEVED_CHUNKS.json"  # Stores latest retrieved chunks (overwritten each query)

# ═══════════════════════════════════════════════════════════════════════════════
# TERMINAL COLORS & STYLING
# ═══════════════════════════════════════════════════════════════════════════════

class Colors:
    """ANSI color codes for beautiful terminal output."""
    # Basic colors
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    
    # Foreground colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    
    # Bright foreground
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"
    
    # Background colors
    BG_BLACK = "\033[40m"
    BG_BLUE = "\033[44m"
    BG_CYAN = "\033[46m"
    BG_WHITE = "\033[47m"


def print_banner():
    """Print beautiful ASCII banner."""
    banner = f"""
{Colors.BRIGHT_CYAN}{Colors.BOLD}
    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║                                                                           ║
    ║   ██████╗  █████╗  ██████╗      ██████╗██╗  ██╗ █████╗ ████████╗          ║
    ║   ██╔══██╗██╔══██╗██╔════╝     ██╔════╝██║  ██║██╔══██╗╚══██╔══╝          ║
    ║   ██████╔╝███████║██║  ███╗    ██║     ███████║███████║   ██║             ║
    ║   ██╔══██╗██╔══██║██║   ██║    ██║     ██╔══██║██╔══██║   ██║             ║
    ║   ██║  ██║██║  ██║╚██████╔╝    ╚██████╗██║  ██║██║  ██║   ██║             ║
    ║   ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝      ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝             ║
    ║                                                                           ║
    ║           {Colors.BRIGHT_YELLOW}Document Intelligence with Azure OpenAI{Colors.BRIGHT_CYAN}                       ║
    ║                                                                           ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
{Colors.RESET}"""
    print(banner)


def print_divider(char="─", length=80, color=Colors.BRIGHT_BLACK):
    """Print a divider line."""
    print(f"{color}{char * length}{Colors.RESET}")


def print_section(title, icon=""):
    """Print a section header."""
    print(f"\n{Colors.BRIGHT_CYAN}{Colors.BOLD}{icon} {title}{Colors.RESET}")
    print_divider()


def print_success(message):
    """Print success message."""
    print(f"{Colors.BRIGHT_GREEN}✓ {message}{Colors.RESET}")


def print_error(message):
    """Print error message."""
    print(f"{Colors.BRIGHT_RED}✗ {message}{Colors.RESET}")


def print_info(message):
    """Print info message."""
    print(f"{Colors.BRIGHT_BLUE}ℹ {message}{Colors.RESET}")


def print_warning(message):
    """Print warning message."""
    print(f"{Colors.BRIGHT_YELLOW}⚠ {message}{Colors.RESET}")


# ═══════════════════════════════════════════════════════════════════════════════
# AZURE OPENAI CLIENT
# ═══════════════════════════════════════════════════════════════════════════════

def get_llm_client():
    """Initialize Azure OpenAI client."""
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
    resource = os.getenv("OPENAI_RESOURCE", "").strip()
    api_version = os.getenv("OPENAI_API_VERSION", "2024-02-01").strip()
    
    if not api_key or not resource:
        raise ValueError("Missing Azure OpenAI credentials in .env file")
    
    endpoint = f"https://{resource}.openai.azure.com/"
    
    return AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=endpoint
    )


def get_embeddings():
    """Get Azure OpenAI embeddings using dedicated embedding credentials."""
    # Use dedicated embedding credentials (separate from chat model)
    api_key = os.getenv("OPENAI_EMBEDDING_API_KEY", "").strip()
    resource = os.getenv("OPENAI_EMBEDDING_RESOURCE", "").strip()
    api_version = os.getenv("OPENAI_EMBEDDING_VERSION", "2024-02-01").strip()
    embedding_deployment = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
    
    endpoint = f"https://{resource}.openai.azure.com/"
    
    return AzureOpenAIEmbeddings(
        azure_deployment=embedding_deployment,
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# HYBRID SEARCH
# ═══════════════════════════════════════════════════════════════════════════════

class HybridSearcher:
    """Hybrid Search combining BM25 (keyword) + Semantic (embedding) search."""
    
    def __init__(self, chunks: list, vectorstore):
        self.chunks = chunks
        self.vectorstore = vectorstore
        self.texts = [c["text"] for c in chunks]
        
        # Build text-to-index map for reliable chunk matching
        # This fixes the chunk_index mismatch bug when ENHANCED_CHUNKS.json lacks chunk_index
        self.text_to_idx = {text: i for i, text in enumerate(self.texts)}
        
        # Build BM25 index
        if BM25_AVAILABLE:
            tokenized = [self._tokenize(text) for text in self.texts]
            self.bm25 = BM25Okapi(tokenized)
        else:
            self.bm25 = None
    
    def _tokenize(self, text: str) -> list:
        """Simple tokenizer for BM25."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.split()
    
    def search(self, query: str, k: int = 8, alpha: float = 0.5) -> list:
        """
        Hybrid search combining BM25 and semantic scores.
        
        Args:
            query: Search query
            k: Number of results to return
            alpha: Weight for semantic (1-alpha for BM25)
        """
        # Get semantic results
        semantic_results = self.vectorstore.similarity_search_with_score(query, k=min(k*3, len(self.texts)))
        
        # Build semantic scores dict
        semantic_scores = {}
        for doc, score in semantic_results:
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
            max_score = max(scores) if max(scores) > 0 else 1
            for i, score in enumerate(scores):
                bm25_scores[i] = score / max_score
        
        # Combine scores
        combined = {}
        all_indices = set(semantic_scores.keys()) | set(bm25_scores.keys())
        
        for idx in all_indices:
            sem_score = semantic_scores.get(idx, 0)
            bm25_score = bm25_scores.get(idx, 0)
            combined_score = (alpha * sem_score) + ((1 - alpha) * bm25_score)
            combined[idx] = (combined_score, bm25_score, sem_score)
        
        # Sort by combined score
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


# ═══════════════════════════════════════════════════════════════════════════════
# RAG PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

class RAGChat:
    """RAG Chat with Azure OpenAI."""
    
    def __init__(self):
        self.llm_client = None
        self.searcher = None
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
        self.max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "4096"))
        self.temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.5"))
        self.k = DEFAULT_K
        self.alpha = DEFAULT_ALPHA
        self.show_sources = True
        self.conversation_history = []
    
    def initialize(self):
        """Initialize all components."""
        print_section("Initializing RAG System", "🚀")
        
        # Load LLM client
        print_info("Connecting to Azure OpenAI...")
        try:
            self.llm_client = get_llm_client()
            print_success(f"Connected to Azure OpenAI (Model: {self.model})")
        except Exception as e:
            print_error(f"Failed to connect to Azure OpenAI: {e}")
            return False
        
        # Load chunks
        print_info(f"Loading chunks from {ENHANCED_CHUNKS_PATH}...")
        try:
            with open(ENHANCED_CHUNKS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            chunks = data["chunks"]
            print_success(f"Loaded {len(chunks)} chunks")
        except Exception as e:
            print_error(f"Failed to load chunks: {e}")
            return False
        
        # Connect to ChromaDB
        print_info("Connecting to ChromaDB...")
        try:
            embeddings = get_embeddings()
            vectorstore = Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=embeddings,
                persist_directory=CHROMA_PERSIST_DIR,
            )
            print_success("Connected to ChromaDB")
        except Exception as e:
            print_error(f"Failed to connect to ChromaDB: {e}")
            return False
        
        # Initialize hybrid searcher
        print_info("Building hybrid search index...")
        self.searcher = HybridSearcher(chunks, vectorstore)
        if BM25_AVAILABLE:
            print_success("Hybrid search ready (BM25 + Semantic)")
        else:
            print_warning("BM25 not available, using semantic search only")
        
        return True
    
    def build_context(self, results: list) -> str:
        """Build context string from search results."""
        context_parts = []
        for i, result in enumerate(results, 1):
            text = result["text"]
            # Remove the header prefix for cleaner context
            if text.startswith("[Source:"):
                text = text.split("]\n\n", 1)[-1] if "]\n\n" in text else text
            context_parts.append(f"[Document {i}]\n{text}")
        return "\n\n---\n\n".join(context_parts)
    
    def generate_answer(self, query: str, context: str) -> str:
        """Generate answer using Azure OpenAI."""
        system_prompt = """You are a helpful document assistant. Answer questions based on the provided document context.

INSTRUCTIONS:
- Answer based ONLY on the provided context
- If the answer is not in the context, say "I couldn't find this information in the documents"
- Be concise but complete
- If there are specific values, numbers, or names, quote them exactly
- Format your response clearly with line breaks for readability"""

        user_message = f"""CONTEXT FROM DOCUMENTS:
{context}

---

QUESTION: {query}

Please provide a clear and accurate answer based on the above context."""

        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {e}"
    
    def query(self, user_query: str) -> dict:
        """Process a query through the RAG pipeline."""
        # Step 1: Retrieve relevant chunks
        results = self.searcher.search(user_query, k=self.k, alpha=self.alpha)
        
        # Step 2: Build context
        context = self.build_context(results)
        
        # Step 3: Generate answer
        answer = self.generate_answer(user_query, context)
        
        # Step 4: Save retrieved chunks to file (overwrite)
        self.save_retrieved_chunks(user_query, results)
        
        return {
            "query": user_query,
            "answer": answer,
            "sources": results,
            "num_sources": len(results)
        }
    
    def save_retrieved_chunks(self, query: str, results: list):
        """Save retrieved chunks to JSON file (overwrites each time)."""
        output_data = {
            "query": query,
            "timestamp": __import__('datetime').datetime.now().isoformat(),
            "num_chunks": len(results),
            "settings": {
                "k": self.k,
                "alpha": self.alpha
            },
            "chunks": [
                {
                    "rank": i + 1,
                    "text": r["text"],
                    "metadata": r["metadata"],
                    "scores": {
                        "combined": round(r["combined_score"], 4),
                        "bm25": round(r["bm25_score"], 4),
                        "semantic": round(r["semantic_score"], 4)
                    }
                }
                for i, r in enumerate(results)
            ]
        }
        
        with open(RETRIEVED_CHUNKS_PATH, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"{Colors.DIM}  📁 Chunks saved to {RETRIEVED_CHUNKS_PATH}{Colors.RESET}")
    
    def display_answer(self, result: dict):
        """Display the answer beautifully."""
        print(f"\n{Colors.BRIGHT_GREEN}{Colors.BOLD}{'─' * 80}{Colors.RESET}")
        print(f"{Colors.BRIGHT_GREEN}{Colors.BOLD}  💬 ANSWER{Colors.RESET}")
        print(f"{Colors.BRIGHT_GREEN}{Colors.BOLD}{'─' * 80}{Colors.RESET}\n")
        
        # Print answer with word wrapping
        answer = result["answer"]
        print(f"{Colors.WHITE}{answer}{Colors.RESET}")
        
        # Show sources if enabled
        if self.show_sources and result["sources"]:
            print(f"\n{Colors.BRIGHT_YELLOW}{Colors.BOLD}{'─' * 80}{Colors.RESET}")
            print(f"{Colors.BRIGHT_YELLOW}{Colors.BOLD}  📚 SOURCES ({result['num_sources']} documents retrieved){Colors.RESET}")
            print(f"{Colors.BRIGHT_YELLOW}{Colors.BOLD}{'─' * 80}{Colors.RESET}\n")
            
            # Show all sources or limit based on DISPLAY_ALL_SOURCES setting
            sources_to_show = result["sources"] if DISPLAY_ALL_SOURCES else result["sources"][:3]
            
            for i, source in enumerate(sources_to_show, 1):
                metadata = source["metadata"]
                content_type = metadata.get("content_type", "text")
                page = metadata.get("page_number", "?")
                section = metadata.get("section", "")
                
                # Score display
                scores = f"Score: {source['combined_score']:.2f}"
                if BM25_AVAILABLE:
                    scores += f" (BM25: {source['bm25_score']:.2f}, Semantic: {source['semantic_score']:.2f})"
                
                print(f"{Colors.BRIGHT_CYAN}  [{i}] {content_type.upper()} | Page {page} | {section}{Colors.RESET}")
                print(f"{Colors.DIM}      {scores}{Colors.RESET}")
                
                # Preview content
                text = source["text"]
                if text.startswith("[Source:"):
                    text = text.split("]\n\n", 1)[-1] if "]\n\n" in text else text
                preview = text[:150].replace('\n', ' ') + "..." if len(text) > 150 else text.replace('\n', ' ')
                print(f"{Colors.BRIGHT_BLACK}      {preview}{Colors.RESET}\n")
    
    def display_help(self):
        """Display help information."""
        help_text = f"""
{Colors.BRIGHT_CYAN}{Colors.BOLD}╔══════════════════════════════════════════════════════════════════════════════╗
║                              AVAILABLE COMMANDS                               ║
╠══════════════════════════════════════════════════════════════════════════════╣{Colors.RESET}
{Colors.BRIGHT_WHITE}║  {Colors.BRIGHT_YELLOW}quit, exit, q{Colors.BRIGHT_WHITE}      │  Exit the application                              ║
║  {Colors.BRIGHT_YELLOW}help, h, ?{Colors.BRIGHT_WHITE}         │  Show this help message                            ║
║  {Colors.BRIGHT_YELLOW}clear, cls{Colors.BRIGHT_WHITE}         │  Clear the screen                                  ║
║  {Colors.BRIGHT_YELLOW}sources on/off{Colors.BRIGHT_WHITE}     │  Toggle showing source documents                   ║
║  {Colors.BRIGHT_YELLOW}k=<number>{Colors.BRIGHT_WHITE}         │  Set number of documents to retrieve (e.g., k=5)   ║
║  {Colors.BRIGHT_YELLOW}alpha=<number>{Colors.BRIGHT_WHITE}     │  Set search weight (0=BM25, 1=Semantic, 0.5=both)  ║
║  {Colors.BRIGHT_YELLOW}status{Colors.BRIGHT_WHITE}             │  Show current settings                             ║{Colors.RESET}
{Colors.BRIGHT_CYAN}{Colors.BOLD}╚══════════════════════════════════════════════════════════════════════════════╝{Colors.RESET}

{Colors.BRIGHT_MAGENTA}💡 TIP: Just type your question and press Enter to query the documents!{Colors.RESET}
"""
        print(help_text)
    
    def display_status(self):
        """Display current settings."""
        print(f"""
{Colors.BRIGHT_CYAN}╔══════════════════════════════════════════════════════════════════════════════╗
║                              CURRENT SETTINGS                                 ║
╠══════════════════════════════════════════════════════════════════════════════╣{Colors.RESET}
{Colors.WHITE}║  Model:              {Colors.BRIGHT_GREEN}{self.model:<40}{Colors.WHITE}          ║
║  Documents to retrieve (k):  {Colors.BRIGHT_GREEN}{self.k:<32}{Colors.WHITE}          ║
║  Search weight (alpha):      {Colors.BRIGHT_GREEN}{self.alpha:<32}{Colors.WHITE}          ║
║  Show sources:               {Colors.BRIGHT_GREEN}{str(self.show_sources):<32}{Colors.WHITE}          ║
║  Temperature:                {Colors.BRIGHT_GREEN}{self.temperature:<32}{Colors.WHITE}          ║{Colors.RESET}
{Colors.BRIGHT_CYAN}╚══════════════════════════════════════════════════════════════════════════════╝{Colors.RESET}
""")
    
    def run(self):
        """Run the interactive chat loop."""
        print_banner()
        
        if not self.initialize():
            print_error("Failed to initialize. Please check your configuration.")
            return
        
        print(f"\n{Colors.BRIGHT_GREEN}{'═' * 80}{Colors.RESET}")
        print(f"{Colors.BRIGHT_GREEN}{Colors.BOLD}  ✨ RAG Chat is ready! Type 'help' for commands or ask a question.{Colors.RESET}")
        print(f"{Colors.BRIGHT_GREEN}{'═' * 80}{Colors.RESET}\n")
        
        while True:
            try:
                # Get user input with styled prompt
                user_input = input(f"{Colors.BRIGHT_MAGENTA}{Colors.BOLD}  🔍 You: {Colors.RESET}").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                lower_input = user_input.lower()
                
                if lower_input in ['quit', 'exit', 'q']:
                    print(f"\n{Colors.BRIGHT_CYAN}👋 Goodbye! Thanks for using RAG Chat.{Colors.RESET}\n")
                    break
                
                elif lower_input in ['help', 'h', '?']:
                    self.display_help()
                    continue
                
                elif lower_input in ['clear', 'cls']:
                    os.system('clear' if os.name != 'nt' else 'cls')
                    print_banner()
                    continue
                
                elif lower_input == 'sources on':
                    self.show_sources = True
                    print_success("Source display enabled")
                    continue
                
                elif lower_input == 'sources off':
                    self.show_sources = False
                    print_success("Source display disabled")
                    continue
                
                elif lower_input.startswith('k='):
                    try:
                        self.k = int(lower_input.split('=')[1])
                        print_success(f"Now retrieving {self.k} documents per query")
                    except:
                        print_error("Invalid value. Use: k=5")
                    continue
                
                elif lower_input.startswith('alpha='):
                    try:
                        self.alpha = float(lower_input.split('=')[1])
                        if 0 <= self.alpha <= 1:
                            print_success(f"Search weight set to {self.alpha}")
                        else:
                            print_error("Alpha must be between 0 and 1")
                    except:
                        print_error("Invalid value. Use: alpha=0.5")
                    continue
                
                elif lower_input == 'status':
                    self.display_status()
                    continue
                
                # Process as query
                print(f"\n{Colors.DIM}  ⏳ Searching documents and generating answer...{Colors.RESET}")
                
                result = self.query(user_input)
                self.display_answer(result)
                
            except KeyboardInterrupt:
                print(f"\n\n{Colors.BRIGHT_CYAN}👋 Goodbye! Thanks for using RAG Chat.{Colors.RESET}\n")
                break
            except EOFError:
                break
            except Exception as e:
                print_error(f"An error occurred: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    chat = RAGChat()
    chat.run()
