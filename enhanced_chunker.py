"""
Enhanced Chunker for Azure Document Intelligence

Implements production-grade chunking strategy:
1. Dedicated table chunks with row-based formatting
2. Page-based splitting using <!-- PageBreak -->
3. Header prefixes for context
4. Deduplication via content hashing
5. Quality filters for noise removal
6. Spatial pairing for column layouts (SHIPPER/CONSIGNEE)
7. Pattern detection for inline labels

Based on production patterns from enterprise document processing.
"""

import json
import re
import hashlib
from typing import List, Dict, Optional, Tuple
from bs4 import BeautifulSoup
from dataclasses import dataclass, asdict

# Import spatial pairing module (COMMENTED OUT - uncomment to enable spatial pairing)
# from spatial_pairing import SpatialPairing, PatternDetector

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION - Change these for different documents
# ═══════════════════════════════════════════════════════════════════════════════
PDF_NAME = "invoice.pdf"                    # Source PDF name (for metadata)
RAW_OCR_PATH = "RAW_OCR.json"           # Input: Raw OCR from azure_ocr.py
ENHANCED_CHUNKS_PATH = "ENHANCED_CHUNKS.json"  # Output: Enhanced chunks

# Chunking mode
USE_ONE_SHOT = True  # True = single/page-based chunks, False = detailed section splitting


@dataclass
class EnhancedChunk:
    """Represents a processed chunk with metadata."""
    content: str
    content_type: str  # "table", "text", "figure"
    page_number: Optional[int]
    section: Optional[str]
    metadata: Dict


class EnhancedChunker:
    """
    Production-grade chunker for Azure DI markdown output.
    
    Features:
    - Dedicated table extraction with header context
    - Page-based splitting
    - Content deduplication
    - Noise filtering
    - Contextual headers
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.min_chunk_length = self.config.get("min_chunk_length", 50)
        self.max_chunk_length = self.config.get("max_chunk_length", 4000)
        self.content_hash_length = self.config.get("content_hash_length", 300)  # Increased for better dedup
        self.use_one_shot = self.config.get("use_one_shot", False)
        self.use_spatial_pairing = self.config.get("use_spatial_pairing", False)  # Disabled by default
        
        # Noise patterns to filter out
        self.noise_patterns = [
            r'^Page \d+ of \d+$',
            r'^:selected:$',
            r'^\s*$',
            r'^F\d{3,4}\s+\d+\s+\d+$',  # Footer codes like "F014 11 16"
        ]
        
        # Track seen content for deduplication
        self.seen_hashes = set()
        
        # Initialize spatial pairing and pattern detection (COMMENTED OUT)
        # Uncomment below to enable spatial pairing for column layouts
        # self.spatial_pairing = SpatialPairing(config={
        #     "column_threshold": 0.5,       # X tolerance for same column
        #     "vertical_threshold": 0.2,      # Y distance for label-value pairing
        #     "max_label_length": 25,         # Max chars for a label
        #     "max_label_words": 3,           # Max words for a label
        # })
        # self.pattern_detector = PatternDetector()
    
    def load_raw_ocr(self, file_path: str) -> Dict:
        """Load the raw Azure DI JSON output."""
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def extract_chunks(self, raw_ocr: Dict, filename: str = "document") -> List[EnhancedChunk]:
        """
        Main entry point: Extract enhanced chunks from Azure DI output.
        
        Args:
            raw_ocr: The raw JSON from Azure DI
            filename: Source filename for metadata
            
        Returns:
            List of EnhancedChunk objects
        """
        analyze_result = raw_ocr.get("analyzeResult", raw_ocr)
        
        content = analyze_result.get("content", "")
        content_format = analyze_result.get("contentFormat", "text")
        tables = analyze_result.get("tables", [])
        pages = analyze_result.get("pages", [])
        figures = analyze_result.get("figures", [])
        
        print(f"\n{'='*60}")
        print(f"📄 ENHANCED CHUNKER")
        print(f"{'='*60}")
        print(f"Content format: {content_format}")
        print(f"Content length: {len(content):,} chars")
        print(f"Tables: {len(tables)}")
        print(f"Pages: {len(pages)}")
        print(f"Figures: {len(figures)}")
        print(f"Spatial pairing: {'ENABLED' if self.use_spatial_pairing else 'DISABLED'}")
        
        chunks = []
        self.seen_hashes.clear()
        
        # === STEP 1: Extract dedicated table chunks ===
        print(f"\n📊 Extracting table chunks...")
        table_chunks = self._extract_table_chunks(tables, filename, pages)
        chunks.extend(table_chunks)
        print(f"   Created {len(table_chunks)} table chunks")
        
        # === STEP 2: Process remaining content with spatial pairing ===
        if self.use_one_shot:
            print(f"\n📝 ONE-SHOT MODE: Processing as single document...")
            text_chunks = self._process_one_shot(content, filename, pages)
        else:
            print(f"\n📝 DETAILED MODE: Splitting by pages and sections...")
            if self.use_spatial_pairing:
                print(f"   🔗 Applying spatial pairing for column layouts...")
                text_chunks = self._process_by_pages_with_spatial(raw_ocr, content, filename, pages)
            else:
                text_chunks = self._process_by_pages(content, filename, pages)
        
        chunks.extend(text_chunks)
        print(f"   Created {len(text_chunks)} text chunks")
        
        # === STEP 3: Extract figure chunks ===
        print(f"\n🖼️  Extracting figure chunks...")
        figure_chunks = self._extract_figure_chunks(figures, content, filename)
        chunks.extend(figure_chunks)
        print(f"   Created {len(figure_chunks)} figure chunks")
        
        # === STEP 4: Final deduplication and quality check ===
        print(f"\n🔍 Running quality filters...")
        final_chunks = self._filter_chunks(chunks)
        print(f"   Final chunks after filtering: {len(final_chunks)}")
        
        print(f"\n✅ Total chunks created: {len(final_chunks)}")
        
        return final_chunks
    
    def _is_kv_table(self, grid: List[List], col_count: int, row_count: int) -> bool:
        """
        Detect if table is a Key-Value table (2 columns with label-value pattern).
        
        KV tables look like:
        | Invoice No | F250335786 |
        | Date       | 16-Jun-25  |
        """
        if col_count != 2 or row_count < 2:
            return False
        
        # Check if left column looks like labels (short, text-like)
        label_like_count = 0
        non_empty_rows = 0  # Count only rows with non-empty first column
        
        for row in grid:
            if row[0]:
                non_empty_rows += 1
                left_cell = row[0].strip()
                # Label heuristics: short, mostly letters, may end with ':'
                if len(left_cell) < 40 and len(left_cell.split()) <= 5:
                    # Check if it looks like a label (more letters than numbers)
                    alpha_count = sum(1 for c in left_cell if c.isalpha())
                    if alpha_count > len(left_cell) * 0.3:
                        label_like_count += 1
        
        # If most NON-EMPTY rows have label-like left cells, it's a KV table
        # Fix: compare against non_empty_rows, not total row_count
        if non_empty_rows < 2:
            return False
        return label_like_count >= non_empty_rows * 0.6
    
    def _extract_table_chunks(self, tables: List[Dict], filename: str, pages: List[Dict]) -> List[EnhancedChunk]:
        """
        Extract dedicated chunks for each table with smart formatting.
        
        - Detects KV tables (2-col label-value) and formats as "Label: Value"
        - Only treats headers when DI explicitly marks them (kind == "columnHeader")
        - Creates row-centric chunks for better retrieval
        """
        chunks = []
        
        for table_idx, table in enumerate(tables):
            row_count = table.get("rowCount", 0)
            col_count = table.get("columnCount", 0)
            cells = table.get("cells", [])
            
            if not cells:
                continue
            
            # Get page number
            bounding_regions = table.get("boundingRegions", [])
            page_num = bounding_regions[0].get("pageNumber", 1) if bounding_regions else 1
            
            # Build grid and detect explicit headers
            grid = [[None for _ in range(col_count)] for _ in range(row_count)]
            headers = [None] * col_count
            has_explicit_headers = False
            
            for cell in cells:
                row_idx = cell.get("rowIndex", 0)
                col_idx = cell.get("columnIndex", 0)
                content = cell.get("content", "").strip()
                kind = cell.get("kind", "")
                
                if row_idx < row_count and col_idx < col_count:
                    grid[row_idx][col_idx] = content
                    
                    # FIX: Only treat as header if DI explicitly marks it
                    if kind == "columnHeader":
                        headers[col_idx] = content
                        has_explicit_headers = True
            
            # Detect if this is a KV table (2-column label-value)
            is_kv_table = self._is_kv_table(grid, col_count, row_count)
            
            # === KV TABLE MODE: Format as "Label: Value" pairs ===
            if is_kv_table:
                kv_pairs = []
                for row in grid:
                    label = (row[0] or "").strip().rstrip(':')
                    value = (row[1] or "").strip()
                    if label and value:
                        kv_pairs.append(f"{label}: {value}")
                
                if kv_pairs:
                    kv_content = "\n".join(kv_pairs)
                    header = self._build_header(
                        filename=filename,
                        section=f"Table {table_idx + 1} (Key-Value)",
                        page=page_num
                    )
                    chunks.append(EnhancedChunk(
                        content=header + kv_content,
                        content_type="table_kv",
                        page_number=page_num,
                        section=f"Table {table_idx + 1}",
                        metadata={
                            "table_index": table_idx,
                            "table_type": "key_value",
                            "row_count": row_count,
                            "column_count": col_count
                        }
                    ))
                continue  # Skip normal table processing for KV tables
            
            # === REGULAR TABLE: Create markdown chunk ===
            # If no explicit headers, use row 0 as headers only if it looks like headers
            if not has_explicit_headers and row_count > 1:
                # Check if row 0 looks like headers (short, distinct from data)
                row0_looks_like_header = all(
                    grid[0][i] and len(grid[0][i]) < 30 and not grid[0][i].replace(',', '').replace('.', '').isdigit()
                    for i in range(col_count) if grid[0][i]
                )
                if row0_looks_like_header:
                    for col_idx in range(col_count):
                        if grid[0][col_idx]:
                            headers[col_idx] = grid[0][col_idx]
                    has_explicit_headers = True
            
            table_markdown = self._table_to_markdown(grid, headers if has_explicit_headers else [None] * col_count)
            if table_markdown and len(table_markdown) > 20:
                header = self._build_header(
                    filename=filename,
                    section=f"Table {table_idx + 1}",
                    page=page_num
                )
                chunks.append(EnhancedChunk(
                    content=header + table_markdown,
                    content_type="table",
                    page_number=page_num,
                    section=f"Table {table_idx + 1}",
                    metadata={
                        "table_index": table_idx,
                        "row_count": row_count,
                        "column_count": col_count,
                        "headers": [h for h in headers if h]
                    }
                ))
            
            # === ROW-CENTRIC CHUNKS: DISABLED (matching production TS behavior) ===
            # Production only emits whole-table chunks, not per-row chunks
            # Uncomment below if you need row-level retrieval for large tables
            #
            # start_row = 1 if has_explicit_headers else 0
            # for row_idx in range(start_row, row_count):
            #     row_parts = []
            #     for col_idx in range(col_count):
            #         cell_value = grid[row_idx][col_idx]
            #         col_header = headers[col_idx] if has_explicit_headers else None
            #         if cell_value:
            #             if col_header and col_header != cell_value:
            #                 row_parts.append(f"{col_header}: {cell_value}")
            #             else:
            #                 row_parts.append(cell_value)
            #     if row_parts:
            #         row_content = " | ".join(row_parts)
            #         if len(row_content) > 20:
            #             header = self._build_header(filename=filename, section=f"Table {table_idx + 1}, Row {row_idx + 1}", page=page_num)
            #             chunks.append(EnhancedChunk(content=header + row_content, content_type="table_row", page_number=page_num, section=f"Table {table_idx + 1}", metadata={"table_index": table_idx, "row_index": row_idx, "headers": [h for h in headers if h]}))
        
        return chunks
    
    def _table_to_markdown(self, grid: List[List], headers: List) -> str:
        """Convert table grid to markdown format."""
        if not grid or not grid[0]:
            return ""
        
        lines = []
        
        # Header row
        header_row = " | ".join(h or "" for h in headers)
        lines.append(f"| {header_row} |")
        lines.append("|" + "|".join(["---"] * len(headers)) + "|")
        
        # Data rows
        for row in grid[1:]:
            row_content = " | ".join(cell or "" for cell in row)
            lines.append(f"| {row_content} |")
        
        return "\n".join(lines)
    
    def _process_one_shot(self, content: str, filename: str, pages: List[Dict]) -> List[EnhancedChunk]:
        """
        One-shot mode: Keep markdown as minimal chunks, split only by page breaks.
        """
        chunks = []
        
        # Remove table HTML blocks (already extracted separately)
        clean_content = self._remove_tables_from_content(content)
        
        if not clean_content or len(clean_content) < self.min_chunk_length:
            return chunks
        
        # For small documents, create single chunk
        if len(clean_content) <= self.max_chunk_length:
            header = self._build_header(
                filename=filename,
                section="Document",
                page=1
            )
            chunks.append(EnhancedChunk(
                content=header + clean_content,
                content_type="text",
                page_number=1,
                section="Document",
                metadata={"total_pages": len(pages)}
            ))
            return chunks
        
        # Split by page breaks
        parts = clean_content.split("<!-- PageBreak -->") if "<!-- PageBreak -->" in clean_content else [clean_content]
        
        for idx, part in enumerate(parts):
            trimmed = part.strip()
            if len(trimmed) >= self.min_chunk_length:
                header = self._build_header(
                    filename=filename,
                    section=f"Page {idx + 1}",
                    page=idx + 1
                )
                chunks.append(EnhancedChunk(
                    content=header + trimmed,
                    content_type="text",
                    page_number=idx + 1,
                    section=f"Page {idx + 1}",
                    metadata={}
                ))
        
        return chunks
    
    def _process_by_pages(self, content: str, filename: str, pages: List[Dict]) -> List[EnhancedChunk]:
        """
        Detailed mode: Split by pages, then by sections within pages.
        """
        chunks = []
        
        # Remove table HTML blocks
        clean_content = self._remove_tables_from_content(content)
        
        # Split by page breaks
        page_contents = clean_content.split("<!-- PageBreak -->") if "<!-- PageBreak -->" in clean_content else [clean_content]
        
        for page_idx, page_content in enumerate(page_contents):
            page_num = page_idx + 1
            
            # Split by headings within each page
            sections = self._split_by_headings(page_content)
            
            for section_title, section_content in sections:
                trimmed = section_content.strip()
                
                # Skip noise
                if self._is_noise(trimmed):
                    continue
                
                if len(trimmed) >= self.min_chunk_length:
                    header = self._build_header(
                        filename=filename,
                        section=section_title or f"Page {page_num}",
                        page=page_num
                    )
                    chunks.append(EnhancedChunk(
                        content=header + trimmed,
                        content_type="text",
                        page_number=page_num,
                        section=section_title,
                        metadata={}
                    ))
        
        return chunks
    
    def _process_by_pages_with_spatial(self, raw_ocr: Dict, content: str, filename: str, pages: List[Dict]) -> List[EnhancedChunk]:
        """
        Process pages with spatial pairing for column layouts.
        
        This method:
        1. Uses bounding boxes to detect column layouts
        2. Pairs labels with values based on spatial proximity
        3. Creates improved text chunks with proper key-value associations
        """
        chunks = []
        
        # Get spatially-paired content for each page
        paired_pages = self.spatial_pairing.process_document(raw_ocr)
        
        for page_num, paired_content in paired_pages.items():
            # Clean the content
            cleaned_content = self.pattern_detector.process_text(paired_content)
            
            # Skip if too short
            if len(cleaned_content.strip()) < self.min_chunk_length:
                continue
            
            # Skip noise
            if self._is_noise(cleaned_content):
                continue
            
            # Split by sections (headings) within the page
            sections = self._split_by_headings(cleaned_content)
            
            for section_title, section_content in sections:
                trimmed = section_content.strip()
                
                # Skip noise or short content
                if self._is_noise(trimmed) or len(trimmed) < self.min_chunk_length:
                    continue
                
                header = self._build_header(
                    filename=filename,
                    section=section_title or f"Page {page_num}",
                    page=page_num
                )
                
                chunks.append(EnhancedChunk(
                    content=header + trimmed,
                    content_type="text_spatial",  # Mark as spatially processed
                    page_number=page_num,
                    section=section_title,
                    metadata={
                        "spatial_pairing": True,
                        "processing_method": "column_detection"
                    }
                ))
        
        return chunks
    
    def _split_by_headings(self, content: str) -> List[Tuple[str, str]]:
        """Split content by markdown headings (#, ##, etc.)."""
        # Pattern to match markdown headings
        heading_pattern = r'^(#{1,6})\s+(.+)$'
        
        sections = []
        current_title = None
        current_content = []
        
        for line in content.split('\n'):
            match = re.match(heading_pattern, line)
            if match:
                # Save previous section
                if current_content:
                    sections.append((current_title, '\n'.join(current_content)))
                
                current_title = match.group(2).strip()
                current_content = []
            else:
                current_content.append(line)
        
        # Don't forget the last section
        if current_content:
            sections.append((current_title, '\n'.join(current_content)))
        
        return sections if sections else [(None, content)]
    
    def _remove_tables_from_content(self, content: str) -> str:
        """Remove both HTML and Markdown table blocks from content."""
        # Remove <table>...</table> HTML blocks
        clean = re.sub(r'<table>.*?</table>', '', content, flags=re.DOTALL)
        
        # Remove Markdown tables (lines starting with |)
        # Pattern: consecutive lines that start with | and contain |
        lines = clean.split('\n')
        filtered_lines = []
        in_table = False
        
        for line in lines:
            stripped = line.strip()
            # Detect markdown table row (starts with | or is separator like |---|---|)
            is_table_line = (
                stripped.startswith('|') and '|' in stripped[1:] or
                re.match(r'^\|[\s\-:|]+\|$', stripped)  # Table separator row
            )
            
            if is_table_line:
                in_table = True
                continue  # Skip table lines
            else:
                if in_table and stripped == '':
                    in_table = False  # End of table
                    continue
                in_table = False
                filtered_lines.append(line)
        
        return '\n'.join(filtered_lines)
    
    def _extract_figure_chunks(self, figures: List[Dict], content: str, filename: str) -> List[EnhancedChunk]:
        """Extract chunks for figures/images."""
        chunks = []
        
        for fig_idx, figure in enumerate(figures):
            bounding_regions = figure.get("boundingRegions", [])
            page_num = bounding_regions[0].get("pageNumber", 1) if bounding_regions else 1
            
            # Get caption if available
            caption = ""
            elements = figure.get("elements", [])
            
            # Try to extract text near the figure
            spans = figure.get("spans", [])
            if spans and content:
                for span in spans:
                    offset = span.get("offset", 0)
                    length = span.get("length", 0)
                    caption = content[offset:offset + length].strip()
            
            if caption and len(caption) > 10:
                header = self._build_header(
                    filename=filename,
                    section=f"Figure {fig_idx + 1}",
                    page=page_num
                )
                chunks.append(EnhancedChunk(
                    content=header + f"Figure: {caption}",
                    content_type="figure",
                    page_number=page_num,
                    section=f"Figure {fig_idx + 1}",
                    metadata={"figure_id": figure.get("id")}
                ))
        
        return chunks
    
    def _build_header(self, filename: str, section: str, page: int) -> str:
        """Build a contextual header prefix for chunks."""
        return f"[Source: {filename} | {section} | Page {page}]\n\n"
    
    def _is_noise(self, text: str) -> bool:
        """Check if text matches noise patterns."""
        for pattern in self.noise_patterns:
            if re.match(pattern, text.strip(), re.IGNORECASE):
                return True
        return False
    
    def _get_content_hash(self, text: str) -> str:
        """Generate hash for deduplication - hashes body AFTER stripping header prefix."""
        # Strip the [Source: ...] header prefix before hashing
        body = text
        if text.startswith("[Source:"):
            # Find end of header (]\n\n)
            header_end = text.find("]\n\n")
            if header_end != -1:
                body = text[header_end + 3:]  # Skip past "]\n\n"
        
        # Normalize and hash the body content
        normalized = body[:self.content_hash_length].lower().strip()
        normalized = re.sub(r'\s+', ' ', normalized)
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _filter_chunks(self, chunks: List[EnhancedChunk]) -> List[EnhancedChunk]:
        """Apply final quality filters and deduplication."""
        filtered = []
        
        for chunk in chunks:
            # Skip if too short
            if len(chunk.content) < self.min_chunk_length:
                continue
            
            # Skip noise
            if self._is_noise(chunk.content):
                continue
            
            # Deduplication
            content_hash = self._get_content_hash(chunk.content)
            if content_hash in self.seen_hashes:
                continue
            self.seen_hashes.add(content_hash)
            
            filtered.append(chunk)
        
        return filtered
    
    def to_vectordb_format(self, chunks: List[EnhancedChunk]) -> List[Dict]:
        """Convert chunks to format ready for vector DB storage."""
        return [
            {
                "text": chunk.content,
                "metadata": {
                    "content_type": chunk.content_type,
                    "page_number": chunk.page_number,
                    "section": chunk.section,
                    **chunk.metadata
                }
            }
            for chunk in chunks
        ]
    
    def save_chunks(self, chunks: List[EnhancedChunk], output_path: str):
        """Save chunks to JSON file."""
        data = {
            "total_chunks": len(chunks),
            "chunks": self.to_vectordb_format(chunks)
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Saved {len(chunks)} chunks to {output_path}")


def main():
    """Main function to run the enhanced chunker."""
    
    mode_str = "ONE-SHOT (page-based)" if USE_ONE_SHOT else "DETAILED (section-based)"
    
    print(f"\n{'='*60}")
    print("📄 ENHANCED CHUNKER")
    print(f"{'='*60}")
    print(f"   PDF Name:     {PDF_NAME}")
    print(f"   Input:        {RAW_OCR_PATH}")
    print(f"   Output:       {ENHANCED_CHUNKS_PATH}")
    print(f"   Mode:         {mode_str}")
    print(f"{'='*60}")
    
    # Initialize chunker
    chunker = EnhancedChunker(config={
        "min_chunk_length": 50,
        "max_chunk_length": 4000,
        "use_one_shot": USE_ONE_SHOT,  # Controlled by config variable at top
    })
    
    # Load raw OCR output
    raw_ocr = chunker.load_raw_ocr(RAW_OCR_PATH)
    
    # Extract chunks
    chunks = chunker.extract_chunks(raw_ocr, filename=PDF_NAME)
    
    # Save to file
    chunker.save_chunks(chunks, ENHANCED_CHUNKS_PATH)
    
    print(f"\n{'='*60}")
    print(f"✅ Chunking complete! Output saved to {ENHANCED_CHUNKS_PATH}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

