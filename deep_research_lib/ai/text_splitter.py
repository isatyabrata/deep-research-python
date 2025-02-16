# deep-research-python/deep_research_lib/ai/text_splitter.py
from typing import List, Optional

class TextSplitter:
    """Abstract class for splitting text into chunks."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Initialize the text splitter."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("Cannot have chunk_overlap >= chunk_size")

    def split_text(self, text: str) -> List[str]:
        """Split text into chunks."""
        raise NotImplementedError()

    def create_documents(self, texts: List[str]) -> List[str]:
        """Create documents from a list of texts."""
        documents = []
        for text in texts:
            for chunk in self.split_text(text):
                documents.append(chunk)
        return documents

    def split_documents(self, documents: List[str]) -> List[str]:
        """Split documents."""
        return self.create_documents(documents)

    def _join_docs(self, docs: List[str], separator: str) -> Optional[str]:
        """Join documents with separator."""
        text = separator.join(docs).strip()
        return text if text else None

    def merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """Merge splits into documents."""
        docs = []
        current_doc = []
        total = 0
        for d in splits:
            _len = len(d)
            if total + _len >= self.chunk_size:
                if total > self.chunk_size:
                    print(
                        f"Created a chunk of size {total}, "
                        f"which is longer than the specified {self.chunk_size}"
                    )
                if current_doc:
                    doc = self._join_docs(current_doc, separator)
                    if doc is not None:
                        docs.append(doc)
                    # Keep on popping if:
                    # - we have a larger chunk than in the chunk overlap
                    # - or if we still have any chunks and the length is long
                    while (
                        total > self.chunk_overlap
                        or (total + _len > self.chunk_size and total > 0)
                    ):
                        total -= len(current_doc[0])
                        current_doc.pop(0)
            current_doc.append(d)
            total += _len
        doc = self._join_docs(current_doc, separator)
        if doc is not None:
            docs.append(doc)
        return docs


class RecursiveCharacterTextSplitter(TextSplitter):
    """Implementation of splitting text by characters recursively."""

    def __init__(
        self,
        separators: Optional[List[str]] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """Initialize the recursive text splitter."""
        self.separators = separators or ["\n\n", "\n", ".", ",", ">", "<", " ", ""]
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def split_text(self, text: str) -> List[str]:
        """Split text into chunks."""
        final_chunks = []

        # Get appropriate separator to use
        separator = self.separators[-1]
        for s in self.separators:
            if s == "":
                separator = s
                break
            if s in text:
                separator = s
                break

        # Now that we have the separator, split the text
        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)  # Split by character if no separator found

        # Now go merging things, recursively splitting longer texts.
        good_splits = []
        for s in splits:
            if len(s) < self.chunk_size:
                good_splits.append(s)
            else:
                if good_splits:
                    merged_text = self.merge_splits(good_splits, separator)
                    final_chunks.extend(merged_text)
                    good_splits = []
                other_info = self.split_text(s)
                final_chunks.extend(other_info)
        if good_splits:
            merged_text = self.merge_splits(good_splits, separator)
            final_chunks.extend(merged_text)
        return final_chunks