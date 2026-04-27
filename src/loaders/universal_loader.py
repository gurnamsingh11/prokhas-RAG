"""
Universal document loader — supports PDF, DOCX, and images (via OCR stub).
Identical to the provided snippet; OCR is stubbed so the server starts
without a GPU/Tesseract installation.  Replace the stub with the real
extract_text_from_image import when your OCR service is ready.
"""

import logging
import os
from pathlib import Path
from typing import List
from pdf2image import convert_from_path
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
import tempfile
from src.config.config import settings

logger = logging.getLogger(__name__)


# ── OCR stub (replace with real import when available) ───────────────────────
def _ocr_stub(image_path: str, image_name: str) -> Document:
    """Placeholder OCR — returns an empty document so the pipeline still runs."""
    logger.warning(
        "OCR stub called for %s — install OCR service and replace this stub.",
        image_name,
    )
    return Document(
        page_content="[Image content not extracted — OCR service unavailable]",
        metadata={"source": image_name, "page_label": ""},
    )


try:
    from src.ocr_extraction.main import extract_text_from_image  # type: ignore
except ImportError:
    extract_text_from_image = _ocr_stub  # type: ignore


# ── Loader ────────────────────────────────────────────────────────────────────


class UniversalDocumentLoader:
    """Load PDF, Word, or Image files and return LangChain Document objects."""

    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
    PDF_EXTENSIONS = {".pdf"}
    WORD_EXTENSIONS = {".docx"}

    def load(self, file_path: str) -> List[Document]:
        ext = Path(file_path).suffix.lower()

        if ext in self.PDF_EXTENSIONS:
            return self._load_pdf(file_path)
        elif ext in self.WORD_EXTENSIONS:
            return self._load_word(file_path)
        elif ext in self.IMAGE_EXTENSIONS:
            return self._load_image(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _get_filename(file_path: str) -> str:
        return os.path.basename(file_path)

    def _load_pdf(self, file_path: str) -> List[Document]:
        loader = PyPDFLoader(file_path)
        if not loader:
            logger.error("PyPDFLoader unavailable for %s", file_path)

        first_page = next(loader.lazy_load())

        if not first_page:
            logger.warning("No first page found in %s", file_path)
        first_text = first_page.page_content.strip()

        is_scanned = not first_text or len(first_text) < 20
        filename = self._get_filename(file_path)

        if is_scanned:
            logger.info("PDF appears scanned — falling back to OCR: %s", filename)
            final_docs: List[Document] = []
            images = convert_from_path(file_path, poppler_path=settings.POPPLER_PATH)

            for i, image in enumerate(images):
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    image_path = tmp.name
                    image.save(image_path, "PNG")

                ocr_docs = self._load_image(image_path)

                for ocr_doc in ocr_docs:
                    ocr_doc.metadata = {
                        "source": filename,
                        "page_label": str(i),
                        "ocr": True,
                    }

                final_docs.extend(ocr_docs)
                os.remove(image_path)

            return final_docs

        else:
            docs = loader.load()
            for doc in docs:
                page_label = doc.metadata.get("page_label", "")
                doc.metadata = {"source": filename, "page_label": str(page_label)}
            return docs

    def _load_word(self, file_path: str) -> List[Document]:
        loader = Docx2txtLoader(file_path)
        docs = loader.load()
        filename = self._get_filename(file_path)
        for doc in docs:
            doc.metadata["source"] = filename
            doc.metadata.setdefault("page_label", "")
        return docs

    def _load_image(self, file_path: str) -> List[Document]:
        filename = self._get_filename(file_path)
        doc = extract_text_from_image(image_path=file_path, image_name=filename)
        # doc.metadata.setdefault("source", filename)
        # doc.metadata.setdefault("page_label", "")
        return [doc]
