from __future__ import annotations

from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient

from pathlib import Path
from typing import Iterable
import logging

from .config import AZURE_ENDPOINT, AZURE_KEY

_logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Extracts plain text from PDF files using Azure Document Intelligence."""

    def __init__(
        self,
        endpoint: str | None = None,
        api_key: str | None = None,
    ) -> None:
        endpoint = endpoint or AZURE_ENDPOINT
        api_key  = api_key or AZURE_KEY
        if not endpoint or not api_key:
            raise ValueError(
                "Azure credentials not found. "
                "Set AZURE_ENDPOINT and AZURE_KEY as environment variables."
            )
        self._client = DocumentIntelligenceClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(api_key),
        )

    # --------------------------------------------------------------------- #
    def extract_text(self, pdf: Path | str) -> str:
        """Extracts **all** text from a single PDF file.

        Args:
            pdf: Path to the PDF file (str or Path).

        Returns:
            Content extracted from the document or an empty string on failure.
        """
        pdf = Path(pdf)
        try:
            with pdf.open("rb") as fh:
                poller = self._client.begin_analyze_document(
                    model_id="prebuilt-layout",
                    analyze_request=fh,
                    content_type="application/octet-stream",
                )
            result = poller.result()
            return result.content or ""
        except Exception as exc:
            _logger.error("Error while processing %s – %s", pdf, exc)
            return ""

    # --------------------------------------------------------------------- #
    def batch_extract(self, pdfs: Iterable[Path | str]) -> list[str]:
        return [self.extract_text(p) for p in pdfs]