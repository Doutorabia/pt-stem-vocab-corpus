from pathlib import Path
import os

ROOT_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = ROOT_DIR / "data"
PDF_DIR    = DATA_DIR / "pdfs"
OUTPUT_DIR = DATA_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

UNWANTED_WORDS_PATH = ROOT_DIR / "unwanted_words.json"

AZURE_ENDPOINT: str = os.getenv("AZURE_ENDPOINT", "")
AZURE_KEY: str      = os.getenv("AZURE_KEY", "")