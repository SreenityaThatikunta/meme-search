"""Entry point for the Hugging Face Space.

The Space runs one process with no API and no database, so it needs the two
non-default settings below. Setting them here rather than in the Space's
Variables UI makes the deployment self-configuring: there is no manual step to
forget, and no failure mode where the Space silently tries to reach a MongoDB
that was never deployed.

``setdefault`` means a real environment variable still wins, so this file is
also the easiest way to run the demo locally:

    streamlit run demo/space_app.py
"""

import os
import runpy
import sys
from pathlib import Path

os.environ.setdefault("MEME_SEARCH_MODE", "local")
os.environ.setdefault("MEME_METADATA_BACKEND", "jsonl")
os.environ.setdefault("MEME_DEMO_SUBSET", "1")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Streamlit puts the *script's* directory on sys.path, which is demo/, so
# `import config` inside app.py would fail. Put the project root first.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Prefer demo/data when it is present, which is the case in a checkout that has
# run the subset builder. On the Space there is no demo/data — the payload is
# deployed to the default data/ paths — so this falls through and needs no
# configuring there. Checking for the subset rather than for data/ matters: a
# checkout always has data/, so testing that would silently load the full
# 6,989-meme dataset and the demo would never exercise what gets deployed.
DEMO_DATA = PROJECT_ROOT / "demo" / "data"
if (DEMO_DATA / "id_map.json").exists():
    os.environ.setdefault("MEME_DATA_DIR", str(DEMO_DATA))
    os.environ.setdefault("MEME_IMAGE_DIR", str(DEMO_DATA / "images" / "images"))
    os.environ.setdefault("MEME_TEXT_INDEX", str(DEMO_DATA / "text.index"))
    os.environ.setdefault("MEME_IMAGE_INDEX", str(DEMO_DATA / "image.index"))
    os.environ.setdefault("MEME_ID_MAP", str(DEMO_DATA / "id_map.json"))
    os.environ.setdefault("MEME_METADATA", str(DEMO_DATA / "memes_final.jsonl"))

runpy.run_path(str(PROJECT_ROOT / "app.py"), run_name="__main__")
