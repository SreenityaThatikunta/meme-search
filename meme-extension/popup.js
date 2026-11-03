const API_BASE_URL = "http://127.0.0.1:8000";
const SEARCH_URL = `${API_BASE_URL}/search/text`;
const TOP_K = 6;

const queryInput = document.getElementById("query");
const modeSelect = document.getElementById("mode");
const statusDiv = document.getElementById("status");
const resultsDiv = document.getElementById("results");

/** Render a short message above the results grid. */
function setStatus(message, isError = false) {
  statusDiv.textContent = message;
  statusDiv.className = isError ? "error" : "";
}

/** Build the card for one ranked meme. */
function renderMeme(meme) {
  const card = document.createElement("div");
  card.className = "card";

  const img = document.createElement("img");
  img.src = meme.image_url;
  img.alt = meme.ocr_text || meme.meme_id;
  img.loading = "lazy";

  const score = document.createElement("div");
  score.className = "score";
  score.textContent = `Score: ${meme.score.toFixed(3)}`;

  card.append(img, score);
  return card;
}

async function runSearch() {
  const query = queryInput.value.trim();
  resultsDiv.innerHTML = "";

  if (!query) {
    setStatus("Type something to search.");
    return;
  }

  setStatus("Searching...");

  try {
    const response = await fetch(SEARCH_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        top_k: TOP_K,
        mode: modeSelect.value,
      }),
    });

    if (!response.ok) {
      const detail = await response.text();
      setStatus(`API error ${response.status}: ${detail}`, true);
      return;
    }

    const memes = await response.json();

    if (memes.length === 0) {
      setStatus("No memes matched that query.");
      return;
    }

    setStatus("");
    memes.forEach((meme) => resultsDiv.appendChild(renderMeme(meme)));
  } catch (err) {
    // The backend runs locally, so this is almost always "server not started".
    setStatus(`Cannot reach the API at ${API_BASE_URL}. Is it running?`, true);
    console.error(err);
  }
}

document.getElementById("search").addEventListener("click", runSearch);
queryInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") runSearch();
});
