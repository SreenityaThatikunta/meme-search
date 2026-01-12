const API_URL = "http://127.0.0.1:8000/search/text";

document.getElementById("search").addEventListener("click", async () => {
  const query = document.getElementById("query").value.trim();
  const resultsDiv = document.getElementById("results");

  resultsDiv.innerHTML = "";

  if (!query) return;

  try {
    const res = await fetch(API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        query: query,
        top_k: 5
      })
    });

    const data = await res.json();

    if (data.length === 0) {
      resultsDiv.innerText = "No memes found";
      return;
    }

    data.forEach(meme => {
      const img = document.createElement("img");
      img.src = meme.image_url || meme.image_path;

      const score = document.createElement("div");
      score.className = "score";
      score.innerText = `Score: ${meme.score.toFixed(3)}`;

      resultsDiv.appendChild(img);
      resultsDiv.appendChild(score);
    });

  } catch (err) {
    resultsDiv.innerText = "Error contacting server";
    console.error(err);
  }
});