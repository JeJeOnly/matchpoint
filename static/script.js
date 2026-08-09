(function () {
  const playerA = document.getElementById("player-a");
  const playerB = document.getElementById("player-b");
  const playersList = document.getElementById("players-list");
  const predictBtn = document.getElementById("predict-btn");
  const errorMsg = document.getElementById("error-msg");
  const resultCard = document.getElementById("result");
  const boButtons = Array.from(document.querySelectorAll(".bo-btn"));
  const previewA = document.getElementById("preview-a");
  const previewB = document.getElementById("preview-b");

  let bestOf = 3;
  let playersByName = {};

  boButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      boButtons.forEach((b) => b.classList.remove("is-active"));
      btn.classList.add("is-active");
      bestOf = Number(btn.dataset.bestOf);
    });
  });

  fetch("/api/players")
    .then((r) => r.json())
    .then((players) => {
      playersByName = Object.fromEntries(players.map((p) => [p.name, p]));
      playersList.innerHTML = players
        .map((p) => `<option value="${escapeHtml(p.name)}"></option>`)
        .join("");
    })
    .catch(() => {
      showError("Could not load the player list. Is the server running?");
    });

  function escapeHtml(s) {
    return s.replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[c]));
  }

  function flagEmoji(countryCode) {
    if (!countryCode || countryCode.length !== 2) return "";
    const base = 127397;
    return String.fromCodePoint(...[...countryCode.toUpperCase()].map((c) => base + c.charCodeAt(0)));
  }

  function updatePreview(input, previewEl) {
    const player = playersByName[input.value.trim()];
    if (!player) {
      previewEl.hidden = true;
      return;
    }
    previewEl.querySelector(".preview-flag").textContent = flagEmoji(player.country_code);
    previewEl.querySelector(".preview-country").textContent = `${player.country_code} · ${player.hand === "L" ? "Left-handed" : "Right-handed"}`;
    previewEl.querySelector(".preview-style").textContent = player.style;
    previewEl.hidden = false;
  }

  playerA.addEventListener("input", () => updatePreview(playerA, previewA));
  playerB.addEventListener("input", () => updatePreview(playerB, previewB));

  function showError(msg) {
    errorMsg.textContent = msg;
    errorMsg.hidden = false;
  }

  function clearError() {
    errorMsg.hidden = true;
    errorMsg.textContent = "";
  }

  predictBtn.addEventListener("click", async () => {
    clearError();
    resultCard.hidden = true;

    const a = playerA.value.trim();
    const b = playerB.value.trim();

    if (!a || !b) {
      showError("Enter both players.");
      return;
    }
    if (a === b) {
      showError("Choose two different players.");
      return;
    }

    predictBtn.disabled = true;
    predictBtn.querySelector("span").textContent = "Crunching the numbers…";

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ player_a: a, player_b: b, best_of: bestOf }),
      });
      const data = await res.json();

      if (!res.ok) {
        showError(data.error || "Something went wrong.");
        return;
      }

      renderResult(data);
    } catch (err) {
      showError("Could not reach the server.");
    } finally {
      predictBtn.disabled = false;
      predictBtn.querySelector("span").textContent = "Predict winner";
    }
  });

  function renderResult(data) {
    const pctA = Math.round(data.prob_a * 1000) / 10;
    const pctB = Math.round(data.prob_b * 1000) / 10;
    const favorite = pctA >= pctB ? data.player_a : data.player_b;
    const favPct = Math.max(pctA, pctB);

    document.getElementById("favorite-name").textContent = favorite.name;
    document.getElementById("favorite-pct").textContent = `${favPct.toFixed(1)}%`;

    document.getElementById("bar-a").style.width = `${pctA}%`;
    document.getElementById("bar-b").style.width = `${pctB}%`;
    document.getElementById("bar-split").style.left = `${pctA}%`;

    document.getElementById("name-a").textContent = data.player_a.name;
    document.getElementById("name-b").textContent = data.player_b.name;
    document.getElementById("pct-a").textContent = `${pctA.toFixed(1)}%`;
    document.getElementById("pct-b").textContent = `${pctB.toFixed(1)}%`;

    document.querySelector(".stats-name-a").textContent = data.player_a.name;
    document.querySelector(".stats-name-b").textContent = data.player_b.name;
    document.querySelector(".stat-spw-a").textContent = `${data.player_a.serve_pts_won.toFixed(0)}%`;
    document.querySelector(".stat-spw-b").textContent = `${data.player_b.serve_pts_won.toFixed(0)}%`;
    document.querySelector(".stat-rpw-a").textContent = `${data.player_a.return_pts_won.toFixed(0)}%`;
    document.querySelector(".stat-rpw-b").textContent = `${data.player_b.return_pts_won.toFixed(0)}%`;
    document.querySelector(".stat-p-a").textContent = `${(data.point_win_pct_a * 100).toFixed(1)}% on serve`;
    document.querySelector(".stat-p-b").textContent = `${(data.point_win_pct_b * 100).toFixed(1)}% on serve`;

    renderBio("a", data.player_a);
    renderBio("b", data.player_b);
    renderH2H(data.player_a, data.player_b, data.head_to_head);

    resultCard.hidden = false;
    resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function renderBio(slot, player) {
    document.getElementById(`bio-flag-${slot}`).textContent = flagEmoji(player.country_code);
    document.getElementById(`bio-name-${slot}`).textContent = player.name;
    const hand = player.hand === "L" ? "Left-handed" : "Right-handed";
    document.getElementById(`bio-line-${slot}`).textContent = `${player.style} · ${hand}`;
  }

  function renderH2H(playerAInfo, playerBInfo, h2h) {
    const known = document.getElementById("h2h-known");
    const unknown = document.getElementById("h2h-unknown");

    if (!h2h || (h2h.wins_a === 0 && h2h.wins_b === 0)) {
      known.hidden = true;
      unknown.hidden = false;
      return;
    }

    const total = h2h.wins_a + h2h.wins_b;
    const pctA = (h2h.wins_a / total) * 100;
    const pctB = 100 - pctA;

    document.getElementById("h2h-bar-a").style.width = `${pctA}%`;
    document.getElementById("h2h-bar-b").style.width = `${pctB}%`;
    document.getElementById("h2h-record-a").textContent = `${playerAInfo.name} ${h2h.wins_a}`;
    document.getElementById("h2h-record-b").textContent = `${h2h.wins_b} ${playerBInfo.name}`;

    unknown.hidden = true;
    known.hidden = false;
  }
})();
