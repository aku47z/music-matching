// MIDI Plagiarism Detector - Frontend JavaScript

document.addEventListener("DOMContentLoaded", function () {
  const form = document.getElementById("analyzeForm");
  const analyzeBtn = document.getElementById("analyzeBtn");
  const resultsSection = document.getElementById("results");
  const resultsContent = document.getElementById("results-content");
  const errorMessage = document.getElementById("error-message");

  // File input handlers
  const fileA = document.getElementById("file_a");
  const fileB = document.getElementById("file_b");
  const sampleA = document.getElementById("sample_a");
  const sampleB = document.getElementById("sample_b");
  const fileAName = document.getElementById("file_a_name");
  const fileBName = document.getElementById("file_b_name");
  const toggleNodesBtn = document.getElementById("toggle-nodes-btn");
  const forensicBtn = document.getElementById("forensic-btn");
  const forensicSection = document.getElementById("forensic-section");
  const forensicResults = document.getElementById("forensic-results");

  let showAllNodes = false;
  let lastAnalysisData = null;
  let currentSongA = null;
  let currentSongB = null;

  // Load samples from API
  async function loadSamples() {
    try {
      const response = await fetch("/api/samples");
      const samples = await response.json();

      samples.forEach((sample) => {
        const optionA = document.createElement("option");
        optionA.value = sample.filename;
        optionA.textContent = sample.name;
        sampleA.appendChild(optionA);

        const optionB = document.createElement("option");
        optionB.value = sample.filename;
        optionB.textContent = sample.name;
        sampleB.appendChild(optionB);
      });
    } catch (error) {
      console.error("Failed to load samples:", error);
    }
  }

  loadSamples();

  // Toggle nodes button
  toggleNodesBtn.addEventListener("click", async function () {
    if (!lastAnalysisData) return;

    showAllNodes = !showAllNodes;
    toggleNodesBtn.textContent = showAllNodes
      ? "Show Top 15"
      : "Show All Nodes";

    // Re-run analysis with new max_nodes setting
    setLoading(true);

    const formData = new FormData();

    if (lastAnalysisData.hasFileA) {
      formData.append("file_a", fileA.files[0]);
    } else {
      formData.append("sample_a", sampleA.value);
    }

    if (lastAnalysisData.hasFileB) {
      formData.append("file_b", fileB.files[0]);
    } else {
      formData.append("sample_b", sampleB.value);
    }

    formData.append("max_nodes", showAllNodes ? "all" : "15");

    try {
      const response = await fetch("/api/analyze", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (data.success) {
        displayResults(data);
      } else {
        showError(data.error || "Analysis failed");
      }
    } catch (error) {
      showError("Network error: " + error.message);
    } finally {
      setLoading(false);
    }
  });

  // Show selected file name
  fileA.addEventListener("change", function () {
    if (this.files.length > 0) {
      fileAName.textContent = "✓ " + this.files[0].name;
      sampleA.value = ""; // Clear sample selection
    } else {
      fileAName.textContent = "";
    }
  });

  fileB.addEventListener("change", function () {
    if (this.files.length > 0) {
      fileBName.textContent = "✓ " + this.files[0].name;
      sampleB.value = ""; // Clear sample selection
    } else {
      fileBName.textContent = "";
    }
  });

  // Clear file input when sample is selected
  sampleA.addEventListener("change", function () {
    if (this.value) {
      fileA.value = "";
      fileAName.textContent = "";
    }
  });

  sampleB.addEventListener("change", function () {
    if (this.value) {
      fileB.value = "";
      fileBName.textContent = "";
    }
  });

  // Form submission
  form.addEventListener("submit", async function (e) {
    e.preventDefault();

    // Validate inputs
    const hasFileA = fileA.files.length > 0;
    const hasSampleA = sampleA.value !== "";
    const hasFileB = fileB.files.length > 0;
    const hasSampleB = sampleB.value !== "";

    if (!hasFileA && !hasSampleA) {
      alert("Please select or upload a file for Song A");
      return;
    }

    if (!hasFileB && !hasSampleB) {
      alert("Please select or upload a file for Song B");
      return;
    }

    // Show loading state
    setLoading(true);

    // Prepare form data
    const formData = new FormData();

    if (hasFileA) {
      formData.append("file_a", fileA.files[0]);
    } else {
      formData.append("sample_a", sampleA.value);
    }

    if (hasFileB) {
      formData.append("file_b", fileB.files[0]);
    } else {
      formData.append("sample_b", sampleB.value);
    }

    // Add max_nodes parameter
    formData.append("max_nodes", showAllNodes ? "all" : "15");

    try {
      const response = await fetch("/api/analyze", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (data.success) {
        lastAnalysisData = { hasFileA, hasSampleA, hasFileB, hasSampleB };
        // Store current songs for forensic mode
        currentSongA = hasFileA ? fileA.files[0].name : sampleA.value;
        currentSongB = hasFileB ? fileB.files[0].name : sampleB.value;
        displayResults(data);
      } else {
        showError(data.error || "An error occurred during analysis");
      }
    } catch (error) {
      console.error("Error:", error);
      showError("Failed to connect to server. Please try again.");
    } finally {
      setLoading(false);
    }
  });

  function setLoading(loading) {
    analyzeBtn.disabled = loading;
    analyzeBtn.querySelector(".btn-text").style.display = loading
      ? "none"
      : "inline";
    analyzeBtn.querySelector(".btn-loading").style.display = loading
      ? "inline-flex"
      : "none";
  }

  function showError(message) {
    resultsSection.style.display = "block";
    resultsContent.style.display = "none";
    errorMessage.style.display = "block";
    errorMessage.textContent = message;

    // Scroll to results
    resultsSection.scrollIntoView({ behavior: "smooth" });
  }

  function displayResults(data) {
    resultsSection.style.display = "block";
    resultsContent.style.display = "block";
    errorMessage.style.display = "none";

    // Scores
    updateScore("hook", data.hook_score);
    updateScore("ngram", data.ngram_score);

    // Details
    document.getElementById("notes-a").textContent = data.notes_a;
    document.getElementById("notes-b").textContent = data.notes_b;
    document.getElementById("fragments-a").textContent = data.fragments_a;
    document.getElementById("fragments-b").textContent = data.fragments_b;

    // Bipartite Graph Visualization
    const graphSection = document.getElementById("graph-section");
    const graphImage = document.getElementById("graph-image");

    if (data.graph_image) {
      graphSection.style.display = "block";
      graphImage.src = "data:image/png;base64," + data.graph_image;
    } else {
      graphSection.style.display = "none";
    }

    // Top matches
    const matchesSection = document.getElementById("matches-section");
    const matchesTbody = document.getElementById("matches-tbody");

    if (data.top_matches && data.top_matches.length > 0) {
      matchesSection.style.display = "block";
      matchesTbody.innerHTML = "";

      data.top_matches.forEach((match, index) => {
        const row = document.createElement("tr");
        const scoreClass =
          match.score > 0.7 ? "high" : match.score >= 0.4 ? "medium" : "low";

        row.innerHTML = `
                    <td>${index + 1}</td>
                    <td><span class="similarity-badge ${scoreClass}">${(match.score * 100).toFixed(1)}%</span></td>
                    <td>${match.time_a}</td>
                    <td>${match.time_b}</td>
                `;
        matchesTbody.appendChild(row);
      });
    } else {
      matchesSection.style.display = "none";
    }

    // Show Forensic Ranking Mode section (but hide previous results)
    forensicSection.style.display = "block";
    forensicResults.style.display = "none";

    // Scroll to results
    resultsSection.scrollIntoView({ behavior: "smooth" });
  }

  function updateScore(type, score) {
    const scoreValue = document.getElementById(`${type}-score`);
    const scoreFill = document.getElementById(`${type}-fill`);

    // Update value
    scoreValue.textContent = (score * 100).toFixed(1) + "%";

    // Update fill bar
    const percentage = Math.min(score * 100, 100);
    scoreFill.style.width = percentage + "%";

    // Update color class
    scoreFill.className = "score-fill";
    if (score > 0.7) {
      scoreFill.classList.add("high");
    } else if (score >= 0.4) {
      scoreFill.classList.add("medium");
    } else {
      scoreFill.classList.add("low");
    }
  }

  // Forensic Ranking Mode
  forensicBtn.addEventListener("click", async function () {
    if (!lastAnalysisData) return;

    // Set loading state
    forensicBtn.disabled = true;
    forensicBtn.querySelector(".btn-text").style.display = "none";
    forensicBtn.querySelector(".btn-loading").style.display = "inline-flex";

    const formData = new FormData();

    // Add Song A (query melody)
    if (lastAnalysisData.hasFileA) {
      formData.append("file_a", fileA.files[0]);
    } else {
      formData.append("sample_a", sampleA.value);
    }

    // Add Song B (suspect)
    if (lastAnalysisData.hasFileB) {
      formData.append("file_b", fileB.files[0]);
    } else {
      formData.append("sample_b", sampleB.value);
    }

    try {
      const response = await fetch("/api/forensic", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (data.success) {
        displayForensicResults(data);
      } else {
        alert("Forensic analysis failed: " + (data.error || "Unknown error"));
      }
    } catch (error) {
      console.error("Error:", error);
      alert("Failed to run forensic analysis. Please try again.");
    } finally {
      forensicBtn.disabled = false;
      forensicBtn.querySelector(".btn-text").style.display = "inline";
      forensicBtn.querySelector(".btn-loading").style.display = "none";
    }
  });

  let showSegmentScores = true; // Default to segment score (best for plagiarism detection)
  let forensicData = null;

  function displayForensicResults(data) {
    forensicResults.style.display = "block";
    forensicData = data;
    showSegmentScores = true; // Reset to default

    // Reset toggle button
    const toggleBtn = document.getElementById("toggle-score-btn");
    toggleBtn.textContent = "Show Overall Scores";
    document.getElementById("score-header").textContent = "Segment Score";

    // Summary
    document.getElementById("forensic-summary-text").innerHTML =
      `Compared <strong>${data.total_compared}</strong> songs. Suspect song ranked <strong>#${data.suspect_rank}</strong>.`;

    // Verdict
    const verdictEl = document.getElementById("forensic-verdict");
    if (data.suspect_is_top) {
      verdictEl.className = "forensic-verdict high-confidence";
      verdictEl.innerHTML =
        "🚨 <strong>High Forensic Confidence:</strong> The suspect song is the closest structural match in the entire library.";
    } else if (data.suspect_rank <= 3) {
      verdictEl.className = "forensic-verdict high-confidence";
      verdictEl.innerHTML =
        "⚠️ <strong>Suspicious:</strong> The suspect song ranks in the top 3 most similar songs.";
    } else {
      verdictEl.className = "forensic-verdict low-confidence";
      verdictEl.innerHTML =
        "✅ <strong>Low Similarity:</strong> The suspect song does not rank among the top matches.";
    }

    renderForensicTable(data.rankings, "segment_score");

    // Scroll to forensic results
    forensicResults.scrollIntoView({ behavior: "smooth" });
  }

  function renderForensicTable(rankings, scoreKey) {
    // Sort by the selected score type
    const sorted = [...rankings].sort((a, b) => b[scoreKey] - a[scoreKey]);

    const tbody = document.getElementById("forensic-tbody");
    tbody.innerHTML = "";

    sorted.forEach((item, index) => {
      const row = document.createElement("tr");

      // Highlight suspect song
      if (item.is_suspect) {
        row.classList.add("suspect-row");
      }

      // Score class based on value
      const score = item[scoreKey];
      const scoreClass = score > 0.7 ? "high" : score >= 0.4 ? "medium" : "low";

      row.innerHTML = `
        <td>${index + 1}</td>
        <td>${item.name}${item.is_suspect ? " ★" : ""}</td>
        <td><span class="similarity-badge ${scoreClass}">${(score * 100).toFixed(2)}%</span></td>
      `;
      tbody.appendChild(row);
    });
  }

  // Toggle score type button - switches between segment and overall scores
  document
    .getElementById("toggle-score-btn")
    .addEventListener("click", function () {
      if (!forensicData) return;

      showSegmentScores = !showSegmentScores;
      const scoreKey = showSegmentScores ? "segment_score" : "overall_score";

      this.textContent = showSegmentScores
        ? "Show Overall Scores"
        : "Show Segment Scores";
      document.getElementById("score-header").textContent = showSegmentScores
        ? "Segment Score"
        : "Overall Score";

      renderForensicTable(forensicData.rankings, scoreKey);
    });
});
