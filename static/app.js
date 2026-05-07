// DOM elements
const fileInput = document.getElementById("fileInput");
const preview = document.getElementById("preview");
const upload = document.getElementById("customUpload");
const analyzeButton = document.getElementById("analyzeButton");
const mlButton = document.getElementById("mlButton");
const resultBox = document.getElementById("result");
const loadingIndicator = document.getElementById("loadingIndicator");

// clicking an evidence image opens it up and shows a full-screen overlay
const imageViewer = document.createElement("div");
imageViewer.id = "imageViewer";
imageViewer.innerHTML = '<img id="imageViewer-img" alt=""><button id="imageViewer-close">&times;</button><p id="imageViewer-hint">Click image to zoom</p>';
document.body.appendChild(imageViewer);

const imageViewerImg = document.getElementById("imageViewer-img");
const imageViewerHint = document.getElementById("imageViewer-hint");

let zoomModeActive = false;

function setZoomMode(active) {
    zoomModeActive = active;
    imageViewerImg.classList.toggle("zoom-active", active);
    imageViewerHint.textContent = active ? "Click image to zoom out" : "Click image to zoom in";
    if (!active) {
        imageViewerImg.style.transform = "scale(1)";
        imageViewerImg.style.transformOrigin = "center center";
    }
}

function openImageViewer(src) {
    imageViewerImg.src = src;
    imageViewer.classList.add("active");
}

function closeImageViewer() {
    imageViewer.classList.remove("active");
    imageViewerImg.src = "";
    setZoomMode(false);
}

document.getElementById("imageViewer-close").addEventListener("click", closeImageViewer);
imageViewer.addEventListener("click", (e) => { if (e.target === imageViewer) closeImageViewer(); });
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeImageViewer(); });

// click image to toggle zoom mode on/off
imageViewerImg.addEventListener("click", (e) => {
    e.stopPropagation();
    setZoomMode(!zoomModeActive);
});

// while zoom mode is active, follow cursor and scale at that origin
imageViewerImg.addEventListener("mousemove", (e) => {
    if (!zoomModeActive) return;
    const rect = imageViewerImg.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * 100;
    const y = ((e.clientY - rect.top) / rect.height) * 100;
    imageViewerImg.style.transformOrigin = `${x}% ${y}%`;
    imageViewerImg.style.transform = "scale(2.5)";
});

imageViewerImg.addEventListener("mouseleave", () => {
    if (!zoomModeActive) return;
    imageViewerImg.style.transform = "scale(1)";
    imageViewerImg.style.transformOrigin = "center center";
});

// delegate clicks on evidence images to open the full-screen viewer
resultBox.addEventListener("click", (e) => {
    if (e.target.classList.contains("evidence-img")) {
        openImageViewer(e.target.src);
    }
});

// show image preview immediately when a file is chosen
fileInput.addEventListener("change", () => {
    const file = fileInput.files[0];

    // clear previous results
    resultBox.innerHTML = ""; 

    if (file) {
    const reader = new FileReader();
    reader.onload = function(e) {
        preview.src = e.target.result;
        preview.style.display = "block";
        };
    reader.readAsDataURL(file);
    } else {
    preview.src = "";
    preview.style.display = "none";
    }
    refreshAnalyzeState();
});

// toggle for which model to use (read flags on submit)
let modelChoice = null;

// function setActive(whichId) {
//     mlButton.classList.toggle("active", whichId === "mlButton");
//     // dlButton.classList.toggle("active", whichId === "dlButton");
// }
// setActive(null);  // clear initial selection

// mlButton.addEventListener("click", () => {
//     if (!fileInput.files.length) return;
//     modelChoice = "ml";
//     setActive("mlButton");
//     refreshAnalyzeState();
// });

// dlButton.addEventListener("click", () => {
//     modelChoice = "dl";
//     setActive("dlButton");
//     refreshAnalyzeState();
// });

// analyze button state
function refreshAnalyzeState() {
    // const hasModel = !!modelChoice;
    const hasFile = fileInput.files.length > 0;

    analyzeButton.hidden = !hasFile;
}
refreshAnalyzeState();

function setLoading(isLoading) {
    if (isLoading) {
        loadingIndicator.classList.remove("hidden");
        analyzeButton.disabled = true;
        analyzeButton.innerHTML = '<span class="btn-spinner"></span>';
    } else {
        loadingIndicator.classList.add("hidden");
        analyzeButton.disabled = false;
        analyzeButton.textContent = "Analyze";
    }
}

// clicking Analyze triggers submit (submit form)
analyzeButton.addEventListener("click", () => {
    document.getElementById("uploadForm").requestSubmit();
});

document.getElementById("uploadForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!fileInput.files.length) {
        alert("Please select an image first!");
        return;
    }

    const formData = new FormData();
    modelChoice = "ml"
    formData.append("file", fileInput.files[0]);
    formData.append("model", modelChoice);

    setLoading(true);

    try {
        // send to FastAPI
        const response = await fetch("/process/", {
            method: "POST",
            body: formData
        });

        // read JSON response
        const contentType = response.headers.get("content-type") || "";
        if (!contentType.includes("application/json")) {
            throw new Error(`Server returned HTTP ${response.status} (not JSON — possible upload size limit exceeded)`);
        }
        const data = await response.json();
        console.log("Response JSON from server:", data);

        // if backend sent an error message, show it
        if (!response.ok || data.error) {
            resultBox.innerHTML = `
                <div class="error">
                    <b>Error:</b> ${data.error || ("HTTP " + response.status)}
                </div>
            `;
            return;
        }
        renderResults(data);
    } catch (err) {
        console.error("Fetch or JSON parse error:", err)
        resultBox.innerHTML = `
            <div class="error">
                <b>Client error:</b> ${err}
            </div>
        `;
    } finally {
        setLoading(false);
    }

});

// display voting result
function formatConsensus(v, reason) {
    if (v === 0 || v === "0") return `<span class="real">Real</span>`;
    if (v === 1 || v === "1") return `<span class="tampered">Tampered</span>`;
    return `<span class="unknown">Inconclusive</span>`;
}

function formatConsensusNote(v, reason, meanScore) {
    if (v === 0 || v === "0" || v === 1 || v === "1") return '';

    if (reason === "tie") {
        const meanHint = meanScore != null
            ? `Mean score across all modules: ${meanScore.toFixed(3)} &ndash; ${meanScore > 0.5 ? 'leans suspicious' : 'leans real'}.`
            : '';
        return `<p class="consensus-note">${meanHint} Modules split evenly &ndash; manual review recommended.</p>`;
    }
    return `<p class="consensus-note">All modules returned borderline scores &ndash; manual review recommended.</p>`;
}

function formatVote(v) {
    // Normalize "0"/"1" strings to numbers
    if (v === "0") v = 0;
    if (v === "1") v = 1;

    if (v === 0) {
        return `<span class="real">Real</span>`;
    }
    if (v === 1) {
        return `<span class="tampered">Tampered</span>`;
    }

    if (v === null || v === undefined) {
        return `<span class="unknown">N/A</span>`;
    }

    // fallback: show whatever we got
    return `<span class="unknown">${v}</span>`;
}

// captions explaining what each evidence image shows
const evidenceCaptions = {
    texture: {
        shadow_overlay: 'Detected shadow regions (red). These are the areas analyzed for texture consistency.',
        lbp_map: 'LBP texture map - red outline marks shadow boundaries. Uniform patterns across the boundary suggest a real shadow; a sharp shift in pattern is suspicious.'
    },
    lighting: {
        component_overlay: 'Each shadow region shown in a distinct color. Real images typically have shadows from a single consistent light source.',
        ratio_heatmap: 'Shadow brightness ratio map - each region is colored by how dark it is relative to its surroundings. Similar colors across all shadows mean a consistent light source; varied colors suggest different sources.'
    },
    depth: {
        contour_overlay: 'Green dots mark the shadow boundary locations sampled to measure penumbra width. The penumbra is the outer edge of a shadow where light gradually transitions to dark. The system checks each marked point for that gradual fade vs. a sharp cutoff.',
        orientation_overlay: 'Cyan box = tightest fitted rectangle around each shadow (box proportions show the elongation ratio). Magenta line = long axis. Consistent box shapes and parallel axes suggest one light source; mismatched shapes or scattered axes suggest composited shadows.'
    }
};

// display results on page
function renderResults(data) {
    resultBox.innerHTML = "";

    // backend-side error safeguard (in case it wasn't caught earlier)
    if (data.error) {
        resultBox.innerHTML = `
            <div class="error">
                <b>Error:</b> ${data.error}
            </div>
        `;
        return;
    }

    const scores = data.rule_based_scores;
    const votes = data.rule_based_votes;
    const evidence = data.evidence_images || {};

    console.log("scores from server:", scores);
    console.log("votes from server:", votes);
    console.log("final vote from server:", data.final_rule_based_vote);
    console.log("ml prediction:", data.ml_prediction);
    console.log("ml probability tampered:", data.ml_probability_tampered);
    console.log("ml module probabilities:", data.ml_module_probabilities);

    const shouldShowMl =
        modelChoice === "ml" &&
        data.ml_prediction !== null &&
        data.ml_prediction !== undefined;

    const moduleProbs = data.ml_module_probabilities || {};

    // rule-based cards (no evidence images)
    let ruleModuleColsHtml = '';
    for (const feature of ['texture', 'lighting', 'depth']) {
        const score = scores[feature];
        const vote = votes[feature];
        const cardClass = vote === 1 ? 'card-red' : vote === 0 ? 'card-green' : 'card-uncertain';
        ruleModuleColsHtml += `
            <div class="module-col">
                <div class="feature-card ${cardClass}">
                    <div class="feature-title">${feature.toUpperCase()}</div>
                    <div class="feature-score">Score: ${score === null ? 'N/A' : score.toFixed(3)}</div>
                    <div class="feature-vote">Vote: ${formatVote(vote)}</div>
                </div>
            </div>`;
    }

    let ruleHtml = `
        <h2>Rule-Based Detection Results</h2>
        <p><b>Final Consensus Decision:</b> ${formatConsensus(data.final_rule_based_vote, data.final_vote_reason)}</p>
        <p><b>Tamper Threshold:</b> ${data.threshold}</p>
        <br><hr><br>
        <div class="module-evidence-row">${ruleModuleColsHtml}</div>
        ${formatConsensusNote(data.final_rule_based_vote, data.final_vote_reason, data.overall_rule_based_score)}
    `;

    let finalHtml = "";

    if (shouldShowMl) {
        // ML cards (no evidence images)
        let mlModuleColsHtml = '';
        for (const feature of ['texture', 'lighting', 'depth']) {
            const prob = moduleProbs[feature];
            const mlCardClass = prob == null ? 'card-uncertain'
                : prob >= 0.6 ? 'card-red'
                : prob <= 0.4 ? 'card-green'
                : 'card-uncertain';
            mlModuleColsHtml += `
                <div class="module-col">
                    <div class="feature-card ${mlCardClass}">
                        <div class="feature-title">${feature.toUpperCase()}</div>
                        <div class="feature-score">Probability Tampered: ${prob != null ? prob.toFixed(3) : 'N/A'}</div>
                    </div>
                </div>`;
        }

        let mlHtml = `
            <h2>ML Stacked Model Results</h2>
            <p><b>Final Prediction:</b> ${formatVote(data.ml_prediction)}</p>
        `;
        if (data.ml_probability_tampered != null) {
            mlHtml += `<p><b>Final Stacked Probability Tampered:</b> ${data.ml_probability_tampered.toFixed(3)}</p>`;
        } else {
            mlHtml += `<p><i>No probability available</i></p>`;
        }
        mlHtml += `<br><hr><br><div class="module-evidence-row">${mlModuleColsHtml}</div>`;

        finalHtml = `
            <div class="results-container">
                <div class="results-column">${ruleHtml}</div>
                <div class="results-column">${mlHtml}</div>
            </div>
        `;
    } else {
        finalHtml = `
            <div class="results-container">
                <div class="results-column">${ruleHtml}</div>
            </div>
        `;
    }

    // shared evidence section — row-based grid so images and captions align across columns
    const allVizKeys = ['shadow_overlay', 'lbp_map', 'component_overlay', 'ratio_heatmap', 'contour_overlay', 'orientation_overlay'];

    // collect flagged modules with their panel data
    const flaggedModules = [];
    for (const feature of ['texture', 'lighting', 'depth']) {
        const ruleVote = votes[feature];
        const mlProb = moduleProbs[feature];
        const mlVote = mlProb == null ? null : mlProb >= 0.6 ? 1 : mlProb <= 0.4 ? 0 : null;

        const flaggedByRule = ruleVote !== 0;
        const flaggedByMl = shouldShowMl && mlVote !== 0;
        if (!flaggedByRule && !flaggedByMl) continue;

        const moduleEvidence = evidence[feature] || {};
        const captions = evidenceCaptions[feature] || {};
        const panels = allVizKeys
            .filter(key => moduleEvidence[key])
            .map(key => ({ src: moduleEvidence[key], caption: captions[key] || '' }));

        if (panels.length > 0) {
            flaggedModules.push({ feature, label: feature.toUpperCase(), panels });
        }
    }

    if (flaggedModules.length > 0) {
        const colWidth = 'calc(33.33% - 17px)';
        const gridCols = Array(flaggedModules.length).fill(colWidth).join(' ');
        const maxPanels = Math.max(...flaggedModules.map(m => m.panels.length));

        let gridCells = '';

        // label row
        for (const m of flaggedModules) {
            gridCells += `<div class="evidence-module-label">${m.label}</div>`;
        }

        // one image row then one caption row per panel index
        for (let i = 0; i < maxPanels; i++) {
            for (const m of flaggedModules) {
                const panel = m.panels[i];
                gridCells += panel
                    ? `<img class="evidence-img" src="data:image/png;base64,${panel.src}" alt="${m.feature}">`
                    : `<div></div>`;
            }
            for (const m of flaggedModules) {
                const panel = m.panels[i];
                gridCells += panel
                    ? `<p class="evidence-caption">${panel.caption}</p>`
                    : `<div></div>`;
            }
        }

        finalHtml += `
            <div class="shared-evidence-section">
                <br>
                <h2>Supporting Evidence</h2>
                <hr>
                <div class="evidence-grid" style="grid-template-columns: ${gridCols}">
                    ${gridCells}
                </div>
            </div>
        `;
    }

    resultBox.innerHTML = finalHtml;
}
