// DOM elements
const fileInput = document.getElementById("fileInput");
const preview = document.getElementById("preview");
const upload = document.getElementById("customUpload");
const analyzeButton = document.getElementById("analyzeButton");
const mlButton = document.getElementById("mlButton");
const resultBox = document.getElementById("result");
const loadingIndicator = document.getElementById("loadingIndicator");

// lightbox modal
const lightbox = document.createElement("div");
lightbox.id = "lightbox";
lightbox.innerHTML = '<img id="lightbox-img" alt=""><button id="lightbox-close">&times;</button>';
document.body.appendChild(lightbox);

const lightboxImg = document.getElementById("lightbox-img");

function openLightbox(src) {
    lightboxImg.src = src;
    lightbox.classList.add("active");
}

function closeLightbox() {
    lightbox.classList.remove("active");
    lightboxImg.src = "";
}

document.getElementById("lightbox-close").addEventListener("click", closeLightbox);
lightbox.addEventListener("click", (e) => { if (e.target === lightbox) closeLightbox(); });
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeLightbox(); });

// delegate clicks on evidence images to open lightbox
resultBox.addEventListener("click", (e) => {
    if (e.target.classList.contains("evidence-img")) {
        openLightbox(e.target.src);
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
    } else {
        loadingIndicator.classList.add("hidden");
        analyzeButton.disabled = false;
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
        contour_overlay: 'Shadow edge contours on grayscale. Real shadows from one light source have consistent edge softness (penumbra) throughout.',
        orientation_overlay: 'Long-axis orientation of each shadow. Lines pointing in roughly the same direction suggest one light source; scattered directions suggest composited shadows.'
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

    // build evidence block for one module: shown only when tampered or uncertain
    function buildEvidenceBlock(feature, vote, vizKeys) {
        if (vote === 0) return '';  // real vote: no evidence shown
        const moduleEvidence = evidence[feature] || {};
        const captions = evidenceCaptions[feature] || {};
        return vizKeys
            .filter(key => moduleEvidence[key])
            .map(key => `
                <div class="evidence-panel">
                    <img class="evidence-img" src="data:image/png;base64,${moduleEvidence[key]}" alt="${feature} ${key}">
                    <p class="evidence-caption">${captions[key] || ''}</p>
                </div>`)
            .join('');
    }

    // LEFT SIDE: rule-based results
    let moduleColsHtml = '';
    for (const [feature, score] of Object.entries(scores)) {
        const vote = votes[feature];
        const cardClass = vote === 1 ? 'card-red' : vote === 0 ? 'card-green' : 'card-uncertain';
        const evidenceHtml = buildEvidenceBlock(feature, vote, ['shadow_overlay', 'lbp_map', 'component_overlay', 'ratio_heatmap', 'contour_overlay', 'orientation_overlay']);
        moduleColsHtml += `
            <div class="module-col">
                <div class="feature-card ${cardClass}">
                    <div class="feature-title">${feature.toUpperCase()}</div>
                    <div class="feature-score">Score: ${score === null ? 'N/A' : score.toFixed(3)}</div>
                    <div class="feature-vote">Vote: ${formatVote(vote)}</div>
                </div>
                ${evidenceHtml}
            </div>`;
    }

    let ruleHtml = `
        <h2>Rule-Based Detection Results</h2>
        <p><b>Final Consensus Decision:</b> ${formatVote(data.final_rule_based_vote)}</p>
        <p><b>Tamper Threshold:</b> ${data.threshold}</p>
        <br><hr><br>
        <div class="module-evidence-row">${moduleColsHtml}</div>
    `;

    // if there are results to show for ML side, show them
    const shouldShowMl =
        modelChoice === "ml" &&
        data.ml_prediction !== null &&
        data.ml_prediction !== undefined;

    let finalHtml = "";

    if (shouldShowMl) {
        const moduleProbs = data.ml_module_probabilities || {};

        let mlModuleColsHtml = '';
        for (const feature of ['texture', 'lighting', 'depth']) {
            const prob = moduleProbs[feature];
            const mlCardClass = prob == null ? 'card-uncertain'
                : prob >= 0.6 ? 'card-red'
                : prob <= 0.4 ? 'card-green'
                : 'card-uncertain';
            const mlVote = prob == null ? null : prob >= 0.6 ? 1 : prob <= 0.4 ? 0 : null;
            const evidenceHtml = buildEvidenceBlock(feature, mlVote, ['shadow_overlay', 'lbp_map', 'component_overlay', 'ratio_heatmap', 'contour_overlay', 'orientation_overlay']);
            mlModuleColsHtml += `
                <div class="module-col">
                    <div class="feature-card ${mlCardClass}">
                        <div class="feature-title">${feature.toUpperCase()}</div>
                        <div class="feature-score">Probability Tampered: ${prob != null ? prob.toFixed(3) : 'N/A'}</div>
                    </div>
                    ${evidenceHtml}
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

    resultBox.innerHTML = finalHtml;
}
