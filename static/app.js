// DOM elements
const fileInput = document.getElementById("fileInput");
const preview = document.getElementById("preview");
const upload = document.getElementById("customUpload");
const analyzeButton = document.getElementById("analyzeButton");
const mlButton = document.getElementById("mlButton");
const dlButton = document.getElementById("dlButton");
const resultBox = document.getElementById("result");
const loadingIndicator = document.getElementById("loadingIndicator");

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

function setActive(whichId) {
    mlButton.classList.toggle("active", whichId === "mlButton");
    dlButton.classList.toggle("active", whichId === "dlButton");
}
setActive(null);  // clear initial selection

mlButton.addEventListener("click", () => {
    modelChoice = "ml";
    setActive("mlButton");
    refreshAnalyzeState();
});

dlButton.addEventListener("click", () => {
    modelChoice = "dl";
    setActive("dlButton");
    refreshAnalyzeState();
});

// analyze button state
function refreshAnalyzeState() {
    const hasModel = !!modelChoice;
    const hasFile = fileInput.files.length > 0;

    analyzeButton.hidden = !hasModel;
    analyzeButton.disabled = !(hasModel && hasFile);
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

// turn rule-based numbers into a human-readable explanation
function buildRuleBasedExplanation(data) {
    const votes = data.rule_based_votes || {};
    const scores = data.rule_based_scores || {};
    const finalVote = data.final_rule_based_vote;

    // categorize features by vote
    const suspicious = [];
    const consistent = [];
    const uncertain = [];

    for (const [feature, vote] of Object.entries(votes)) {
        if (vote === 1) {
            suspicious.push(feature);
        }
        else if (vote === 0) {
            consistent.push(feature);
        }
        else {
            uncertain.push(feature);
        }
    }

    // human-readable descriptions per feature & vote type
    const featureDescriptions = {
        texture: {
            suspicious: `
                The <b>texture cue</b> suggests tampering: the shadow regions have
                texture statistics that differ strongly from the nearby lit regions,
                which is unusual for real shadows.
            `,
            real: `
                The <b>texture cue</b> looks consistent: textures inside shadows
                match nearby lit areas, as expected for real shadows.
            `
        },
        lighting: {
            suspicious: `
                The <b>lighting cue</b> suggests tampering: the detected shadows and
                brightness patterns are not well aligned with a single, plausible
                light source.
            `,
            real: `
                The <b>lighting cue</b> looks consistent with a single, plausible light 
                source and all shadows have close values in brightness ratios.
            `
        },
        depth: {
            suspicious: `
                The <b>depth cue</b> suggests tampering: the relationship between
                shadow blur (penumbra), shadow direction consistency, and the shadows
                in the image is atypical for real scenes.
            `,
            real: `
                The <b>depth cue</b> looks consistent: the way shadows appear is
                compatible with key depth features of shadows (penumbra and direction).
            `
        }
    };

    // final decision explanation
    let decisionSentence = "";
    if (finalVote === 1)  {
        decisionSentence = `
            Based on texture, lighting, and depth cues, our rule-based system 
            considers this image <b>likely tampered</b>.
        `;
    } else if (finalVote === 0) {
        decisionSentence = `
            Our rule-based checks indicate this image is <b>likely real</b>.
        `;
    } else {
        decisionSentence = `
            Our rule-based system cannot give a clear decision because several
            cues fell between an <b>uncertain</b> range: 0.45-0.65.
        `;
    }

    const parts = [decisionSentence];

    // feature-level detail
    if (suspicious.length) {
        parts.push(`
            <b>Suspicious cues:</b> ${suspicious.map(f => f.toUpperCase()).join(", ")}.    
        `);
    }

    if (consistent.length) {
        parts.push(`
            <b>Real-looking cues:</b> ${consistent.map(f => f.toUpperCase()).join(", ")}.
        `);
    }

    if (uncertain.length) {
        parts.push(`
            <b>Uncertain:</b> ${uncertain.map(f => f.toUpperCase()).join(", ")}.
        `);    
    }

    // detailed reasons based on which cues crossed thresholds
    const detailedLines = [];

    // for each suspicious cue, add a specific “this indicated tampering” line
    for (const feature of suspicious) {
        const desc = featureDescriptions[feature];
        if (desc && desc.suspicious) {
            detailedLines.push(desc.suspicious);
        }
    }

    // optional: also add “looks real” details for consistent cues
    for (const feature of consistent) {
        const desc = featureDescriptions[feature];
        if (desc && desc.real) {
            detailedLines.push(desc.real);
        }
    }

    if (detailedLines.length) {
        parts.push(detailedLines.join(" "));
    }

    return parts.join(" ");
}

// turn ML probability into a human-readable explanation
function buildMlExplanation(data) {
    const pred = data.ml_prediction;
    const prob = data.ml_probability_tampered;

    if (prob == null) {
        return `
            Our ML model made a classification, but no probability
            value was available for this run.
        `;
    }

    let baseSentence = `
        Our ML model estimates a <b>${(prob * 100).toFixed(1)}%</b> chance that
        this image is <b>tampered</b>.
    `;

    let confidenceSentence = "";
    if (prob < 0.2) {
        confidenceSentence = `
            This is a <b>strong indication of a real image</b> from our model's perspective.
        `;
    } else if (prob < 0.4) {
        confidenceSentence = `
            Our model still leans toward the image being real, but it does see a few
            mild signs that could be associated with editing.
        `;
    } else if (prob < 0.6) {
        confidenceSentence = `
            Our model is <b>uncertain</b> here. The features it sees are compatible with both
            real and manipulated images.
        `;
    } else if (prob < 0.8) {
        confidenceSentence = `
            Our model sees a <b>noticeable amount of evidence</b> that matches tampered
            examples it was trained on.
        `;
    } else {
        confidenceSentence = `
            Our model sees this as <b>highly likely tampered</b>, with patterns that are
            strongly aligned with edited images in our training data.
        `;
    }

    // align wording with the predicted class
    let consistencySentence = "";
    if (pred === 0 && prob < 0.5) {
        consistencySentence = `
            The final decision is <b>Real</b>, which matches the low tamper probability.
        `;
    } else if (pred === 1 && prob >= 0.5) {
        consistencySentence = `
            The final decision is <b>Tampered</b>, which matches the high tamper probability.
        `;
    } else {
        consistencySentence = `
            The final decision and probability are somewhat <b>borderline</b>,
            so this result should be understood with caution.
        `;
    }

    return `
        ${baseSentence} 
        ${confidenceSentence} 
        ${consistencySentence}
    `;    
}

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

    console.log("scores from server:", scores);
    console.log("votes from server:", votes);
    console.log("final vote from server:", data.final_rule_based_vote);
    console.log("ml prediction:", data.ml_prediction);
    console.log("ml probability tampered:", data.ml_probability_tampered);

    // get explanationn text for rule-based side
    const ruleExplanation = buildRuleBasedExplanation(data);

    // LEFT SIDE is rule-based results
    let ruleHtml = `
        <h2>Rule-Based Detection Results</h2>
        <p><b>Final Consensus Decision:</b> ${formatVote(data.final_rule_based_vote)}</p>
        <p><b>Tamper Threshold:</b> ${data.threshold}</p>
        <br>
        <hr>
        <br>
        <p class="explanation rule-explanation">
            ${ruleExplanation}
        </p>
        <div class="feature-row">
    `;

    for (const [feature, score] of Object.entries(scores)) {
        const vote = votes[feature];

        let cardClass = "card-uncertain"; // default vidual for card

        if (vote === 1) {
            cardClass = "card-red";
        } else if (vote === 0) {
            cardClass = "card-green";
        }

        ruleHtml += `
            <div class="feature-card ${cardClass}">
                <div class="feature-title">${feature.toUpperCase()}</div>
                <div class="feature-score">
                    Score: ${score === null ? "N/A" : score.toFixed(3)}
                </div>
                <div class="feature-vote">
                    Vote: ${formatVote(vote)}
                </div>
            </div>
        `;
    }

    ruleHtml += `</div>`;

    // if there are results to show for ML side, show them
    const shouldShowMl = 
        modelChoice === "ml" &&
        data.ml_prediction !== null &&
        data.ml_prediction !== undefined;

    let finalHtml = "";

    if (shouldShowMl) {
        // RIGHT SIDE is ML based results
        console.log("ML raw prediction:", data.ml_prediction);
        console.log("ML formatted prediction:", formatVote(data.ml_prediction));

        let mlHtml = `
            <h2>ML Model (Logistic Regression) Results</h2>
            <p><b>Prediction:</b> ${formatVote(data.ml_prediction)}</p>
        `;

        if (data.ml_probability_tampered !== null && data.ml_probability_tampered !== undefined) {
            mlHtml += `
                <p><b>Probability Tampered:</b> ${data.ml_probability_tampered.toFixed(3)}</p>
                <br>
                <hr>
                <br>
            `;
        } else {
            mlHtml += `<p><i>No probability available</i></p>`;
        }

        const mlExplanation = buildMlExplanation(data);
        mlHtml += `
            <p class="explanation ml-explanation">
                ${mlExplanation}
            </p>
        `;

        finalHtml = `
            <div class="results-container">
                <div class="results-column">
                    ${ruleHtml}
                </div>
                <div class="results-column">
                    ${mlHtml}
                </div>
            </div>
        `;
    } else {
        // only show rule based column if ml is not available
        finalHtml = `
            <div class="results-container">
                <div class="results-column">
                    ${ruleHtml}
                </div>
            </div>
        `;
    }

    resultBox.innerHTML = finalHtml;
}
