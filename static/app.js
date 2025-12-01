// DOM elements
const fileInput = document.getElementById("fileInput");
const preview = document.getElementById("preview");
const upload = document.getElementById("customUpload");
const analyzeButton = document.getElementById("analyzeButton");
const mlButton = document.getElementById("mlButton");
const dlButton = document.getElementById("dlButton");
const resultBox = document.getElementById("result");

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
    }

});

// display voting result
function formatVote(v) {
    if (v === 0) 
        return `<span class="real">Real</span>`;
    if (v === 1)
        return `<span class="tampered"Tampered</span>`;
    
    return `<span class="unknown">Uncertain</span>`;
}

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

    // LEFT SIDE is rule-based results
    let ruleHtml = `
        <h2>Rule-Based Detection Results</h2>
        <p><b>Final Consensus Decision:</b> ${formatVote(data.final_rule_based_vote)}</p>
        <p><b>Tamper Threshold:</b> ${data.threshold}</p>
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
        let mlHtml = `
            <h2>ML Model (Logistic Regression) Results</h2>
            <p><b>Prediction:</b> ${formatVote(data.ml_prediction)}</p>
        `;

        if (data.ml_probability_tampered !== null && data.ml_probability_tampered !== undefined) {
            mlHtml += `
                <p><b>Probability Tampered:</b> ${data.ml_probability_tampered.toFixed(3)}</p>
            `;
        } else {
            mlHtml += `<p><i>No probability available</i></p>`;
        }

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

// // Display score
//     document.getElementById("result").innerHTML =
//     `<b>Tamper Score:</b> ${data.tamper_score.toFixed(2)}<br>
//         <small>(0 = normal, 1 = likely tampered)</small>`;
