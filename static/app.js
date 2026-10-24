const $ = (selector) => document.querySelector(selector);
const systemStatus = $("#system-status");
const modelSelect = $("#model-select");
const singlePanel = $("#single-panel");
const batchPanel = $("#batch-panel");
const singleModeButton = $("#single-mode");
const batchModeButton = $("#batch-mode");
const reviewInput = $("#review-input");
const errorMessage = $("#error-message");
const loadingMessage = $("#loading-message");
const singleResult = $("#single-result");
const batchResult = $("#batch-result");
const trainingLog = $("#training-log");
let previousBatch = [];

async function apiRequest(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", ...options.headers },
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const message = typeof payload.detail === "string"
      ? payload.detail
      : `Request failed (${response.status}).`;
    throw new Error(message);
  }
  return payload;
}

function setStatus(text, state) {
  systemStatus.className = `system-status ${state}`;
  systemStatus.lastElementChild.textContent = text;
}

function setMode(mode) {
  const isSingle = mode === "single";
  singlePanel.classList.toggle("hidden", !isSingle);
  batchPanel.classList.toggle("hidden", isSingle);
  singleModeButton.classList.toggle("active", isSingle);
  batchModeButton.classList.toggle("active", !isSingle);
  singleModeButton.setAttribute("aria-selected", String(isSingle));
  batchModeButton.setAttribute("aria-selected", String(!isSingle));
  clearError();
}

function clearError() {
  errorMessage.textContent = "";
  errorMessage.classList.add("hidden");
}

function showError(error) {
  errorMessage.textContent = error.message || String(error);
  errorMessage.classList.remove("hidden");
}

function setLoading(isLoading, text = "Preparing AI and analyzing your review…") {
  loadingMessage.lastElementChild.textContent = text;
  loadingMessage.classList.toggle("hidden", !isLoading);
  $("#analyze-button").disabled = isLoading;
  $("#batch-button").disabled = isLoading;
}

function sentimentLabel(value) {
  return value === "positive" ? "Positive" : "Negative";
}

function renderSingleResult(result) {
  const sentiment = sentimentLabel(result.sentiment);
  $("#result-title").textContent = sentiment;
  $("#result-model").textContent = `${result.model} · ${result.device}`;
  $("#result-icon").className = `result-icon ${result.sentiment}`;
  $("#result-icon").textContent = result.sentiment === "positive" ? "↗" : "↘";
  const scores = result.scores && Object.keys(result.scores).length
    ? result.scores
    : {
        positive: result.sentiment === "positive" ? result.confidence : 1 - result.confidence,
        negative: result.sentiment === "negative" ? result.confidence : 1 - result.confidence,
      };
  for (const label of ["positive", "negative"]) {
    const percentage = Math.round(scores[label] * 100);
    $(`#${label}-score`).textContent = `${percentage}%`;
    $(`#${label}-bar`).style.width = `${percentage}%`;
  }
  const topScore = `${Math.round(result.confidence * 100)}%`;
  $("#result-note").textContent = result.confidence < 0.6
    ? `The model leans ${sentiment.toLowerCase()}, but its top score is ${topScore}. The result is uncertain; mixed opinions, sarcasm, or unfamiliar topics can be difficult to classify.`
    : `The model favors an overall ${sentiment.toLowerCase()} tone (top score ${topScore}). This is not a phrase-by-phrase explanation or a guarantee.`;
  singleResult.classList.remove("hidden");
  batchResult.classList.add("hidden");
}

async function analyzeSingle() {
  clearError();
  singleResult.classList.add("hidden");
  const review = reviewInput.value.trim();
  if (!review) {
    showError(new Error("Write or choose a review before analyzing."));
    reviewInput.focus();
    return;
  }
  setLoading(true);
  try {
    const result = await apiRequest("/api/predict", {
      method: "POST",
      body: JSON.stringify({ review, model: modelSelect.value }),
    });
    renderSingleResult(result);
    singleResult.scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch (error) {
    showError(error);
  } finally {
    setLoading(false);
  }
}

function parseCsvRows(text) {
  const rows = [];
  let row = [];
  let cell = "";
  let insideQuotes = false;
  for (let index = 0; index < text.length; index += 1) {
    const character = text[index];
    if (character === '"' && insideQuotes && text[index + 1] === '"') {
      cell += '"';
      index += 1;
    } else if (character === '"') {
      insideQuotes = !insideQuotes;
    } else if (character === "," && !insideQuotes) {
      row.push(cell);
      cell = "";
    } else if ((character === "\n" || character === "\r") && !insideQuotes) {
      if (character === "\r" && text[index + 1] === "\n") index += 1;
      row.push(cell);
      if (row.some((value) => value.trim())) rows.push(row);
      row = [];
      cell = "";
    } else {
      cell += character;
    }
  }
  row.push(cell);
  if (row.some((value) => value.trim())) rows.push(row);
  return rows;
}

function reviewsFromCsv(text) {
  const rows = parseCsvRows(text);
  if (!rows.length) return [];
  const header = rows[0].map((cell) => cell.trim().toLowerCase());
  const headerIndex = header.findIndex((cell) =>
    ["review", "text", "content", "sentence"].includes(cell),
  );
  const columnIndex = headerIndex >= 0 ? headerIndex : 0;
  const body = headerIndex >= 0 ? rows.slice(1) : rows;
  return body.map((row) => (row[columnIndex] || "").trim()).filter(Boolean);
}

async function analyzeBatch() {
  clearError();
  batchResult.classList.add("hidden");
  const reviews = $("#batch-input").value.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
  if (!reviews.length) {
    showError(new Error("Paste reviews, one per line, or upload a text/CSV file."));
    return;
  }
  if (reviews.length > 100) {
    showError(new Error("The batch limit is 100 reviews. Split the file into smaller batches."));
    return;
  }
  setLoading(true, `Analyzing ${reviews.length} reviews…`);
  try {
    const payload = await apiRequest("/api/predict/batch", {
      method: "POST",
      body: JSON.stringify({ reviews, model: modelSelect.value }),
    });
    previousBatch = payload.results;
    renderBatchResults(payload.results);
    batchResult.scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch (error) {
    showError(error);
  } finally {
    setLoading(false);
  }
}

function renderBatchResults(results) {
  const positive = results.filter((result) => result.sentiment === "positive").length;
  const negative = results.length - positive;
  $("#batch-summary-title").textContent = `${results.length} reviews · ${positive} positive · ${negative} negative`;
  const body = $("#batch-table-body");
  body.replaceChildren();
  results.forEach((result, index) => {
    const row = document.createElement("tr");
    const number = document.createElement("td");
    number.textContent = String(index + 1);
    const review = document.createElement("td");
    review.className = "review-cell";
    review.textContent = result.review;
    const sentiment = document.createElement("td");
    sentiment.className = `label-${result.sentiment}`;
    sentiment.textContent = sentimentLabel(result.sentiment);
    const score = document.createElement("td");
    score.textContent = `${Math.round(result.confidence * 100)}%`;
    row.append(number, review, sentiment, score);
    body.append(row);
  });
  batchResult.classList.remove("hidden");
}

function csvEscape(value) {
  return `"${String(value).replaceAll('"', '""')}"`;
}

function downloadBatch() {
  const rows = [
    ["review", "sentiment", "confidence"],
    ...previousBatch.map((result) => [
      result.review,
      result.sentiment,
      result.confidence.toFixed(4),
    ]),
  ];
  const csv = rows.map((row) => row.map(csvEscape).join(",")).join("\r\n");
  const link = document.createElement("a");
  link.href = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
  link.download = "sentiment-analysis.csv";
  link.click();
  URL.revokeObjectURL(link.href);
}

function drawLearningCurve(history) {
  const container = $("#loss-chart");
  if (!history || !history.train_loss?.length) return;
  const width = 520;
  const height = 125;
  const padding = { top: 12, right: 9, bottom: 16, left: 12 };
  const values = [...history.train_loss, ...history.validation_loss];
  const maxValue = Math.max(...values, 0.01);
  const minValue = Math.min(...values, 0);
  const x = (index, length) => padding.left + (length < 2 ? 0 : index * (width - padding.left - padding.right) / (length - 1));
  const y = (value) => height - padding.bottom - (value - minValue) * (height - padding.top - padding.bottom) / (maxValue - minValue || 1);
  const points = (series) => series.map((value, index) => `${x(index, series.length)},${y(value)}`).join(" ");
  const grid = [0.25, 0.5, 0.75].map((fraction) => {
    const position = padding.top + fraction * (height - padding.top - padding.bottom);
    return `<line x1="${padding.left}" x2="${width - padding.right}" y1="${position}" y2="${position}" stroke="#27344b" stroke-dasharray="3 5"/>`;
  }).join("");
  container.innerHTML = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Training and validation loss by epoch">${grid}<polyline points="${points(history.train_loss)}" fill="none" stroke="#ad8bff" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/><polyline points="${points(history.validation_loss)}" fill="none" stroke="#74d8d0" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
}

function drawConfusionMatrix(metrics) {
  const container = $("#confusion-matrix");
  if (!metrics) return;
  const matrix = metrics.confusion_matrix;
  if (!Array.isArray(matrix) || matrix.length !== 2) return;
  container.innerHTML = `
    <span></span><span class="matrix-caption">PRED. NEGATIVE</span><span class="matrix-caption">PRED. POSITIVE</span>
    <span class="matrix-caption">ACTUAL NEG.</span><span class="matrix-cell correct">${matrix[0][0]}</span><span class="matrix-cell incorrect">${matrix[0][1]}</span>
    <span class="matrix-caption">ACTUAL POS.</span><span class="matrix-cell incorrect">${matrix[1][0]}</span><span class="matrix-cell correct">${matrix[1][1]}</span>`;
}

function renderMetrics(metrics) {
  if (!metrics) {
    $("#evaluation-badge").textContent = "Train the project CNN to populate evaluation";
    $("#metric-interpretation-text").textContent = "Train the Text CNN to evaluate it on the held-out test set.";
    return;
  }
  $("#metric-accuracy").textContent = `${(metrics.accuracy * 100).toFixed(1)}%`;
  $("#metric-precision").textContent = `${(metrics.macro_precision * 100).toFixed(1)}%`;
  $("#metric-recall").textContent = `${(metrics.macro_recall * 100).toFixed(1)}%`;
  $("#metric-f1").textContent = `${(metrics.macro_f1 * 100).toFixed(1)}%`;
  const matrix = metrics.confusion_matrix;
  if (Array.isArray(matrix) && matrix.length === 2 && matrix.every((row) => Array.isArray(row) && row.length === 2)) {
    const [[trueNegative, falsePositive], [falseNegative, truePositive]] = matrix;
    const total = trueNegative + falsePositive + falseNegative + truePositive;
    const negativeRecall = trueNegative + falsePositive
      ? trueNegative / (trueNegative + falsePositive)
      : 0;
    const positiveRecall = falseNegative + truePositive
      ? truePositive / (falseNegative + truePositive)
      : 0;
    const recallDifference = Math.abs(negativeRecall - positiveRecall);
    let classSummary = "It found the two sentiment classes at similar rates.";
    if (recallDifference >= 0.05) {
      const strongerClass = negativeRecall > positiveRecall ? "negative" : "positive";
      const weakerClass = strongerClass === "negative" ? "positive" : "negative";
      const strongerRecall = Math.max(negativeRecall, positiveRecall);
      const weakerRecall = Math.min(negativeRecall, positiveRecall);
      classSummary = `It recognizes ${strongerClass} reviews more reliably (${(strongerRecall * 100).toFixed(0)}% recall) than ${weakerClass} reviews (${(weakerRecall * 100).toFixed(0)}% recall).`;
    }
    $("#metric-interpretation-text").textContent =
      `It correctly classified ${Math.round(metrics.accuracy * total)} of ${total} held-out reviews. ${classSummary} Treat this small test set as an educational baseline, not a guarantee.`;
  }
  const sampleCount = matrix?.flat().reduce((sum, count) => sum + count, 0);
  const epochLabel = metrics.best_epoch
    ? `checkpoint selected at epoch ${metrics.best_epoch}`
    : "held-out test set";
  $("#evaluation-badge").textContent = `Text CNN · ${sampleCount || "—"} test reviews · ${epochLabel}`;
  drawConfusionMatrix(metrics);
}

async function refreshInsights() {
  try {
    const result = await apiRequest("/api/metrics");
    renderMetrics(result.test_metrics);
    drawLearningCurve(result.training_history);
  } catch (error) {
    $("#evaluation-badge").textContent = "Evaluation unavailable";
  }
}

async function checkHealth() {
  try {
    const health = await apiRequest("/api/health");
    setStatus(`API ready · ${health.device}`, "ready");
    if (!health.cnn_checkpoint_available) {
      $("#evaluation-badge").textContent = "Project CNN not trained yet";
    }
  } catch {
    setStatus("API unavailable", "error");
  }
}

async function startTraining() {
  clearError();
  trainingLog.textContent = "";
  trainingLog.classList.remove("hidden");
  const button = $("#train-button");
  button.disabled = true;
  let jobId;
  try {
    const maxTrain = Number($("#train-limit").value);
    const maxTest = maxTrain > 0 ? Math.min(200, Math.max(2, Math.round(maxTrain / 9))) : null;
    const started = await apiRequest("/api/train", {
      method: "POST",
      body: JSON.stringify({
        epochs: Number($("#train-epochs").value),
        batch_size: Number($("#train-batch-size").value),
        learning_rate: Number($("#train-learning-rate").value),
        max_train_samples: maxTrain > 0 ? maxTrain : null,
        max_test_samples: maxTest,
      }),
    });
    jobId = started.job_id;
  } catch (error) {
    showError(error);
    button.disabled = false;
    return;
  }

  const poll = async () => {
    try {
      const status = await apiRequest(`/api/train/${jobId}`);
      trainingLog.textContent = status.log || "Preparing training run…";
      trainingLog.scrollTop = trainingLog.scrollHeight;
      if (status.status === "running") {
        window.setTimeout(poll, 1500);
        return;
      }
      button.disabled = false;
      if (status.status === "failed") {
        showError(new Error("Training failed. Review the training log for details."));
        return;
      }
      setStatus("Model updated", "ready");
      await refreshInsights();
    } catch (error) {
      button.disabled = false;
      showError(error);
    }
  };
  await poll();
}

singleModeButton.addEventListener("click", () => setMode("single"));
batchModeButton.addEventListener("click", () => setMode("batch"));
$("#analyze-button").addEventListener("click", analyzeSingle);
$("#batch-button").addEventListener("click", analyzeBatch);
$("#train-button").addEventListener("click", startTraining);
$("#download-csv").addEventListener("click", downloadBatch);
reviewInput.addEventListener("input", () => {
  $("#character-count").textContent = `${reviewInput.value.length.toLocaleString()} / 10,000`;
});
$("#batch-input").addEventListener("input", () => {
  const count = $("#batch-input").value.split(/\r?\n/).filter((line) => line.trim()).length;
  $("#batch-count").textContent = `${count} reviews · max 100`;
});
document.querySelectorAll(".sample-chip").forEach((button) => {
  button.addEventListener("click", () => {
    reviewInput.value = button.dataset.example;
    reviewInput.dispatchEvent(new Event("input"));
    reviewInput.focus();
  });
});
$("#batch-file").addEventListener("change", async (event) => {
  const file = event.target.files?.[0];
  if (!file) return;
  clearError();
  try {
    const text = await file.text();
    const reviews = file.name.toLowerCase().endsWith(".csv")
      ? reviewsFromCsv(text)
      : text.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
    if (reviews.length > 100) throw new Error("The uploaded file has more than 100 reviews.");
    $("#batch-input").value = reviews.join("\n");
    $("#batch-input").dispatchEvent(new Event("input"));
    setMode("batch");
  } catch (error) {
    showError(error);
  } finally {
    event.target.value = "";
  }
});
reviewInput.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") analyzeSingle();
});
$("#batch-input").addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") analyzeBatch();
});
modelSelect.addEventListener("change", () => {
  singleResult.classList.add("hidden");
  batchResult.classList.add("hidden");
});

checkHealth();
refreshInsights();
