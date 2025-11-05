// ============================================================
// API Configuration
// ============================================================
const API_BASE = "http://127.0.0.1:8000";

// ============================================================
// Element Selectors
// ============================================================
const $ = (sel) => document.querySelector(sel);

// Theme & Tabs
const themeToggle = $("#themeToggle");
const html = document.documentElement;
const tabBtns = document.querySelectorAll(".tab-btn");
const tabContents = document.querySelectorAll(".tab-content");

// File Inputs & Previews
const embedFileInput = $("#embedFile");
const embedFileName = $("#embedFileName");
const embedPreview = $("#embedPreview");
const extractFileInput = $("#extractFile");
const extractFileName = $("#extractFileName");
const extractPreview = $("#extractPreview");

// Forms & Results
const messageTextarea = $("#message");
const charCount = $("#charCount");
const algorithmSelect = $("#algorithm");
const algoInfo = $("#algoInfo");
const loadingOverlay = $("#loadingOverlay");
const embedForm = $("#embedForm");
const embedResults = $("#embedResults");
const extractForm = $("#extractForm");
const extractResults = $("#extractResults");
const extractedMessage = $("#extractedMessage");

// Metrics
const mseValue = $("#mseValue");
const psnrValue = $("#psnrValue");
const ssimValue = $("#ssimValue");
const timeValue = $("#timeValue");

// ============================================================
// Theme Management
// ============================================================
const savedTheme = localStorage.getItem("theme") || "dark";
html.setAttribute("data-color-scheme", savedTheme);

if (themeToggle) {
  themeToggle.addEventListener("click", () => {
    const current = html.getAttribute("data-color-scheme");
    const next = current === "dark" ? "light" : "dark";
    html.setAttribute("data-color-scheme", next);
    localStorage.setItem("theme", next);
  });
}

// ============================================================
// Tab Switching
// ============================================================
if (tabBtns && tabBtns.length) {
  tabBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      const tabName = btn.getAttribute("data-tab");
      tabBtns.forEach((b) => b.classList.remove("active"));
      tabContents.forEach((c) => c.classList.remove("active"));
      btn.classList.add("active");
      const pane = document.getElementById(`${tabName}-tab`);
      if (pane) pane.classList.add("active");
    });
  });
}

// ============================================================
// Notifications + Loading
// ============================================================
function showNotification(msg, type = "info") {
  const n = document.createElement("div");
  n.textContent = msg;
  n.style.cssText = `
    position:fixed;bottom:24px;right:24px;padding:16px 24px;
    border-radius:12px;color:white;z-index:9999;
    font-weight:500;max-width:400px;
    background:${
      type === "success"
        ? "#2dd4bf"
        : type === "error"
        ? "#ef4444"
        : type === "warning"
        ? "#f59e0b"
        : "#64748b"
    };
    box-shadow:0 4px 16px rgba(0,0,0,0.3);
    animation:fadeIn .3s ease-out;
  `;
  document.body.appendChild(n);
  setTimeout(() => {
    n.style.animation = "fadeOut .3s ease-in";
    setTimeout(() => n.remove(), 300);
  }, 2500);
}

function showLoading() {
  loadingOverlay.classList.add("show");
}
function hideLoading() {
  loadingOverlay.classList.remove("show");
}

// ============================================================
// File Validation & Preview
// ============================================================
function validateImageFile(file) {
  const valid = ["image/jpeg", "image/png", "image/bmp", "image/webp"];
  if (!valid.includes(file.type)) throw new Error("Invalid image type!");
  if (file.size > 50 * 1024 * 1024)
    throw new Error("File too large (max 50MB)");
}

function handleFileInput(input, label, preview) {
  input.addEventListener("change", () => {
    const file = input.files?.[0];
    if (!file) {
      label.textContent = "No file selected";
      preview.innerHTML = "";
      return;
    }
    label.textContent = file.name;
    try {
      validateImageFile(file);
      const reader = new FileReader();
      reader.onload = (e) => {
        preview.innerHTML = `<img src="${e.target.result}" style="max-width:100%;border-radius:10px;">`;
        preview.classList.add("show");
      };
      reader.readAsDataURL(file);
    } catch (e) {
      alert(e.message);
      input.value = "";
      label.textContent = "No file selected";
    }
  });
}
handleFileInput(embedFileInput, embedFileName, embedPreview);
handleFileInput(extractFileInput, extractFileName, extractPreview);

// ============================================================
// Character Counter
// ============================================================
messageTextarea?.addEventListener("input", () => {
  const len = messageTextarea.value.length;
  charCount.textContent = len;
});

// ============================================================
// Embed Logic (fixed)
// ============================================================
let downloadPath = "";

embedForm?.addEventListener("submit", async (e) => {
  e.preventDefault();

  const file = embedFileInput.files?.[0];
  const message = messageTextarea.value.trim();
  const algorithm = algorithmSelect.value;

  if (!file) return alert("⚠️ Select an image");
  if (!message) return alert("⚠️ Enter a secret message");
  if (!algorithm) return alert("⚠️ Select an algorithm");

  const formData = new FormData();
  formData.append("file", file);
  formData.append("message", message);
  formData.append("algorithm", algorithm);

  showLoading();

  try {
    const res = await fetch(`${API_BASE}/embed/`, {
      method: "POST",
      body: formData,
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    const data = await res.json();
    console.log("✅ Embed response:", data);

    const m = data.metrics || {};
    mseValue.textContent = m.MSE ? m.MSE.toFixed(4) : "-";
    psnrValue.textContent = m.PSNR ? m.PSNR.toFixed(2) + " dB" : "-";
    ssimValue.textContent = m.SSIM ? m.SSIM.toFixed(4) : "-";
    timeValue.textContent = m.Time ? m.Time.toFixed(3) + " s" : "-";

    downloadPath = data.output_path;

    embedResults.style.display = "block";

    // Show stego preview
    const previewDiv = document.createElement("div");
    previewDiv.innerHTML = `<img src="${API_BASE}${downloadPath}?t=${Date.now()}" style="max-width:100%;border-radius:10px;margin-top:15px;">`;
    embedResults.appendChild(previewDiv);

    // Auto-download stego image
    const imgRes = await fetch(`${API_BASE}${downloadPath}`);
    if (imgRes.ok) {
      const blob = await imgRes.blob();
      const a = document.createElement("a");
      const url = URL.createObjectURL(blob);
      a.href = url;
      a.download = downloadPath.split("/").pop();
      a.click();
      URL.revokeObjectURL(url);
    }

    showNotification(
      "✅ Message embedded and stego image downloaded!",
      "success"
    );
  } catch (err) {
    console.error(err);
    alert("❌ Error embedding message: " + err.message);
    showNotification("Embedding failed", "error");
  } finally {
    hideLoading();
  }
});

// ============================================================
// Extract Logic (fixed)
// ============================================================
extractForm?.addEventListener("submit", async (e) => {
  e.preventDefault();

  const file = extractFileInput.files?.[0];
  const algorithm = $("#extractAlgorithm")?.value || "";

  if (!file) return alert("⚠️ Select a stego image");
  if (!algorithm) return alert("⚠️ Select an extraction algorithm");

  const formData = new FormData();
  formData.append("file", file);
  formData.append("algorithm", algorithm);

  showLoading();

  try {
    const res = await fetch(`${API_BASE}/extract/`, {
      method: "POST",
      body: formData,
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    const data = await res.json();
    console.log("✅ Extract response:", data);

    const msg = data.message || "No hidden message found";
    extractedMessage.textContent = msg;
    extractResults.style.display = "block";
    extractResults.scrollIntoView({ behavior: "smooth" });

    if (msg && msg !== "No hidden message found")
      showNotification("✅ Message extracted successfully!", "success");
    else showNotification("⚠️ No hidden message found", "warning");

    extractFileInput.value = ""; // Reset file input
  } catch (err) {
    console.error("Extraction error:", err);
    alert("❌ Extraction failed: " + err.message);
    showNotification("Extraction failed", "error");
  } finally {
    hideLoading();
  }
});

// ============================================================
// Copy Extracted Message
// ============================================================
$("#copyBtn")?.addEventListener("click", async () => {
  const msg = extractedMessage.textContent.trim();
  if (!msg) return alert("⚠️ Nothing to copy");
  try {
    await navigator.clipboard.writeText(msg);
    showNotification("✅ Message copied!", "success");
  } catch {
    alert("Copy failed");
  }
});

console.log("🚀 Steganography Lab Ready!");
