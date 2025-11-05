// ============================================================
// API Configuration
// ============================================================
// Update this URL to match your FastAPI backend server
const API_BASE = "http://localhost:8000";

// ============================================================
// Theme Management
// ============================================================
const themeToggle = document.getElementById("themeToggle");
const html = document.documentElement;

// Load saved theme preference or default to dark mode
const savedTheme = localStorage.getItem("theme") || "dark";
html.setAttribute("data-color-scheme", savedTheme);

// Toggle between light and dark themes
themeToggle.addEventListener("click", () => {
	const currentTheme = html.getAttribute("data-color-scheme");
	const newTheme = currentTheme === "dark" ? "light" : "dark";
	html.setAttribute("data-color-scheme", newTheme);
	localStorage.setItem("theme", newTheme);
});

// ============================================================
// Tab Navigation
// ============================================================
const tabBtns = document.querySelectorAll(".tab-btn");
const tabContents = document.querySelectorAll(".tab-content");

tabBtns.forEach((btn) => {
	btn.addEventListener("click", () => {
		const tabName = btn.getAttribute("data-tab");

		// Remove active state from all tabs and contents
		tabBtns.forEach((b) => b.classList.remove("active"));
		tabContents.forEach((c) => c.classList.remove("active"));

		// Add active state to clicked tab and corresponding content
		btn.classList.add("active");
		document.getElementById(`${tabName}-tab`).classList.add("active");
	});
});

// ============================================================
// File Upload Handlers
// ============================================================
const embedFileInput = document.getElementById("embedFile");
const embedFileName = document.getElementById("embedFileName");
const embedPreview = document.getElementById("embedPreview");

const extractFileInput = document.getElementById("extractFile");
const extractFileName = document.getElementById("extractFileName");
const extractPreview = document.getElementById("extractPreview");

/**
 * Handle file input changes and show preview
 */
function handleFileInput(input, nameDisplay, previewDiv) {
	input.addEventListener("change", (e) => {
		const file = e.target.files[0];

		if (file) {
			// Display filename
			nameDisplay.textContent = file.name;

			// Validate file type
			if (!file.type.startsWith("image/")) {
				alert("Please select a valid image file");
				input.value = "";
				nameDisplay.textContent = "No file selected";
				previewDiv.classList.remove("show");
				return;
			}

			// Show image preview
			const reader = new FileReader();
			reader.onload = (e) => {
				previewDiv.innerHTML = `<img src="${e.target.result}" alt="Image Preview">`;
				previewDiv.classList.add("show");
			};
			reader.onerror = () => {
				alert("Error reading file");
				previewDiv.classList.remove("show");
			};
			reader.readAsDataURL(file);
		} else {
			nameDisplay.textContent = "No file selected";
			previewDiv.classList.remove("show");
			previewDiv.innerHTML = "";
		}
	});
}

// Initialize file input handlers
handleFileInput(embedFileInput, embedFileName, embedPreview);
handleFileInput(extractFileInput, extractFileName, extractPreview);

// ============================================================
// Character Counter for Message Input
// ============================================================
const messageTextarea = document.getElementById("message");
const charCount = document.getElementById("charCount");

messageTextarea.addEventListener("input", () => {
	const length = messageTextarea.value.length;
	charCount.textContent = length;

	// Optional: Warn if message is too long
	if (length > 10000) {
		charCount.style.color = "var(--color-warning)";
	} else {
		charCount.style.color = "var(--color-text-secondary)";
	}
});

// ============================================================
// Algorithm Information Display
// ============================================================
const algorithmSelect = document.getElementById("algorithm");
const algoInfo = document.getElementById("algoInfo");

const algorithmDescriptions = {
	lsb: "🔹 LSB (Least Significant Bit) - Simple and fast method that modifies the least significant bit of each pixel. Best for beginners and high-capacity embedding.",
	lsb_matching:
		"🔹 LSB Matching - Enhanced version of LSB with random ±1 adjustments. More resistant to statistical detection than standard LSB.",
	dct: "🔹 DCT (Discrete Cosine Transform) - Embeds data in frequency domain coefficients. More robust against JPEG compression and image processing.",
	dwt: "🔹 DWT (Discrete Wavelet Transform) - Uses wavelet decomposition for embedding in high-frequency subbands. Good balance between capacity and imperceptibility.",
	hybrid:
		"🔹 Hybrid (DWT + DCT) - Combines DWT and DCT for maximum robustness and quality. Best resistance to attacks but slower processing time.",
};

algorithmSelect.addEventListener("change", () => {
	const algo = algorithmSelect.value;
	if (algo && algorithmDescriptions[algo]) {
		algoInfo.textContent = algorithmDescriptions[algo];
		algoInfo.classList.add("show");
	} else {
		algoInfo.classList.remove("show");
	}
});

// ============================================================
// Loading Overlay Management
// ============================================================
const loadingOverlay = document.getElementById("loadingOverlay");

function showLoading() {
	loadingOverlay.classList.add("show");
	document.body.style.overflow = "hidden";
}

function hideLoading() {
	loadingOverlay.classList.remove("show");
	document.body.style.overflow = "";
}

// ============================================================
// Embed Form - Upload and Embed Message
// ============================================================
const embedForm = document.getElementById("embedForm");
const embedResults = document.getElementById("embedResults");

let downloadPath = "";

embedForm.addEventListener("submit", async (e) => {
	e.preventDefault();

	// Get form data
	const file = embedFileInput.files[0];
	const message = messageTextarea.value.trim();
	const algorithm = algorithmSelect.value;

	// Validation
	if (!file) {
		alert("⚠️ Please select a cover image");
		return;
	}

	if (!message) {
		alert("⚠️ Please enter a secret message");
		return;
	}

	if (!algorithm) {
		alert("⚠️ Please select an algorithm");
		return;
	}

	// Prepare form data
	const formData = new FormData();
	formData.append("file", file);
	formData.append("message", message);
	formData.append("algorithm", algorithm);

	showLoading();

	try {
		// Send request to FastAPI backend
		const response = await fetch(`${API_BASE}/embed/`, {
			method: "POST",
			body: formData,
		});

		if (!response.ok) {
			const errorData = await response.json().catch(() => ({}));
			throw new Error(
				errorData.error || `HTTP error! status: ${response.status}`
			);
		}

		const data = await response.json();

		// Display metrics
		const metrics = data.metrics;
		document.getElementById("mseValue").textContent =
			metrics.MSE !== null ? metrics.MSE.toFixed(4) : "N/A";
		document.getElementById("psnrValue").textContent =
			metrics.PSNR !== null ? `${metrics.PSNR.toFixed(2)} dB` : "N/A";
		document.getElementById("ssimValue").textContent =
			metrics.SSIM !== null ? metrics.SSIM.toFixed(4) : "N/A";
		document.getElementById("timeValue").textContent =
			metrics.Time !== null ? `${metrics.Time.toFixed(3)} s` : "N/A";

		// Store download path
		downloadPath = data.output_path;

		// Show results section
		embedResults.style.display = "block";
		embedResults.scrollIntoView({ behavior: "smooth", block: "nearest" });

		// Success notification
		showNotification("✅ Message embedded successfully!", "success");
	} catch (error) {
		console.error("Embedding error:", error);
		alert(
			`❌ Error embedding message: ${error.message}\n\nPlease check:\n- Backend server is running\n- Image file is valid\n- Message is not too large`
		);
	} finally {
		hideLoading();
	}
});

// ============================================================
// Download Button - Download Stego Image
// ============================================================
const downloadBtn = document.getElementById("downloadBtn");

downloadBtn.addEventListener("click", async () => {
	if (!downloadPath) {
		alert("⚠️ No file available to download");
		return;
	}

	try {
		showLoading();

		// Fetch the stego image from backend
		const response = await fetch(`${API_BASE}${downloadPath}`);

		if (!response.ok) {
			throw new Error(`Download failed: ${response.status}`);
		}

		const blob = await response.blob();

		// Create download link
		const url = window.URL.createObjectURL(blob);
		const a = document.createElement("a");
		a.href = url;
		a.download = downloadPath.split("/").pop() || "stego_image.png";
		document.body.appendChild(a);
		a.click();

		// Cleanup
		window.URL.revokeObjectURL(url);
		document.body.removeChild(a);

		showNotification("✅ Image downloaded successfully!", "success");
	} catch (error) {
		console.error("Download error:", error);
		alert(`❌ Error downloading file: ${error.message}`);
	} finally {
		hideLoading();
	}
});

// ============================================================
// Extract Form - Extract Hidden Message
// ============================================================
const extractForm = document.getElementById("extractForm");
const extractResults = document.getElementById("extractResults");
const extractedMessage = document.getElementById("extractedMessage");

extractForm.addEventListener("submit", async (e) => {
	e.preventDefault();

	// Get form data
	const file = extractFileInput.files[0];
	const algorithm = document.getElementById("extractAlgorithm").value;

	// Validation
	if (!file) {
		alert("⚠️ Please select a stego image");
		return;
	}

	if (!algorithm) {
		alert("⚠️ Please select an extraction algorithm");
		return;
	}

	// Prepare form data
	const formData = new FormData();
	formData.append("file", file);
	formData.append("algorithm", algorithm);

	showLoading();

	try {
		// Send request to FastAPI backend
		const response = await fetch(`${API_BASE}/extract/`, {
			method: "POST",
			body: formData,
		});

		if (!response.ok) {
			const errorData = await response.json().catch(() => ({}));
			throw new Error(
				errorData.error || `HTTP error! status: ${response.status}`
			);
		}

		const data = await response.json();

		// Display extracted message
		const message = data.message || "No message found";
		extractedMessage.textContent = message;

		// Show results section
		extractResults.style.display = "block";
		extractResults.scrollIntoView({ behavior: "smooth", block: "nearest" });

		// Success notification
		if (message !== "No message found") {
			showNotification("✅ Message extracted successfully!", "success");
		} else {
			showNotification("⚠️ No hidden message detected", "warning");
		}
	} catch (error) {
		console.error("Extraction error:", error);
		alert(
			`❌ Error extracting message: ${error.message}\n\nPlease check:\n- Backend server is running\n- Image contains embedded message\n- Correct algorithm is selected`
		);
	} finally {
		hideLoading();
	}
});

// ============================================================
// Copy to Clipboard Button
// ============================================================
const copyBtn = document.getElementById("copyBtn");

copyBtn.addEventListener("click", async () => {
	const message = extractedMessage.textContent;

	if (!message || message === "No message found") {
		alert("⚠️ No message to copy");
		return;
	}

	try {
		await navigator.clipboard.writeText(message);

		// Visual feedback
		const originalHTML = copyBtn.innerHTML;
		copyBtn.innerHTML = `
            <svg width="18" height="18" fill="none" stroke="currentColor" stroke-width="2">
                <polyline points="20 6 9 17 4 12"></polyline>
            </svg>
            Copied!
        `;
		copyBtn.style.background = "var(--color-success)";

		setTimeout(() => {
			copyBtn.innerHTML = originalHTML;
			copyBtn.style.background = "";
		}, 2000);

		showNotification("✅ Message copied to clipboard!", "success");
	} catch (err) {
		console.error("Copy failed:", err);

		// Fallback for older browsers
		const textArea = document.createElement("textarea");
		textArea.value = message;
		textArea.style.position = "fixed";
		textArea.style.left = "-999999px";
		document.body.appendChild(textArea);
		textArea.select();

		try {
			document.execCommand("copy");
			showNotification("✅ Message copied to clipboard!", "success");
		} catch (err2) {
			alert("❌ Failed to copy to clipboard");
		}

		document.body.removeChild(textArea);
	}
});

// ============================================================
// Notification System
// ============================================================
function showNotification(message, type = "info") {
	// Remove existing notifications
	const existing = document.querySelector(".notification");
	if (existing) {
		existing.remove();
	}

	// Create notification element
	const notification = document.createElement("div");
	notification.className = `notification notification--${type}`;
	notification.textContent = message;

	// Add styles
	Object.assign(notification.style, {
		position: "fixed",
		bottom: "24px",
		right: "24px",
		padding: "16px 24px",
		background:
			type === "success"
				? "var(--color-success)"
				: type === "warning"
				? "var(--color-warning)"
				: "var(--color-info)",
		color: "white",
		borderRadius: "var(--radius-base)",
		boxShadow: "var(--shadow-lg)",
		zIndex: "10000",
		fontWeight: "500",
		fontSize: "var(--font-size-base)",
		animation: "slideInRight 0.3s ease-out",
		maxWidth: "400px",
		wordWrap: "break-word",
	});

	document.body.appendChild(notification);

	// Auto remove after 3 seconds
	setTimeout(() => {
		notification.style.animation = "slideOutRight 0.3s ease-in";
		setTimeout(() => notification.remove(), 300);
	}, 3000);
}

// Add notification animations to page
if (!document.getElementById("notification-styles")) {
	const style = document.createElement("style");
	style.id = "notification-styles";
	style.textContent = `
        @keyframes slideInRight {
            from {
                transform: translateX(400px);
                opacity: 0;
            }
            to {
                transform: translateX(0);
                opacity: 1;
            }
        }
        
        @keyframes slideOutRight {
            from {
                transform: translateX(0);
                opacity: 1;
            }
            to {
                transform: translateX(400px);
                opacity: 0;
            }
        }
    `;
	document.head.appendChild(style);
}

// ============================================================
// Error Handling - Check Backend Connection
// ============================================================
async function checkBackendConnection() {
	try {
		const response = await fetch(`${API_BASE}/`, { method: "HEAD" });
		return response.ok;
	} catch (error) {
		return false;
	}
}

// Optional: Check connection on page load
window.addEventListener("load", async () => {
	const isConnected = await checkBackendConnection();
	if (!isConnected) {
		console.warn("⚠️ Cannot connect to backend server at:", API_BASE);
		console.log(
			"📝 Make sure your FastAPI server is running on the correct port"
		);
	} else {
		console.log("✅ Backend server connected successfully");
	}
});

// ============================================================
// Utility Functions
// ============================================================

/**
 * Format file size for display
 */
function formatFileSize(bytes) {
	if (bytes === 0) return "0 Bytes";
	const k = 1024;
	const sizes = ["Bytes", "KB", "MB", "GB"];
	const i = Math.floor(Math.log(bytes) / Math.log(k));
	return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + " " + sizes[i];
}

/**
 * Validate image file
 */
function validateImageFile(file) {
	const validTypes = [
		"image/jpeg",
		"image/jpg",
		"image/png",
		"image/bmp",
		"image/tiff",
		"image/webp",
	];
	const maxSize = 50 * 1024 * 1024; // 50MB

	if (!validTypes.includes(file.type)) {
		throw new Error(
			"Invalid file type. Please use JPEG, PNG, BMP, TIFF, or WebP"
		);
	}

	if (file.size > maxSize) {
		throw new Error(
			`File too large. Maximum size is ${formatFileSize(maxSize)}`
		);
	}

	return true;
}

console.log("🚀 Steganography Lab initialized");
console.log("📡 API Base URL:", API_BASE);
