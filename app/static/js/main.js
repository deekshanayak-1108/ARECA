// ArecaGuard Interactive Client Controller
document.addEventListener("DOMContentLoaded", () => {
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("file-input");
  const uploadForm = document.getElementById("upload-form");
  const previewContainer = document.getElementById("preview-container");
  const previewImage = document.getElementById("preview-image");
  const submitBtn = document.getElementById("submit-btn");
  const loadingOverlay = document.getElementById("loading-overlay");

  if (!dropzone || !fileInput) return;

  // Click to trigger hidden input
  dropzone.addEventListener("click", () => fileInput.click());

  // Drag and drop event handlers
  ["dragenter", "dragover"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove("dragover");
    });
  });

  dropzone.addEventListener("drop", (e) => {
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFile(files[0]);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      handleFile(e.target.files[0]);
    }
  });

  function handleFile(file) {
    const validTypes = ["image/jpeg", "image/png", "image/webp"];
    if (!validTypes.includes(file.type)) {
      alert("Please upload a valid image file (JPG, PNG, or WEBP).");
      return;
    }

    // Check size limit: 16 MB
    if (file.size > 16 * 1024 * 1024) {
      alert("File exceeds maximum allowed size of 16 MB.");
      return;
    }

    // Set file to input
    const dataTransfer = new DataTransfer();
    dataTransfer.items.add(file);
    fileInput.files = dataTransfer.files;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
      previewImage.src = e.target.result;
      previewContainer.style.display = "block";
      if (submitBtn) submitBtn.style.display = "inline-block";
    };
    reader.readAsDataURL(file);
  }

  // Handle submit animation
  if (uploadForm) {
    uploadForm.addEventListener("submit", () => {
      if (fileInput.files.length === 0) {
        alert("Please select or drop an arecanut image first.");
        return;
      }
      if (loadingOverlay) loadingOverlay.style.display = "block";
      if (submitBtn) {
        submitBtn.disabled = true;
        submitBtn.innerText = "Analyzing Symptoms...";
      }
    });
  }
});
