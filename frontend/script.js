document.addEventListener("submit", function (e) {
  e.preventDefault();
  alert("Form submitted (backend will be connected later)");
});

document.getElementById("photo-upload").addEventListener("change", function(event) {
  const file = event.target.files[0];
  if (file) {
    const reader = new FileReader();
    reader.onload = function(e) {
      const preview = document.getElementById("photo-preview");
      preview.src = e.target.result;
      preview.style.display = "block";
    };
    reader.readAsDataURL(file);
  }
});
