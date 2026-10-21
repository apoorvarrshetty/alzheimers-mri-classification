"""SYNAPSERA NeuroMemory — educational MRI classification prototype."""
from pathlib import Path
import numpy as np
import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError
from model import CLASS_NAMES, image_transform, load_model

st.set_page_config(page_title="SYNAPSERA NeuroMemory", page_icon="🧠", layout="wide")
MODEL_PATH = Path(__file__).parent / "neuromemory_resnet18_fp16.pth"

@st.cache_resource(show_spinner="Loading research model...")
def get_model(path):
    return load_model(path, device="cpu")

st.title("🧠 SYNAPSERA NeuroMemory")
st.caption("Explainable MRI slice classification · Research prototype · Not a diagnostic device")
st.warning("For research and education only. The model cannot diagnose Alzheimer's disease. Do not upload identifiable patient data or use predictions to make medical decisions.")

with st.expander("About the research", expanded=False):
    st.markdown("""
    **Three OASIS-derived dataset labels:** Non Demented, Very mild Dementia, Mild Dementia.
    Moderate Dementia was excluded because the dataset contained only two identified subjects in that class.

    **Validation:** 52 subjects; subject-level accuracy 75%, macro-F1 0.613,
    balanced accuracy 70.28%.

    **Held-out test (original FP32 checkpoint):** 52 subjects; subject-level accuracy 73.08%,
    macro-F1 0.5965, balanced accuracy 69.44%. Only 3 Mild Dementia subjects were in the
    test set. The deployed FP16 checkpoint has not yet been separately benchmarked on
    the full test set. These are internal research results, not clinical validation.

    **Image format:** Single sagittal MRI slices like those in the training dataset (JPG/PNG).
    Other scans, orientations, scanners, and patient populations are out of distribution.
    The app cannot reliably detect that an uploaded image is unsuitable.
    """)
if not MODEL_PATH.is_file():
    st.error("Model weights are missing. Add neuromemory_resnet18_fp16.pth to the repository root to run predictions.")
    st.stop()

try:
    model = get_model(str(MODEL_PATH))
except Exception as exc:
    st.error(f"Unable to load model weights: {exc}")
    st.stop()

uploaded = st.file_uploader("Upload an anonymized MRI slice (JPG or PNG)", type=["jpg", "jpeg", "png"])
if uploaded is None:
    st.info("Upload a dataset-like MRI image to explore model predictions.")
    st.stop()

try:
    image = Image.open(uploaded).convert("RGB")
    if image.width < 64 or image.height < 64:
        st.error("Image is too small. Please provide a dataset-like MRI slice.")
        st.stop()
except (UnidentifiedImageError, OSError, ValueError):
    st.error("The uploaded file could not be opened as an image.")
    st.stop()

tensor = image_transform()(image).unsqueeze(0)
with torch.inference_mode():
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

label_idx = int(np.argmax(probs))
left, right = st.columns([1, 1])
with left:
    st.image(image, caption="Uploaded MRI slice", use_container_width=True)
with right:
    st.subheader("Model output")
    st.metric("Predicted dataset label", CLASS_NAMES[label_idx])
    st.caption("Scores are uncalibrated model outputs, not medical probabilities.")
    for name, value in zip(CLASS_NAMES, probs):
        st.write(f"{name}: {value * 100:.1f}%")
        st.progress(float(value))

st.subheader("Grad-CAM: model attribution")
st.caption("Highlights regions contributing to the predicted class. It does not localize disease or prove an anatomical explanation.")
if st.button("Generate explanation"):
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        from pytorch_grad_cam.utils.image import show_cam_on_image
        target = [ClassifierOutputTarget(label_idx)]
        with GradCAM(model=model, target_layers=[model.layer4[-1]]) as cam:
            heatmap = cam(input_tensor=tensor, targets=target)[0]
        gray = tensor[0, 0].cpu().numpy() * 0.5 + 0.5
        gray = np.clip(gray, 0, 1).astype(np.float32)
        overlay = show_cam_on_image(np.repeat(gray[..., None], 3, axis=-1), heatmap, use_rgb=True)
        a, b = st.columns(2)
        a.image(heatmap, caption="Relative model attribution", clamp=True, use_container_width=True)
        b.image(overlay, caption="Attribution overlay", use_container_width=True)
    except Exception as exc:
        st.error(f"Could not generate Grad-CAM: {exc}")
