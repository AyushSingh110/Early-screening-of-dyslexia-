import streamlit as st
import numpy as np
import torch
from PIL import Image
import logging
from typing import List, Tuple, Dict
from datetime import datetime
from utils.ocr import extract_text_from_image
from language_model.inference import predict_language_risk


from utils.predict import load_model
from utils.preprocess import transform
from utils.patchify import split_into_patches
from utils.gradcam import generate_gradcam_visualization, overlay_gradcam, GradCAM
import config


# Logging Configuration

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Constants
RISK_THRESHOLD_HIGH = 0.5
RISK_THRESHOLD_MEDIUM = 0.3
MIN_PATCHES_REQUIRED = 5
MAX_FILE_SIZE_MB = 10

# Session State Initialization
if 'model' not in st.session_state:
    st.session_state.model = None
if 'device' not in st.session_state:
    st.session_state.device = None
if 'enable_gradcam' not in st.session_state:
    st.session_state.enable_gradcam = False


# Helper Functions
def initialize_system() -> Tuple[torch.nn.Module, torch.device]:

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        with st.spinner("Loading AI model..."):
            model = load_model()
            model.to(device)
            model.eval()
        
        logger.info("Model loaded successfully")
        return model, device
    
    except Exception as e:
        logger.error(f"Failed to initialize system: {str(e)}")
        st.error(f" System initialization failed: {str(e)}")
        st.stop()

def validate_image(uploaded_file) -> bool:

    # Check file size
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        st.error(f" File size ({file_size_mb:.1f}MB) exceeds maximum limit of {MAX_FILE_SIZE_MB}MB")
        return False
    
    # Check if file can be opened
    try:
        image = Image.open(uploaded_file)
        image.verify()
        uploaded_file.seek(0)  # Reset file pointer after verify
        return True
    except Exception as e:
        st.error(f" Invalid image file: {str(e)}")
        return False

def calculate_risk_level(risk_score: float) -> Tuple[str, str, str]:

    if risk_score >= RISK_THRESHOLD_HIGH:
        return (
            "HIGH RISK",
            " The handwriting shows significant patterns associated with dyslexia. We strongly recommend professional assessment.",
            "red"
        )
    elif risk_score >= RISK_THRESHOLD_MEDIUM:
        return (
            "MODERATE RISK",
            "Some dyslexia-related patterns detected. Consider consulting with an educational specialist for further evaluation.",
            "orange"
        )
    else:
        return (
            "LOW RISK",
            " Handwriting appears typical. No strong dyslexia indicators detected at this time.",
            "green"
        )

def run_inference(
    patches: List[np.ndarray],
    model: torch.nn.Module,
    device: torch.device,
    enable_gradcam: bool = False
) -> Dict:

    dyslexic_count = 0
    predictions = []
    patch_confidences = []
    gradcam_visualizations = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        for idx, patch in enumerate(patches):
            # Update progress
            progress = (idx + 1) / len(patches)
            progress_bar.progress(progress)
            status_text.text(f"Analyzing patch {idx + 1}/{len(patches)}...")
            
            # Preprocess patch
            patch_img = Image.fromarray(patch).convert("RGB")
            input_tensor = transform(patch_img).unsqueeze(0).to(device)
            
            # Inference
            with torch.no_grad():
                output = model(input_tensor)
                prob = output.item()
                predictions.append(prob)
                patch_confidences.append(prob)
                
                if prob > 0.5:
                    dyslexic_count += 1
            
            # Generate Grad-CAM for high-confidence patches (both dyslexic and normal)
            if enable_gradcam:
                # Include patches with confidence > 0.6 (dyslexic) or < 0.4 (clearly normal)
                if prob > 0.6 or prob < 0.4:
                    try:
                        cam, overlay = generate_gradcam_visualization(
                            model, patch, input_tensor, device
                        )
                        if overlay is not None:
                            prediction_label = "Dyslexic" if prob > 0.5 else "Normal"
                            gradcam_visualizations.append({
                                'patch_idx': idx,
                                'confidence': prob,
                                'prediction': prediction_label,
                                'original': patch,
                                'overlay': overlay,
                                'cam': cam
                            })
                    except Exception as e:
                        logger.warning(f"Grad-CAM failed for patch {idx}: {str(e)}")
        
        progress_bar.empty()
        status_text.empty()
        
        # Calculate statistics
        risk_score = dyslexic_count / len(patches)
        avg_confidence = np.mean(predictions)
        max_confidence = np.max(predictions)
        min_confidence = np.min(predictions)
        
        return {
            'total_patches': len(patches),
            'dyslexic_patches': dyslexic_count,
            'normal_patches': len(patches) - dyslexic_count,
            'risk_score': risk_score,
            'avg_confidence': avg_confidence,
            'max_confidence': max_confidence,
            'min_confidence': min_confidence,
            'predictions': predictions,
            'gradcam_visualizations': gradcam_visualizations
        }
    
    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")
        st.error(f" Analysis failed: {str(e)}")
        return None

def display_results(
    results: Dict,
    handwriting_risk: float,
    language_risk: float,
    final_risk: float,
    enable_gradcam: bool
):

    """
    Display screening results with professional formatting.
    """
    risk_level, message, color = calculate_risk_level(final_risk)
    
    # Main result card
    st.markdown("---")
    st.markdown(f"###  Screening Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Handwriting Patch Risk",
            value=f"{results['risk_score']:.2%}",
            help="Percentage of patches showing dyslexia-related patterns"
        )
    
    with col2:
        st.metric(
            label="Risk Level",
            value=risk_level,
            help="Categorized risk assessment"
        )
    
    with col3:
        st.metric(
            label="Patches Analyzed",
            value=results['total_patches'],
            help="Total number of handwriting patches processed"
        )
    #Multimodal Risk Summary
    st.markdown("### Multimodal Risk Summary")
    col_h, col_l, col_f = st.columns(3)
    with col_h:
        st.metric(
            "Handwriting Risk",
            f"{handwriting_risk:.2%}",
            help="Risk estimated from handwriting visual patterns"
        )
    with col_l:
        if language_risk is not None:
            st.metric(
                "Language Risk",
                f"{language_risk:.2%}",
                help="Risk estimated from linguistic error patterns"
            )
        else:
            st.metric(
                "Language Risk",
                "Not Available",
                help="Insufficient text extracted for language analysis"
            )
    with col_f:
        st.metric(
            "Final Screening Risk",
            f"{final_risk:.2%}",
            help="Conservative fusion of handwriting and language risks"
        )
    
    # Risk interpretation
    if color == "red":
        st.error(message)
    elif color == "orange":
        st.warning(message)
    else:
        st.success(message)
    
    # Detailed statistics (expandable)
    with st.expander("🔍 Detailed Analysis"):
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.write("**Patch Classification:**")
            st.write(f"- Dyslexia-like patterns: {results['dyslexic_patches']} patches")
            st.write(f"- Typical patterns: {results['normal_patches']} patches")
        
        with col_b:
            st.write("**Confidence Metrics:**")
            st.write(f"- Average: {results['avg_confidence']:.2%}")
            st.write(f"- Highest: {results['max_confidence']:.2%}")
            st.write(f"- Lowest: {results['min_confidence']:.2%}")
        
        # Distribution visualization
        st.write("**Confidence Distribution:**")
        st.bar_chart(results['predictions'])
    
    # Grad-CAM Visualizations
    if enable_gradcam:
        st.markdown("---")
        st.markdown("### 🔬 AI Explainability - What the Model Sees")
        
        if results.get('gradcam_visualizations'):
            st.info(
                "**Grad-CAM Heatmaps**: These visualizations show which parts of the handwriting "
                "the AI model focused on when making predictions. Warmer colors (red/yellow) indicate "
                "regions that strongly influenced the prediction."
            )
            
            vis_data = results['gradcam_visualizations']
            
            # Separate dyslexic and normal predictions
            dyslexic_vis = [v for v in vis_data if v['prediction'] == 'Dyslexic']
            normal_vis = [v for v in vis_data if v['prediction'] == 'Normal']
            
            # Sort by confidence (highest first)
            dyslexic_vis_sorted = sorted(dyslexic_vis, key=lambda x: x['confidence'], reverse=True)
            normal_vis_sorted = sorted(normal_vis, key=lambda x: x['confidence'])
            
            # Display statistics
            col_v1, col_v2 = st.columns(2)
            with col_v1:
                st.metric("Dyslexic Visualizations", len(dyslexic_vis))
            with col_v2:
                st.metric("Normal Visualizations", len(normal_vis))
            
            # Show dyslexic patches
            if dyslexic_vis_sorted:
                st.markdown("#### High Dyslexia Confidence Patches")
                num_to_show = min(3, len(dyslexic_vis_sorted))
                
                for i, vis in enumerate(dyslexic_vis_sorted[:num_to_show]):
                    with st.expander(f"Patch {vis['patch_idx'] + 1} - Dyslexia Probability: {vis['confidence']:.2%}", expanded=(i==0)):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.image(vis['original'], caption="Original Patch", width=300)
                        
                        with col2:
                            st.image(vis['overlay'], caption="Grad-CAM Overlay", width=300)
                        
                        st.markdown(
                            f"**Interpretation**: The model predicted **{vis['confidence']:.1%}** probability "
                            f"of dyslexia patterns. Red/yellow regions show where the model detected concerning features."
                        )
            
            # Show normal patches
            if normal_vis_sorted:
                st.markdown("####  Low Dyslexia Confidence Patches (Normal)")
                num_to_show = min(3, len(normal_vis_sorted))
                
                for i, vis in enumerate(normal_vis_sorted[:num_to_show]):
                    with st.expander(f"Patch {vis['patch_idx'] + 1} - Normal Probability: {(1-vis['confidence']):.2%}"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.image(vis['original'], caption="Original Patch", width=300)
                        
                        with col2:
                            st.image(vis['overlay'], caption="Grad-CAM Overlay", width=300)
                        
                        st.markdown(
                            f"**Interpretation**: The model predicted **{vis['confidence']:.1%}** probability "
                            f"of dyslexia patterns (very low). Red/yellow regions show key handwriting features."
                        )
        else:
            st.warning(
                " No Grad-CAM visualizations generated. This can happen if:\n"
                "- All patches have moderate confidence (0.4-0.6)\n"
                "- Grad-CAM generation failed\n\n"
                "Try uploading a different handwriting sample."
            )
    
    # Educational disclaimer
    st.markdown("---")
    st.info(
        "** Important Notice:**\n\n"
        "This tool provides **early screening support only** and is NOT a medical diagnosis. "
        "It uses AI to identify handwriting patterns that may be associated with dyslexia. "
        "Always consult qualified educational psychologists or medical professionals for "
        "comprehensive assessment and diagnosis."
    )


# Main Application

def main():
    # Page configuration
    st.set_page_config(
        page_title="Dyslexia Screening System",
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header
    st.title(" AI-Based Dyslexia Handwriting Screening System")
    st.markdown(
        "An early screening tool that analyzes handwriting patterns to identify "
        "potential indicators of dyslexia. Designed for educators, parents, and screening contexts."
    )
    
    # Sidebar - Information and Settings
    with st.sidebar:
        st.header(" About This Tool")
        st.write(
            "This system uses deep learning to analyze handwriting images and "
            "provide risk estimation for dyslexia screening."
        )
        
        st.markdown("**Key Features:**")
        st.markdown("- Full-page handwriting analysis")
        st.markdown("- Patch-based inference")
        st.markdown("- Explainable AI approach")
        st.markdown("- High recall prioritization")
        
        st.markdown("---")
        st.markdown("**Model Information:**")
        st.markdown(f"- Architecture: ResNet-50")
        st.markdown(f"- Input Size: {config.IMAGE_SIZE}×{config.IMAGE_SIZE}")
        st.markdown(f"- Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
        
        st.markdown("---")
        st.markdown("**Settings:**")
        
        # Grad-CAM toggle
        enable_gradcam = st.checkbox(
            "Enable Grad-CAM Visualization",
            value=False,
            help="Show AI explainability heatmaps (may slow down processing)"
        )
        st.session_state.enable_gradcam = enable_gradcam
        
        if enable_gradcam:
            st.success("✓ Grad-CAM enabled")
            st.caption("Will show visualizations for patches with confidence >60% or <40%")
        else:
            st.info(" Enable Grad-CAM to see what the AI focuses on")
        
        st.markdown("---")
        st.markdown("**Guidelines:**")
        st.markdown("- Upload clear handwriting images")
        st.markdown("- Full page or multiple sentences")
        st.markdown(f"- Max file size: {MAX_FILE_SIZE_MB}MB")
        st.markdown("- Formats: PNG, JPG, JPEG")
    
    # Initialize system
    if st.session_state.model is None:
        st.session_state.model, st.session_state.device = initialize_system()
    
    model = st.session_state.model
    device = st.session_state.device
    
    # File upload section
    st.markdown("---")
    st.header("Upload Handwriting Sample")
    
    uploaded_file = st.file_uploader(
        "Choose a handwriting image (full page recommended)",
        type=["png", "jpg", "jpeg"],
        help="Upload a clear image of handwritten text. Full pages work best."
    )
    
    if uploaded_file is not None:
        # Validate image
        if not validate_image(uploaded_file):
            return
        
        try:
            # Load and display image
            image = Image.open(uploaded_file).convert("RGB")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.image(image, caption="Uploaded Handwriting Sample", width="stretch")
            
            with col2:
                st.write("**Image Information:**")
                st.write(f"- Dimensions: {image.size[0]} × {image.size[1]}")
                st.write(f"- Format: {image.format}")
                st.write(f"- Mode: {image.mode}")
            
            # Convert to NumPy array
            image_np = np.array(image)
            
            # Process button
            if st.button("🔬 Analyze Handwriting", type="primary"):
                with st.spinner("Processing image..."):
                    # Split into patches
                    patches = split_into_patches(
                        image_np,
                        patch_size=config.IMAGE_SIZE,
                        stride=config.IMAGE_SIZE
                    )
                    
                    # Validate patches
                    if len(patches) < MIN_PATCHES_REQUIRED:
                        st.warning(
                            f"Only {len(patches)} patches detected. "
                            f"For reliable screening, at least {MIN_PATCHES_REQUIRED} patches are recommended. "
                            "Try uploading a larger image with more handwriting."
                        )
                        if len(patches) == 0:
                            return
                    
                    st.success(f" Detected {len(patches)} handwriting patches")
                    
                    # Run inference with Grad-CAM if enabled
                    results = run_inference(
                        patches, 
                        model, 
                        device, 
                        enable_gradcam=st.session_state.enable_gradcam
                    )
                    ##Language Analysis
                    with st.spinner("Analyzing written content (language patterns)..."):
                        extracted_text = extract_text_from_image(image)
                        if len(extracted_text.split()) < 20:
                            st.warning(
                                "Extracted text is very short"
                                "Language based analysis may be less reliable"
                            )
                            language_risk=None
                        else:
                            language_risk = predict_language_risk(extracted_text)

                    #Late Fusion
                    handwriting_risk = results["risk_score"]
                    if language_risk is  None:
                        final_risk = handwriting_risk 
    
                    else:
                 
                        final_risk=(0.6*handwriting_risk + 0.25*language_risk+ 0.15*max(language_risk - handwriting_risk, 0))
                    
                    if handwriting_risk<0.2:
                        final_risk=min(final_risk,0.45)


                    if results:
                        # Display results
                        display_results(results,handwriting_risk,language_risk,final_risk,st.session_state.enable_gradcam)
                        # Log analysis
                        logger.info(
                            f"Analysis completed: Risk={results['risk_score']:.2%}, "
                            f"Patches={results['total_patches']}, "
                            f"GradCAM={'enabled' if st.session_state.enable_gradcam else 'disabled'}"
                        )
        
        except Exception as e:
            logger.error(f"Processing error: {str(e)}")
            st.error(f" An error occurred during processing: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Developed for educational and screening purposes only | "
        "Not a substitute for professional diagnosis | "
        f"Version {config.VERSION if hasattr(config, 'VERSION') else '1.0.0'}"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()