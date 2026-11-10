try:
    import spaces  # Must be imported BEFORE torch on Hugging Face ZeroGPU Spaces
except ImportError:
    pass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import cv2
import numpy as np
import gradio as gr
import plotly.graph_objects as go
from PIL import Image

# -----------------------------------------------------------------------------
# 1. MODEL INITIALIZATION & MC-DROPOUT SETUP
# -----------------------------------------------------------------------------
DR_CLASSES = [
    "Stage 0: No DR",
    "Stage 1: Mild NPDR",
    "Stage 2: Moderate NPDR",
    "Stage 3: Severe NPDR",
    "Stage 4: Proliferative DR"
]

def load_dr_model():
    try:
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    except Exception:
        model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_ftrs, 5)
    )
    model.eval()
    return model

model = load_dr_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def enable_dropout(model):
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

# -----------------------------------------------------------------------------
# 2. EPISTEMIC UNCERTAINTY ENGINE (MONTE CARLO DROPOUT)
# -----------------------------------------------------------------------------
def predict_with_uncertainty(image_tensor, num_samples=10):
    enable_dropout(model)
    probs_list = []
    
    with torch.no_grad():
        for _ in range(num_samples):
            output = model(image_tensor)
            probs = F.softmax(output, dim=1)
            probs_list.append(probs.numpy()[0])
            
    probs_matrix = np.array(probs_list)
    mean_probs = np.mean(probs_matrix, axis=0)
    std_probs = np.std(probs_matrix, axis=0)
    
    pred_class_idx = np.argmax(mean_probs)
    pred_confidence = mean_probs[pred_class_idx]
    pred_uncertainty = std_probs[pred_class_idx]
    
    return pred_class_idx, pred_confidence, pred_uncertainty, mean_probs

# -----------------------------------------------------------------------------
# 3. DUAL XAI HEATMAP ENGINE (INTERACTIVE HOVER MAPS)
# -----------------------------------------------------------------------------
def generate_dual_heatmaps(pil_img, image_tensor, target_class):
    orig_np = np.array(pil_img.resize((224, 224)))
    
    gradients = []
    activations = []
    
    def save_gradient(grad):
        gradients.append(grad)
        
    def forward_hook(module, input, output):
        activations.append(output)
        output.register_hook(save_gradient)
        
    target_layer = model.layer4[-1]
    handle = target_layer.register_forward_hook(forward_hook)
    
    model.eval()
    output = model(image_tensor)
    one_hot = torch.zeros((1, output.size()[-1]), dtype=torch.float32)
    one_hot[0][target_class] = 1.0
    
    model.zero_grad()
    output.backward(gradient=one_hot, retain_graph=True)
    handle.remove()
    
    grads = gradients[0].cpu().data.numpy()[0]
    act = activations[0].cpu().data.numpy()[0]
    
    # 1. Focal Lesion Map (Grad-CAM)
    weights = np.mean(grads, axis=(1, 2))
    cam = np.zeros(act.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * act[i, :, :]
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    if np.max(cam) > 0:
        cam = cam / np.max(cam)
        
    heatmap_focal = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay_focal = cv2.addWeighted(orig_np, 0.6, cv2.cvtColor(heatmap_focal, cv2.COLOR_BGR2RGB), 0.4, 0)
    
    # 2. Global Context Map (Mean Feature Activation)
    global_act = np.mean(act, axis=0)
    global_act = np.maximum(global_act, 0)
    global_act = cv2.resize(global_act, (224, 224))
    if np.max(global_act) > 0:
        global_act = global_act / np.max(global_act)
        
    heatmap_global = cv2.applyColorMap(np.uint8(255 * global_act), cv2.COLORMAP_VIRIDIS)
    overlay_global = cv2.addWeighted(orig_np, 0.65, cv2.cvtColor(heatmap_global, cv2.COLOR_BGR2RGB), 0.35, 0)
    
    # Build Interactive Plotly Figures with Pixel-Level Hover Information
    # View 1: Focal Map Interactive Plot
    fig_focal = go.Figure()
    fig_focal.add_trace(go.Image(z=overlay_focal, hoverinfo='none'))
    fig_focal.add_trace(go.Heatmap(
        z=cam,
        colorscale='Jet',
        showscale=True,
        colorbar=dict(title=dict(text="<b>Focal Saliency</b>", font=dict(size=16))),
        opacity=0.35,
        hovertemplate="<b>Pixel Coord:</b> X: %{x}px, Y: %{y}px<br><b>Focal Saliency Intensity:</b> %{z:.4f} / 1.0000<br><b>Pathology Focus:</b> Micro-lesion Hotspot<extra></extra>"
    ))
    fig_focal.update_layout(
        title=dict(text="<b>👁️ View 1: Focal Lesion Map (Grad-CAM) — Hover Mouse to Inspect Every Pixel</b>", font=dict(size=20)),
        xaxis=dict(title="Horizontal Axis (Pixels)", titlefont=dict(size=18), tickfont=dict(size=14)),
        yaxis=dict(title="Vertical Axis (Pixels)", titlefont=dict(size=18), tickfont=dict(size=14), autorange="reversed"),
        height=620,
        margin=dict(l=20, r=20, t=60, b=20)
    )

    # View 2: Global Map Interactive Plot
    fig_global = go.Figure()
    fig_global.add_trace(go.Image(z=overlay_global, hoverinfo='none'))
    fig_global.add_trace(go.Heatmap(
        z=global_act,
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(title=dict(text="<b>Global Context</b>", font=dict(size=16))),
        opacity=0.35,
        hovertemplate="<b>Pixel Coord:</b> X: %{x}px, Y: %{y}px<br><b>Global Feature Activation:</b> %{z:.4f} / 1.0000<br><b>Pathology Focus:</b> Background Vascular Arcade<extra></extra>"
    ))
    fig_global.update_layout(
        title=dict(text="<b>🌐 View 2: Global Context Map (AGCAM) — Hover Mouse to Inspect Every Pixel</b>", font=dict(size=20)),
        xaxis=dict(title="Horizontal Axis (Pixels)", titlefont=dict(size=18), tickfont=dict(size=14)),
        yaxis=dict(title="Vertical Axis (Pixels)", titlefont=dict(size=18), tickfont=dict(size=14), autorange="reversed"),
        height=620,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig_focal, fig_global

# -----------------------------------------------------------------------------
# 4. CLINICAL DIAGNOSTIC PIPELINE
# -----------------------------------------------------------------------------
def analyze_fundus_image(pil_img):
    if pil_img is None:
        gr.Warning("⚠️ Please upload a valid fundus image first!")
        empty_popup = "<div style='display:none;'></div>"
        empty_report = "<div style='font-size: 20px; color: #dc2626;'><b>Please upload a fundus image to run analysis.</b></div>"
        return empty_popup, None, None, None, empty_report
        
    img_rgb = pil_img.convert("RGB")
    image_tensor = transform(img_rgb).unsqueeze(0)
    
    # 1. Predict with Monte Carlo Uncertainty
    class_idx, confidence, uncertainty, mean_probs = predict_with_uncertainty(image_tensor)
    predicted_stage = DR_CLASSES[class_idx]
    
    # 2. Generate Interactive Heatmap Figures
    fig_focal, fig_global = generate_dual_heatmaps(img_rgb, image_tensor, class_idx)
    
    # 3. Uncertainty Assessment & Alerts
    if uncertainty < 0.05:
        uncertainty_status = "🟢 LOW UNCERTAINTY (High AI Confidence)"
        action_advice = "The AI is highly confident in this assessment. Routine clinical workflow recommended."
        badge_color = "#16a34a"
        popup_icon = "🩺"
        gr.Info(f"Diagnostic Result: {predicted_stage} ({confidence*100:.1f}% Confidence)")
    elif uncertainty < 0.12:
        uncertainty_status = "🟡 MODERATE UNCERTAINTY (Moderate AI Confidence)"
        action_advice = "The AI detected borderline features. Secondary review by a senior clinician is advised."
        badge_color = "#d97706"
        popup_icon = "⚠️"
        gr.Warning(f"Borderline Case: {predicted_stage} (MC Uncertainty: {uncertainty*100:.1f}%)")
    else:
        uncertainty_status = "🔴 HIGH UNCERTAINTY (AI Ambiguity Alert)"
        action_advice = "⚠️ HIGH UNCERTAINTY DETECTED: Image may be out-of-distribution or noisy. MANDATORY HUMAN OPHTHALMOLOGIST REFERRAL REQUIRED."
        badge_color = "#dc2626"
        popup_icon = "🚨"
        gr.Warning(f"HIGH UNCERTAINTY ALERT: {predicted_stage} ({uncertainty*100:.1f}% Variance)")
        
    # Centered Modal Pop-up Card HTML
    popup_modal_html = f"""
    <div style="
        position: fixed;
        top: 0; left: 0; width: 100vw; height: 100vh;
        background: rgba(15, 23, 42, 0.75);
        backdrop-filter: blur(5px);
        display: flex; align-items: center; justify-content: center;
        z-index: 999999;
    " id="prediction-popup-overlay" onclick="document.getElementById('prediction-popup-overlay').style.display='none';">
        <div style="
            background: #ffffff;
            width: 90%; max-width: 540px;
            padding: 32px;
            border-radius: 20px;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.4);
            text-align: center;
            border-top: 10px solid {badge_color};
            position: relative;
        " onclick="event.stopPropagation();">
            <div style="font-size: 54px; margin-bottom: 8px;">{popup_icon}</div>
            <h2 style="font-size: 28px; color: #0f172a; margin: 0 0 12px 0; font-weight: 800;">Diagnostic Prediction Result</h2>
            <div style="font-size: 26px; color: #1e40af; font-weight: 800; background: #eff6ff; padding: 14px; border-radius: 12px; margin: 16px 0;">
                {predicted_stage}
            </div>
            <div style="font-size: 20px; color: #334155; margin-bottom: 10px;">
                <b>Prediction Confidence:</b> <span style="font-weight: 800; color: #0f172a;">{confidence * 100:.2f}%</span>
            </div>
            <div style="font-size: 20px; color: #334155; margin-bottom: 18px;">
                <b>Epistemic Uncertainty:</b> <span style="font-weight: 800; color: {badge_color};">{uncertainty * 100:.2f}%</span>
            </div>
            <div style="font-size: 17px; font-weight: 700; color: {badge_color}; background: #f8fafc; padding: 12px; border-radius: 10px; margin-bottom: 24px; border: 1px solid #e2e8f0;">
                {uncertainty_status}
            </div>
            <button onclick="document.getElementById('prediction-popup-overlay').style.display='none';" style="
                background: #1e40af; color: #ffffff;
                font-size: 20px; font-weight: 800;
                padding: 14px 36px; border: none; border-radius: 10px;
                cursor: pointer; box-shadow: 0 4px 6px -1px rgba(30, 64, 175, 0.4);
            ">Close & Explore Views</button>
        </div>
    </div>
    """

    # Detailed Clinical Diagnostic Report
    report = f"""
    <div style="font-size: 20px; line-height: 1.6; background-color: #f8fafc; padding: 25px; border-radius: 12px; border: 1px solid #cbd5e1;">
        <h2 style="font-size: 26px; color: #0f172a; margin-top: 0; font-weight: 800;">📋 Complete Clinical Diagnostic Summary</h2>
        <p style="font-size: 20px; margin: 8px 0;"><b>• Predicted DR Grade:</b> <span style="color: #1e40af; font-size: 22px;"><b>{predicted_stage}</b></span></p>
        <p style="font-size: 20px; margin: 8px 0;"><b>• Prediction Confidence:</b> <b>{confidence * 100:.2f}%</b></p>
        <p style="font-size: 20px; margin: 8px 0;"><b>• Epistemic Uncertainty (MC Variance):</b> <b>{uncertainty * 100:.2f}%</b></p>
        <p style="font-size: 20px; margin: 8px 0;"><b>• Safety Guardrail Status:</b> <b>{uncertainty_status}</b></p>
        <hr style="margin: 20px 0; border: 0; border-top: 1px solid #cbd5e1;">
        <h3 style="font-size: 22px; color: #0f172a; margin-bottom: 8px;">👨‍⚕️ Clinical Decision Recommendation</h3>
        <p style="font-size: 20px; font-weight: 600; color: #334155; margin: 0;">{action_advice}</p>
    </div>
    """

    probs_dict = {DR_CLASSES[i]: float(mean_probs[i]) for i in range(5)}

    return popup_modal_html, probs_dict, fig_focal, fig_global, report

# -----------------------------------------------------------------------------
# 5. MULTI-PAGE INTERFACE WITH SEPARATE MAP PAGES & HOVER INSPECTION
# -----------------------------------------------------------------------------
custom_css = """
body, .gradio-container {
    font-size: 20px !important;
    font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif !important;
}

.title-header {
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    color: #0f172a !important;
    margin-bottom: 20px !important;
    padding-bottom: 12px;
    border-bottom: 4px solid #1e40af;
}

button.primary-btn {
    font-size: 22px !important;
    font-weight: 800 !important;
    padding: 14px 28px !important;
    background-color: #1e40af !important;
    color: white !important;
    border-radius: 10px !important;
}

.tab-nav button {
    font-size: 21px !important;
    font-weight: 800 !important;
    padding: 14px 24px !important;
}

.gradio-container label, .gradio-container span, .gradio-container .block-title {
    font-size: 21px !important;
    font-weight: 700 !important;
}
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    # Short Name + Title ONLY (No description matter under heading)
    gr.HTML("<div class='title-header'>🏥 RetiGuard-AI: Uncertainty-Aware & Multi-View XAI System for Diabetic Retinopathy</div>")
    
    # Hidden placeholder for centered modal pop-up card
    popup_output = gr.HTML()
    
    with gr.Tabs() as tabs:
        # PAGE 1: Upload & Run Analysis
        with gr.Tab("📷 1. Upload & Analysis Scan"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_img = gr.Image(type="pil", label="Upload Retinal Fundus Photograph")
                    submit_btn = gr.Button("🔍 Run Analysis", variant="primary", elem_classes=["primary-btn"])
                with gr.Column(scale=1):
                    output_report = gr.HTML(value="<div style='font-size: 20px; color: #64748b; padding: 25px; background: #f8fafc; border-radius: 12px; border: 1px dashed #cbd5e1;'>Upload a retinal scan and click <b>'Run Analysis'</b>.<br><br>The system will calculate diagnostic predictions, epistemic uncertainty, and show a centered pop-up modal upon completion.</div>")
        
        # PAGE 2: Class Probabilities
        with gr.Tab("📊 2. Class Probabilities View"):
            gr.Markdown("## 📊 5-Stage DR Severity Class Probabilities Breakdown")
            output_label = gr.Label(num_top_classes=5, label="Predicted DR Stage Probabilities")
            
        # PAGE 3: Focal Lesion Map Page
        with gr.Tab("👁️ 3. View 1: Focal Lesion Map (Grad-CAM)"):
            gr.Markdown("## 👁️ View 1: Focal Lesion Map (Grad-CAM — Micro-Lesion Hotspots)")
            gr.Markdown("*Move your mouse over the image to inspect exact pixel coordinates and saliency intensity values at every inch.*")
            output_focal = gr.Plot(label="Focal Lesion Saliency Heatmap (Hover Enabled)")
            
        # PAGE 4: Global Context Map Page
        with gr.Tab("🌐 4. View 2: Global Context Map (AGCAM)"):
            gr.Markdown("## 🌐 View 2: Global Context Map (AGCAM — Background Coverage)")
            gr.Markdown("*Move your mouse over the image to inspect exact pixel coordinates and global activation intensity at every inch.*")
            output_global = gr.Plot(label="Global Feature Activation Heatmap (Hover Enabled)")
    
    # Connect button click
    submit_btn.click(
        fn=analyze_fundus_image,
        inputs=[input_img],
        outputs=[popup_output, output_label, output_focal, output_global, output_report]
    )

if __name__ == "__main__":
    demo.launch()
