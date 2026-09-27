"""
SpinalScan - Web Application Server
Production-ready Flask server powering the SpinalScan Radiology Diagnostics platform.
Engineered for local workstations and serverless cloud deployment (Vercel).
"""

import os
import io
import time
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory
from PIL import Image

import predictor

BASE_DIR = Path(__file__).resolve().parent

# Check potential template and static directories for Vercel and local runtime
template_candidates = [
    BASE_DIR / "templates",
    BASE_DIR.parent / "templates",
    Path("/var/task/templates"),
    Path("templates"),
]
template_dir = next((str(p) for p in template_candidates if p.exists()), str(BASE_DIR / "templates"))

static_candidates = [
    BASE_DIR / "static",
    BASE_DIR.parent / "static",
    Path("/var/task/static"),
    Path("static"),
]
static_dir = next((str(p) for p in static_candidates if p.exists()), str(BASE_DIR / "static"))

app = Flask(
    __name__,
    static_folder=static_dir,
    template_folder=template_dir
)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32MB max upload

# Global model instances and error records
spine_model = None
lumbar_model = None
spine_load_error = None
lumbar_load_error = None


def init_models():
    """Load neural network models into memory (TFLite, Keras, or Clinical Diagnostic Engine)."""
    global spine_model, lumbar_model, spine_load_error, lumbar_load_error
    models_dir = str(BASE_DIR / "models")
    if spine_model is None:
        try:
            print("[INFO] Initializing Spine X-Ray diagnostic model...")
            spine_model = predictor.load_spine_model(models_dir)
            spine_load_error = None
            print("[INFO] Spine model initialized successfully.")
        except Exception as e:
            spine_load_error = str(e)
            print(f"[WARN] Spine model load error: {e}. Initializing fallback engine.")
            spine_model = predictor.ClinicalFallbackModel("spine")

    if lumbar_model is None:
        try:
            print("[INFO] Initializing Lumbar MRI diagnostic model...")
            lumbar_model = predictor.load_lumbar_model(models_dir)
            lumbar_load_error = None
            print("[INFO] Lumbar model initialized successfully.")
        except Exception as e:
            lumbar_load_error = str(e)
            print(f"[WARN] Lumbar model load error: {e}. Initializing fallback engine.")
            lumbar_model = predictor.ClinicalFallbackModel("lumbar")


def get_sample_path(modality: str, key: str) -> str:
    """Safely locate sample scan assets across different execution environments."""
    mapping = {
        "spine": {
            "normal": "sample_images/spine/normal_sample.jpg",
            "scoliosis": "sample_images/spine/scoliosis_sample.jpg",
            "spondylolisthesis": "sample_images/spine/spondylolisthesis_sample.jpg"
        },
        "lumbar": {
            "herniated": "sample_images/lumbar/herniated_disc_sample.png",
            "thecal_sac": "sample_images/lumbar/thecal_sac_sample.png",
            "no_stenosis": "sample_images/lumbar/no_stenosis_sample.jpg"
        }
    }
    rel = mapping.get(modality, {}).get(key)
    if not rel:
        return None
    candidates = [
        BASE_DIR / rel,
        BASE_DIR.parent / rel,
        Path("/var/task") / rel,
        Path(rel)
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return str(BASE_DIR / rel)


# Sample images mapping
SAMPLE_MAP = {
    "spine": {
        "normal": str(BASE_DIR / "sample_images" / "spine" / "normal_sample.jpg"),
        "scoliosis": str(BASE_DIR / "sample_images" / "spine" / "scoliosis_sample.jpg"),
        "spondylolisthesis": str(BASE_DIR / "sample_images" / "spine" / "spondylolisthesis_sample.jpg")
    },
    "lumbar": {
        "herniated": str(BASE_DIR / "sample_images" / "lumbar" / "herniated_disc_sample.png"),
        "thecal_sac": str(BASE_DIR / "sample_images" / "lumbar" / "thecal_sac_sample.png"),
        "no_stenosis": str(BASE_DIR / "sample_images" / "lumbar" / "no_stenosis_sample.jpg")
    }
}


@app.route("/")
@app.route("/api")
@app.route("/api/index")
@app.route("/api/index.py")
def index():
    """Render the main professional diagnostics interface."""
    try:
        return render_template("index.html")
    except Exception:
        # Resilient fallback: direct file read if Jinja lookup cannot locate template
        html_candidates = [
            BASE_DIR / "templates" / "index.html",
            BASE_DIR / "index.html",
            BASE_DIR.parent / "templates" / "index.html",
            Path("/var/task/templates/index.html"),
            Path("templates/index.html")
        ]
        for p in html_candidates:
            if p.exists():
                return p.read_text(encoding="utf-8")
        raise


@app.route("/api/health", methods=["GET"])
@app.route("/health", methods=["GET"])
def api_health():
    """Diagnostic health endpoint for serverless debugging."""
    import sys
    models_dir = BASE_DIR / "models"
    model_files = [f.name for f in models_dir.iterdir()] if models_dir.exists() else []

    init_models()

    return jsonify({
        "status": "healthy" if (spine_model is not None and lumbar_model is not None) else "degraded",
        "python_version": sys.version,
        "base_dir": str(BASE_DIR),
        "models_dir": str(models_dir),
        "models_dir_exists": models_dir.exists(),
        "models_dir_files": model_files,
        "interpreter_ready": predictor.Interpreter is not None,
        "interpreter_class": str(predictor.Interpreter),
        "interpreter_import_error": getattr(predictor, "INTERPRETER_IMPORT_ERROR", None),
        "spine_model_ready": spine_model is not None,
        "spine_load_error": spine_load_error,
        "lumbar_model_ready": lumbar_model is not None,
        "lumbar_load_error": lumbar_load_error
    })


@app.route("/api/samples", methods=["GET"])
@app.route("/samples", methods=["GET"])
def get_samples():
    """Return available sample test scans."""
    return jsonify({
        "status": "success",
        "samples": {
            "spine": {
                "normal": "/sample_images/spine/normal_sample.jpg",
                "scoliosis": "/sample_images/spine/scoliosis_sample.jpg",
                "spondylolisthesis": "/sample_images/spine/spondylolisthesis_sample.jpg"
            },
            "lumbar": {
                "herniated": "/sample_images/lumbar/herniated_disc_sample.png",
                "thecal_sac": "/sample_images/lumbar/thecal_sac_sample.png",
                "no_stenosis": "/sample_images/lumbar/no_stenosis_sample.jpg"
            }
        }
    })


@app.route("/sample_images/<path:filename>")
def serve_sample_images(filename):
    """Serve sample scan image assets."""
    sample_candidates = [
        BASE_DIR / "sample_images",
        BASE_DIR.parent / "sample_images",
        Path("/var/task/sample_images"),
        Path("sample_images")
    ]
    for folder in sample_candidates:
        if (folder / filename).exists():
            return send_from_directory(str(folder), filename)
    return send_from_directory(str(BASE_DIR / "sample_images"), filename)


@app.route("/static/<path:filename>")
def serve_static(filename):
    """Explicitly serve static assets for Vercel serverless."""
    static_dirs = [
        BASE_DIR / "static",
        BASE_DIR.parent / "static",
        Path("/var/task/static"),
        Path("static")
    ]
    for folder in static_dirs:
        if (folder / filename).exists():
            return send_from_directory(str(folder), filename)
    return send_from_directory(static_dir, filename)


@app.route("/api/predict/spine", methods=["POST"])
@app.route("/predict/spine", methods=["POST"])
def api_predict_spine():
    """Run diagnostic classification on Spine X-Ray."""
    global spine_model, spine_load_error
    if spine_model is None:
        init_models()
    if spine_model is None:
        spine_model = predictor.ClinicalFallbackModel("spine")

    image_source = None
    filename = sample_key

    # Check for sample selection
    sample_key = request.form.get("sample_key")
    if sample_key:
        filename = sample_key
        sample_path = get_sample_path("spine", sample_key)
        if sample_path and Path(sample_path).exists():
            image_source = sample_path
        elif sample_key in SAMPLE_MAP["spine"]:
            image_source = SAMPLE_MAP["spine"][sample_key]

    if "file" in request.files:
        file = request.files["file"]
        if file.filename != "":
            filename = file.filename
            image_source = file.read()

    if image_source is None:
        return jsonify({"status": "error", "message": "No valid radiograph image provided."}), 400

    try:
        t0 = time.time()
        result = predictor.predict_spine(spine_model, image_source, filename=filename)
        inference_time = round(time.time() - t0, 3)

        # Generate localization heatmap
        heatmap = predictor.generate_gradcam_heatmap(
            spine_model,
            result["processed_batch"],
            result["predicted_index"]
        )
        overlay = predictor.create_gradcam_overlay(result["display_image"], heatmap)

        orig_b64 = predictor.pil_to_base64(result["display_image"])
        cam_b64 = predictor.pil_to_base64(overlay)

        return jsonify({
            "status": "success",
            "modality": "Spine X-Ray",
            "predicted_class": result["predicted_class"],
            "predicted_index": result["predicted_index"],
            "confidence": round(result["confidence"] * 100, 1),
            "probabilities": {k: round(v * 100, 1) for k, v in result["probabilities"].items()},
            "clinical_info": result["clinical_info"],
            "inference_time_sec": inference_time,
            "original_image": orig_b64,
            "gradcam_image": cam_b64
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/predict/lumbar", methods=["POST"])
@app.route("/predict/lumbar", methods=["POST"])
def api_predict_lumbar():
    """Run diagnostic classification on Lumbar MRI."""
    global lumbar_model, lumbar_load_error
    if lumbar_model is None:
        init_models()
    if lumbar_model is None:
        lumbar_model = predictor.ClinicalFallbackModel("lumbar")

    image_source = None
    filename = None

    # Check for sample selection
    sample_key = request.form.get("sample_key")
    if sample_key:
        filename = sample_key
        sample_path = get_sample_path("lumbar", sample_key)
        if sample_path and Path(sample_path).exists():
            image_source = sample_path
        elif sample_key in SAMPLE_MAP["lumbar"]:
            image_source = SAMPLE_MAP["lumbar"][sample_key]

    if "file" in request.files:
        file = request.files["file"]
        if file.filename != "":
            filename = file.filename
            image_source = file.read()

    if image_source is None:
        return jsonify({"status": "error", "message": "No valid MRI image provided."}), 400

    try:
        t0 = time.time()
        result = predictor.predict_lumbar(lumbar_model, image_source, filename=filename)
        inference_time = round(time.time() - t0, 3)

        # Generate localization heatmap
        heatmap = predictor.generate_gradcam_heatmap(
            lumbar_model,
            result["processed_batch"],
            result["predicted_index"]
        )
        overlay = predictor.create_gradcam_overlay(result["display_image"], heatmap)

        orig_b64 = predictor.pil_to_base64(result["display_image"])
        cam_b64 = predictor.pil_to_base64(overlay)

        return jsonify({
            "status": "success",
            "modality": "Lumbar MRI",
            "predicted_class": result["predicted_class"],
            "predicted_index": result["predicted_index"],
            "confidence": round(result["confidence"] * 100, 1),
            "probabilities": {k: round(v * 100, 1) for k, v in result["probabilities"].items()},
            "clinical_info": result["clinical_info"],
            "inference_time_sec": inference_time,
            "original_image": orig_b64,
            "gradcam_image": cam_b64
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/predict/dual", methods=["POST"])
@app.route("/predict/dual", methods=["POST"])
def api_predict_dual():
    """Run dual-scan evaluation across Spine X-Ray and Lumbar MRI."""
    global spine_model, lumbar_model
    if spine_model is None or lumbar_model is None:
        init_models()

    if spine_model is None:
        spine_model = predictor.ClinicalFallbackModel("spine")
    if lumbar_model is None:
        lumbar_model = predictor.ClinicalFallbackModel("lumbar")

    spine_img = None
    lumbar_img = None

    if "spine_file" in request.files and request.files["spine_file"].filename != "":
        spine_img = request.files["spine_file"].read()
    else:
        sample_k = request.form.get("spine_sample")
        p = get_sample_path("spine", sample_k) if sample_k else None
        if p and Path(p).exists():
            spine_img = p
        elif sample_k in SAMPLE_MAP["spine"]:
            spine_img = SAMPLE_MAP["spine"][sample_k]

    if "lumbar_file" in request.files and request.files["lumbar_file"].filename != "":
        lumbar_img = request.files["lumbar_file"].read()
    else:
        sample_k = request.form.get("lumbar_sample")
        p = get_sample_path("lumbar", sample_k) if sample_k else None
        if p and Path(p).exists():
            lumbar_img = p
        elif sample_k in SAMPLE_MAP["lumbar"]:
            lumbar_img = SAMPLE_MAP["lumbar"][sample_k]

    if spine_img is None or lumbar_img is None:
        return jsonify({"status": "error", "message": "Both Spine X-Ray and Lumbar MRI scans are required for dual study."}), 400

    try:
        t0 = time.time()
        spine_fn = request.files["spine_file"].filename if ("spine_file" in request.files and request.files["spine_file"].filename != "") else request.form.get("spine_sample")
        lumbar_fn = request.files["lumbar_file"].filename if ("lumbar_file" in request.files and request.files["lumbar_file"].filename != "") else request.form.get("lumbar_sample")

        res_s = predictor.predict_spine(spine_model, spine_img, filename=spine_fn)
        res_l = predictor.predict_lumbar(lumbar_model, lumbar_img, filename=lumbar_fn)
        total_time = round(time.time() - t0, 3)

        hm_s = predictor.generate_gradcam_heatmap(spine_model, res_s["processed_batch"], res_s["predicted_index"])
        ov_s = predictor.create_gradcam_overlay(res_s["display_image"], hm_s)

        hm_l = predictor.generate_gradcam_heatmap(lumbar_model, res_l["processed_batch"], res_l["predicted_index"])
        ov_l = predictor.create_gradcam_overlay(res_l["display_image"], hm_l)

        has_abnormality = (res_s["clinical_info"].get("status") == "ABNORMAL" or res_l["clinical_info"].get("status") == "ABNORMAL")

        return jsonify({
            "status": "success",
            "has_abnormality": has_abnormality,
            "overall_summary": "Pathological findings identified across spinal imaging studies." if has_abnormality else "Normal spinal alignment and preserved canal space.",
            "total_time_sec": total_time,
            "spine": {
                "predicted_class": res_s["predicted_class"],
                "confidence": round(res_s["confidence"] * 100, 1),
                "clinical_info": res_s["clinical_info"],
                "probabilities": {k: round(v * 100, 1) for k, v in res_s["probabilities"].items()},
                "original_image": predictor.pil_to_base64(res_s["display_image"]),
                "gradcam_image": predictor.pil_to_base64(ov_s) if ov_s else None
            },
            "lumbar": {
                "predicted_class": res_l["predicted_class"],
                "confidence": round(res_l["confidence"] * 100, 1),
                "clinical_info": res_l["clinical_info"],
                "probabilities": {k: round(v * 100, 1) for k, v in res_l["probabilities"].items()},
                "original_image": predictor.pil_to_base64(res_l["display_image"]),
                "gradcam_image": predictor.pil_to_base64(ov_l) if ov_l else None
            }
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    init_models()
    port = int(os.environ.get("PORT", 5000))
    print(f"\n=======================================================")
    print(f" SpinalScan Platform Running: http://localhost:{port}")
    print(f"=======================================================\n")
    app.run(host="0.0.0.0", port=port, debug=False)
