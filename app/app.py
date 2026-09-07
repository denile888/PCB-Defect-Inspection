"""
PCB Defect Inspection - Dashboard & GUI

Covers four of the shared functional requirements in one application:

    Object Detection    - bounding boxes (trained YOLO model) + contours
                          (classical segmentation inside each box)
    Data Analysis
    Dashboard            - per-image and per-batch summary of what was found
    GUI                  - the application itself; supports uploading a
                          single image or a whole folder at once
    Reporting            - one-click PDF export of the current session's
                          results (extra effort item)

Preprocessing (grayscale Gaussian / Canny / morphological / combined) is
available as a preview step, and any of those techniques can also be applied
before detection runs. The technique selector defaults to the team's
confirmed best-performing configuration - Gaussian heavy (A3) followed by
Canny medium (B2), validated across multiple training seeds against the
individual techniques on their own - rather than to "no preprocessing".
Anyone can still switch it for a demo or a side-by-side comparison; the
point is that the system runs the winner unless told otherwise, instead of
requiring that choice to be made correctly by hand every session.

This project accepts still images only. Real-time video ingestion (the
other extra-effort item in the brief) was assessed and deliberately left
out, since every input in this dataset - and every input this app is built
to accept - is a single photograph, not a video stream.

Run with:
    streamlit run app.py

No trained weights required to explore the app: without a model file, the
"classical" detector (reference-image differencing) stands in, so every tab
still works end to end. Loading a real trained .pt file switches to genuine
YOLO detection automatically.

How this file executes (worth knowing if you have not used Streamlit
before): there is no main() and no event loop. Streamlit reruns this
ENTIRE script from top to bottom every time something changes - a file is
uploaded, a slider moves, a button is clicked. Nothing in the file below
persists between reruns on its own; the sidebar's current selections are
just the values its widgets return on THIS run. The one exception is
`st.session_state`, set up in the block just below - anything stored there
survives from one rerun to the next, which is how detections keep
accumulating across every image and every upload in a session instead of
being wiped out the moment the script restarts.
"""

from pathlib import Path
from dataclasses import asdict

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from core.preprocessing import TECHNIQUES, DEFAULT_TECHNIQUE, apply_technique, to_gray
from core.contours import find_contour, find_all_defects_by_reference, draw_contour
from core.report import build_report
from core.discovery import find_project_root, find_matching_reference, discover_trained_weights
from core.teammate_techniques import CONTRIBUTORS

CLASS_NAMES = ["missing_hole", "mouse_bite", "open_circuit",
              "short", "spur", "spurious_copper"]

st.set_page_config(page_title="PCB Defect Inspection", layout="wide")

# Resolved once per session, not re-searched on every rerun - the project
# layout does not change while the app is running. None if the app is
# being run outside the project folder structure (e.g. copied somewhere
# standalone); every feature below degrades to manual upload in that case
# rather than failing.
PROJECT_ROOT = find_project_root()
PCB_USED_DIR = (PROJECT_ROOT / "data" / "raw" / "PCB_USED") if PROJECT_ROOT else None


# =======================================================================
# SESSION STATE
# =======================================================================
# Detections accumulate here across every image processed in this session,
# which is what lets the Dashboard tab summarise a whole batch rather than
# just whatever is on screen right now.

if "detections" not in st.session_state:
    st.session_state.detections = []       # list of dict, one row per object
if "annotated_images" not in st.session_state:
    st.session_state.annotated_images = {} # {filename: BGR array}, for the PDF report
if "model" not in st.session_state:
    st.session_state.model = None
if "model_name" not in st.session_state:
    st.session_state.model_name = None
if "model_source" not in st.session_state:
    st.session_state.model_source = None  # "auto" | "manual" | None
if "weights_auto_load_attempted" not in st.session_state:
    # Auto-load only ever runs once per session, on the first script run.
    # Without this flag, a person who deliberately clears the model (by
    # uploading nothing after a failed load, for instance) would find it
    # silently reappearing on the very next rerun.
    st.session_state.weights_auto_load_attempted = False


# =======================================================================
# HELPERS
# =======================================================================

def read_upload(uploaded_file):
    """Decode a Streamlit upload into a BGR image."""
    data = np.frombuffer(uploaded_file.getvalue(), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


@st.cache_resource(show_spinner=False)
def load_yolo_model(weights_bytes, filename):
    """Load a YOLO model from uploaded weight bytes.

    Cached on the exact bytes, so re-running the app with the same upload
    does not reload the model from disk every time.
    """
    from ultralytics import YOLO
    temp_path = Path("uploaded_weights") / filename
    temp_path.parent.mkdir(exist_ok=True)
    temp_path.write_bytes(weights_bytes)
    return YOLO(str(temp_path))


@st.cache_resource(show_spinner=False)
def load_yolo_model_from_path(path_str):
    """Load a YOLO model directly from a file already on disk.

    Used for the auto-detected weights under runs/ - there is no upload
    to save first, unlike load_yolo_model() above, so this is simpler:
    hand the path straight to Ultralytics. Cached on the path string, so
    switching back and forth between two discovered runs does not reload
    either from disk more than once per session.
    """
    from ultralytics import YOLO
    return YOLO(path_str)


def run_detection(detection_input_bgr, original_gray, reference_gray,
                  sensitivity, conf_threshold):
    """Return a list of (class_name, confidence, box_xyxy, source) tuples.

    IMPORTANT: `detection_input_bgr` (which may be preprocessed) is used
    only for the YOLO branch, since a trained model expects whatever it was
    trained on. The classical fallback always runs on `original_gray` and
    the untouched reference, never on a preprocessed image - edge detection
    and morphological enhancement are not stable enough under tiny
    misalignment for differencing two independently processed images to
    work. Two Canny outputs of the "same" photo, even with identical
    settings, can differ by thousands of pixels along every real edge just
    from sub-pixel misalignment, since edge detection amplifies exactly the
    kind of tiny discrepancy that raw photographic pixels absorb.
    """
    if st.session_state.model is not None:
        results = st.session_state.model.predict(detection_input_bgr,
                                                  conf=conf_threshold, verbose=False)[0]
        detections = []
        for box in results.boxes:
            class_id = int(box.cls[0])
            class_name = results.names.get(class_id, f"class_{class_id}")
            confidence = float(box.conf[0])
            xyxy = tuple(box.xyxy[0].tolist())
            detections.append((class_name, confidence, xyxy, "yolo"))
        return detections

    if reference_gray is None:
        return []

    contour_results = find_all_defects_by_reference(
        original_gray, reference_gray, sensitivity=sensitivity)
    detections = []
    for result in contour_results:
        x, y, w, h = result.bbox_px
        detections.append(("unclassified", 0.0, (x, y, x + w, y + h), "classical"))
    return detections


def annotate(image_bgr, original_gray, detections, reference_gray, margin, min_area):
    """Draw boxes and contours; return the annotated image and a rows list.

    Annotation is always drawn on the ORIGINAL colour image, and contour
    refinement always reads from `original_gray` / `reference_gray` - never
    from a preprocessed view - for the same reason run_detection() avoids
    it for the classical path.
    """
    annotated = image_bgr.copy()
    rows = []

    for class_name, confidence, box_xyxy, source in detections:
        x1, y1, x2, y2 = [int(v) for v in box_xyxy]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = f"{class_name} {confidence:.2f}" if source == "yolo" else class_name
        cv2.putText(annotated, label, (x1, max(y1 - 6, 12)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        contour_result = find_contour(original_gray, box_xyxy,
                                      reference_gray=reference_gray,
                                      margin=margin, min_area=min_area)
        draw_contour(annotated, contour_result, color=(0, 255, 255), thickness=2)

        rows.append({
            "class": class_name,
            "confidence": round(confidence, 3) if source == "yolo" else None,
            "source": source,
            "box_x": x1, "box_y": y1,
            "box_w": x2 - x1, "box_h": y2 - y1,
            "box_area_px": (x2 - x1) * (y2 - y1),
            "contour_found": contour_result.found,
            "contour_method": contour_result.method,
            "contour_area_px": round(contour_result.area_px, 1),
            "contour_perimeter_px": round(contour_result.perimeter_px, 1),
        })

    return annotated, rows


@st.cache_data(show_spinner=False)
def read_reference_from_disk(path_str):
    """Read and grayscale a reference board from a local path.

    Cached on the path string. A batch can contain several images from the
    same board design, and this project's auto-matching would otherwise
    re-read that board's reference file from disk once per image in the
    batch on every single script rerun.
    """
    return to_gray(cv2.imread(path_str))


def resolve_reference(filename, manual_reference_gray):
    """Pick the reference for one image: auto-match first, manual upload
    as the fallback, None if neither is available.

    Returns (reference_gray, source_label) so the caller can show which
    one was actually used - this is automatic now, and automatic
    behaviour that cannot be inspected is much harder to trust.
    """
    auto_path = find_matching_reference(filename, PCB_USED_DIR)
    if auto_path is not None:
        return read_reference_from_disk(str(auto_path)), f"auto-matched: {auto_path.name}"
    if manual_reference_gray is not None:
        return manual_reference_gray, "manually uploaded"
    return None, "none available"


# =======================================================================
# SIDEBAR - configuration shared across every tab
# =======================================================================

st.sidebar.title("Configuration")

st.sidebar.subheader("Preprocessing")
st.sidebar.caption(
    "Each contributor's own techniques are listed under their name. "
    "Picking any option makes it the active one - the others stay "
    "visible for comparison, but only one drives detection at a time.")

if "active_owner" not in st.session_state:
    # Defaults to the team's confirmed winner (melvinwongkakian's A3+B2),
    # not to either teammate's technique - see DEFAULT_TECHNIQUE in
    # core/preprocessing.py for that decision. Touching any dropdown below
    # changes this via its on_change callback.
    st.session_state.active_owner = "melvin"


def _set_active_owner(owner):
    """on_change callback: whichever dropdown was just touched becomes the
    active technique. Streamlit runs this before the script reruns, so by
    the time the rest of this file reads active_owner, it is already
    up to date."""
    st.session_state.active_owner = owner


# --- melvinwongkakian's own techniques ---------------------------------
name_col, select_col = st.sidebar.columns([1, 2])
name_col.markdown("**Melvin Wong Ka Kian**")
technique_choices = ["none"] + list(TECHNIQUES)
technique_name = select_col.selectbox(
    "Technique", technique_choices,
    index=technique_choices.index(DEFAULT_TECHNIQUE),
    label_visibility="collapsed", key="melvin_technique",
    on_change=_set_active_owner, args=("melvin",),
    help="Applied before detection runs, and previewed in the first tab. "
         "Defaults to the team's validated best-performing configuration.")

technique_params = {}
if technique_name != "none":
    with st.sidebar.expander("Melvin Wong Ka Kian's parameters", expanded=False):
        if technique_name.startswith("gaussian"):
            technique_params["ksize"] = st.slider("Kernel size", 3, 15, 5, step=2)
            technique_params["sigma"] = st.slider("Sigma", 0.1, 4.0, 1.0)
        elif technique_name.startswith("canny"):
            technique_params["low"] = st.slider("Low threshold", 0, 200, 50)
            technique_params["high"] = st.slider("High threshold", 0, 300, 150)
            technique_params["weight"] = st.slider("Edge weight", 0.0, 1.0, 0.5)
        elif technique_name.startswith("morph"):
            technique_params["ksize"] = st.slider("Structuring element size", 3, 31, 15, step=2)

# --- teammates' techniques -----------------------------------------
# Reproduced from their own notebooks - see core/teammate_techniques.py
# for the exact source and the parameter values, copied verbatim from
# their code rather than re-derived here.
teammate_choices = {}   # owner_key -> the set name currently selected

for owner_key, display_name in [("lee", "Lee Wan Ching"), ("lim", "Lim Sze Ping")]:
    contributor = CONTRIBUTORS[display_name]
    name_col, select_col = st.sidebar.columns([1, 2])
    name_col.markdown(f"**{display_name}**")
    chosen_set = select_col.selectbox(
        display_name, list(contributor["sets"]),
        format_func=lambda s: s.replace("set", "Set "),
        label_visibility="collapsed", key=f"{owner_key}_technique",
        on_change=_set_active_owner, args=(owner_key,))
    teammate_choices[owner_key] = chosen_set

    with st.sidebar.expander(f"{display_name}'s parameters", expanded=False):
        st.json(contributor["sets"][chosen_set])

# --- resolve which one is actually active -------------------------------
active_owner = st.session_state.active_owner

if active_owner == "melvin":
    active_label = technique_name
    active_params = technique_params

    def apply_active_technique(image_bgr):
        return apply_technique(image_bgr, technique_name, **technique_params)
else:
    owner_names = {"lee": "Lee Wan Ching", "lim": "Lim Sze Ping"}
    display_name = owner_names[active_owner]
    chosen_set = teammate_choices[active_owner]
    contributor = CONTRIBUTORS[display_name]
    active_label = f"{display_name} - {chosen_set.replace('set', 'Set ')}"
    active_params = contributor["sets"][chosen_set]

    def apply_active_technique(image_bgr, _fn=contributor["function"],
                               _params=active_params):
        return _fn(image_bgr, **_params)

st.sidebar.divider()
st.sidebar.success(f"Active: {active_label}")
if active_owner == "melvin" and technique_name == DEFAULT_TECHNIQUE:
    st.sidebar.caption(
        "This is the team's confirmed winning configuration from the "
        "comparative study, not just this app's default choice.")

st.sidebar.subheader("Detection")

# Auto-load the winning configuration's trained weights on the very first
# run of the session, so detection uses real YOLO output with zero clicks
# - matching how the preprocessing technique already defaults to the
# winner instead of requiring it to be picked by hand. Runs at most once
# per session; a manual upload below always takes priority from that
# point on and is never silently overwritten by this block again.
discovered_weights = discover_trained_weights(PROJECT_ROOT)

if (not st.session_state.weights_auto_load_attempted
        and st.session_state.model is None and discovered_weights):
    auto_label, auto_path = discovered_weights[0]
    try:
        st.session_state.model = load_yolo_model_from_path(str(auto_path))
        st.session_state.model_name = auto_label
        st.session_state.model_source = "auto"
    except Exception as error:
        st.sidebar.warning(f"Could not auto-load {auto_path.name}: {error}")
st.session_state.weights_auto_load_attempted = True

if discovered_weights and len(discovered_weights) > 1:
    with st.sidebar.expander(
            f"{len(discovered_weights)} trained runs found under runs/", expanded=False):
        labels = [label for label, _ in discovered_weights]
        chosen = st.selectbox("Switch to a different run", labels, index=0)
        if chosen != st.session_state.model_name:
            chosen_path = dict(discovered_weights)[chosen]
            st.session_state.model = load_yolo_model_from_path(str(chosen_path))
            st.session_state.model_name = chosen
            st.session_state.model_source = "auto"

weights_file = st.sidebar.file_uploader(
    "Or upload YOLO weights (.pt) manually", type=["pt"],
    help="Overrides the auto-detected run above. To go back to it, "
         "restart the app.")
if weights_file is not None and weights_file.name != st.session_state.model_name:
    with st.sidebar.status(f"Loading {weights_file.name}...", expanded=False):
        try:
            st.session_state.model = load_yolo_model(weights_file.getvalue(),
                                                      weights_file.name)
            st.session_state.model_name = weights_file.name
            st.session_state.model_source = "manual"
        except Exception as error:
            # An invalid or corrupted .pt file must not crash the whole
            # app - the auto-loaded (or absent) model from before this
            # upload stays active, and the person gets a message telling
            # them what to do next instead of a stack trace.
            st.sidebar.error(f"Could not load {weights_file.name} as a "
                            f"YOLO model: {error}")

if st.session_state.model is not None:
    origin = ("auto-detected from runs/" if st.session_state.model_source == "auto"
             else "manually uploaded")
    st.sidebar.success(f"Model loaded ({origin}): {st.session_state.model_name}")
    conf_threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.25)
else:
    st.sidebar.info("No model loaded and none found under runs/ - using "
                    "classical reference-difference detection as a "
                    "fallback. Upload a trained .pt file above for real "
                    "defect classification.")
    conf_threshold = 0.25

st.sidebar.subheader("Reference board")
st.sidebar.caption(
    "Matched automatically per image from data/raw/PCB_USED/ when the "
    "filename follows this project's <board_id>_<defect>_<n> naming "
    "convention. Upload one manually below only as a fallback for a board "
    "design that is not in PCB_USED, or when auto-matching is not "
    "applicable.")
reference_file = st.sidebar.file_uploader(
    "Defect-free reference (fallback, optional)", type=["jpg", "jpeg", "png"],
    help="Used only for images where automatic matching finds nothing. "
         "Required for the classical fallback detector on such images; "
         "without it, their contours fall back to local thresholding.")
manual_reference_gray = None
if reference_file is not None:
    manual_reference_gray = to_gray(read_upload(reference_file))
    # Deliberately NOT run through the selected preprocessing technique.
    # See the note above run_detection() for why differencing needs raw
    # pixels, not an edge map or morphologically enhanced image, on both
    # sides.

sensitivity = 99.9
if st.session_state.model is None:
    sensitivity = st.sidebar.slider(
        "Classical detector sensitivity (percentile)", 95.0, 99.99, 99.9,
        help="Lower catches fainter differences but raises false positives. "
             "A single global threshold cannot fully compensate for very "
             "faint defects - this is why a trained model is the intended "
             "primary path.")

margin = st.sidebar.slider("Contour search margin (px)", 0, 40, 12)
min_area = st.sidebar.slider("Minimum contour area (px)", 1, 200, 15)

if st.sidebar.button("Clear session results"):
    st.session_state.detections = []
    st.session_state.annotated_images = {}
    st.sidebar.success("Cleared.")


# =======================================================================
# MAIN AREA
# =======================================================================

st.title("PCB Defect Inspection")

uploaded_files = st.file_uploader(
    "Upload one or more board images",
    type=["jpg", "jpeg", "png"], accept_multiple_files=True,
    help="Select multiple files, or an entire folder's contents at once, "
         "to process a batch in one pass.")

tab_preprocess, tab_detect, tab_dashboard = st.tabs(
    ["Preprocessing", "Detection & Contours", "Dashboard"])

if not uploaded_files:
    st.info("Upload at least one image to begin.")
else:
    # Runs once per uploaded file, EVERY time the script reruns - so on a
    # batch of 5 images, changing one sidebar slider reprocesses all 5, not
    # just the one that changed. There is no per-image caching here; for
    # a "simple" app this is an acceptable cost, but it is the first thing
    # to optimise if large batches start feeling slow.
    for uploaded_file in uploaded_files:
        image_bgr = read_upload(uploaded_file)
        if image_bgr is None:
            st.warning(f"Could not read {uploaded_file.name}, skipping.")
            continue

        original_gray = to_gray(image_bgr)
        reference_gray, reference_source = resolve_reference(
            uploaded_file.name, manual_reference_gray)

        processed_bgr = image_bgr
        processed_gray = None
        if active_label != "none":
            processed_gray = apply_active_technique(image_bgr)
            processed_bgr = cv2.cvtColor(processed_gray, cv2.COLOR_GRAY2BGR)

        # ---------------- Preprocessing tab ----------------
        with tab_preprocess:
            st.subheader(uploaded_file.name)
            st.caption(f"Reference board: {reference_source}")
            col_before, col_after = st.columns(2)
            col_before.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
                             caption="Original", width='stretch')
            if active_label != "none":
                col_after.image(processed_gray, caption=f"Processed ({active_label})",
                               width='stretch', clamp=True)
            else:
                col_after.info("Select a technique in the sidebar to see it here.")

        # ---------------- Detection tab ----------------
        # detect_source feeds the YOLO model only, matching what it was
        # trained on. original_gray / reference_gray feed the classical
        # fallback and every contour refinement - see run_detection() and
        # annotate() for why those two paths must not mix.
        detect_source = processed_bgr if active_label != "none" else image_bgr
        detections = run_detection(detect_source, original_gray, reference_gray,
                                   sensitivity, conf_threshold)
        annotated, rows = annotate(image_bgr, original_gray, detections,
                                   reference_gray, margin, min_area)

        for row in rows:
            row["image"] = uploaded_file.name
            row["reference_source"] = reference_source
        st.session_state.detections.extend(rows)
        # Kept for the PDF report, which embeds the same annotated view
        # shown in the Detection tab rather than re-running detection.

        st.session_state.annotated_images[uploaded_file.name] = annotated

        with tab_detect:
            st.subheader(uploaded_file.name)
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                    caption=f"{len(rows)} object(s) - red box = detection, "
                            f"yellow outline = contour",
                    width='stretch')
            if rows:
                st.dataframe(pd.DataFrame(rows).drop(columns=["image"]),
                           width='stretch')
            else:
                st.info("No objects detected in this image.")


# =======================================================================
# DASHBOARD TAB - summarises everything processed this session
# =======================================================================

with tab_dashboard:
    st.subheader("Session summary")

    if not st.session_state.detections:
        st.info("Process at least one image to populate the dashboard.")
    else:
        df = pd.DataFrame(st.session_state.detections)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Images processed", df["image"].nunique())
        col2.metric("Objects found", len(df))
        col3.metric("With a resolved contour", int(df["contour_found"].sum()))
        col4.metric("Mean contour area (px)",
                   f"{df.loc[df['contour_found'], 'contour_area_px'].mean():.0f}"
                   if df["contour_found"].any() else "-")

        left, right = st.columns(2)

        with left:
            st.markdown("**Objects per class**")
            st.bar_chart(df["class"].value_counts())

        with right:
            st.markdown("**Contour area distribution (px)**")
            found = df[df["contour_found"]]
            if len(found):
                st.bar_chart(found["contour_area_px"])
            else:
                st.info("No resolved contours yet.")

        st.markdown("**Full results table**")
        st.dataframe(df, width='stretch')

        st.markdown("**Export**")
        export_left, export_right = st.columns(2)

        with export_left:
            st.download_button(
                "Download results as CSV",
                df.to_csv(index=False).encode("utf-8"),
                file_name="inspection_results.csv",
                mime="text/csv",
            )

        with export_right:
            # Built fresh on every click rather than cached, since the
            # report always reflects whatever is currently in the session
            # (including anything processed since the last click).
            pdf_bytes = build_report(
                df, st.session_state.annotated_images,
                technique_name=active_label,
                model_name=st.session_state.model_name,
            )
            st.download_button(
                "Download PDF report",
                pdf_bytes,
                file_name="inspection_report.pdf",
                mime="application/pdf",
            )

        st.caption(
            "Areas are in pixels. Once the calibration module provides a "
            "pixels-per-mm scale for the board in view, this table is the "
            "natural place to add a physical-units column alongside these."
        )
