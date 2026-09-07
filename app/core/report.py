"""
PDF export - the "Reporting" (extra effort) requirement.

Turns one inspection session into a single PDF: a summary page (headline
numbers + a per-class count table), followed by one page per image that was
processed (its annotated view, plus a table of what was found on it).

This module only builds PDF bytes - it does not know about Streamlit, file
uploads, or the model. app.py collects the results table and the annotated
images during a session, then calls build_report() once, when the person
clicks the "Generate PDF report" button on the Dashboard tab.
"""

from datetime import datetime

import cv2
import numpy as np
from fpdf import FPDF
from PIL import Image

PAGE_MARGIN_MM = 15
MAX_EMBEDDED_IMAGE_WIDTH_PX = 500   # keeps the PDF a sane file size


def build_report(results_df, annotated_images, technique_name="none",
                 model_name=None):
    """Build the PDF and return it as bytes, ready for st.download_button.

    Parameters
    ----------
    results_df
        The same table shown on the Dashboard tab: one row per detected
        object, across every image processed this session. Must have a
        "image" column identifying which file each row belongs to.
    annotated_images
        dict of {filename: BGR numpy array} - the boxes-and-contours image
        drawn for each processed file. If a filename in results_df has no
        matching entry here, that page notes the image was not available
        rather than failing the whole report.
    technique_name
        The preprocessing technique active when this session ran, recorded
        on the summary page so the report is self-describing.
    model_name
        The YOLO weights file name if one was loaded, or None if the
        classical fallback detector was used instead.
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=PAGE_MARGIN_MM)

    _add_summary_page(pdf, results_df, technique_name, model_name)

    for image_name in results_df["image"].unique():
        image_rows = results_df[results_df["image"] == image_name]
        annotated_bgr = annotated_images.get(image_name)
        _add_image_page(pdf, image_name, image_rows, annotated_bgr)

    return bytes(pdf.output())


# ---------------------------------------------------------------------
# Summary page
# ---------------------------------------------------------------------

def _add_summary_page(pdf, results_df, technique_name, model_name):
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "PCB Defect Inspection Report", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 10)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    detector = model_name if model_name else "classical fallback (no trained model loaded)"
    pdf.cell(0, 6, f"Generated: {generated_at}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"Preprocessing: {technique_name}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"Detector: {detector}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    _add_headline_metrics(pdf, results_df)
    pdf.ln(4)
    _add_class_breakdown_table(pdf, results_df)


def _add_headline_metrics(pdf, results_df):
    """The same four numbers shown as st.metric() cards on the dashboard."""
    images_processed = results_df["image"].nunique()
    objects_found = len(results_df)
    with_contour = int(results_df["contour_found"].sum())
    resolved = results_df.loc[results_df["contour_found"], "contour_area_px"]
    mean_area = f"{resolved.mean():.0f} px" if len(resolved) else "-"

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Summary", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 10)
    for label, value in [
        ("Images processed", images_processed),
        ("Objects found", objects_found),
        ("With a resolved contour", with_contour),
        ("Mean contour area", mean_area),
    ]:
        pdf.cell(0, 6, f"{label}: {value}", new_x="LMARGIN", new_y="NEXT")


def _add_class_breakdown_table(pdf, results_df):
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Objects per class", new_x="LMARGIN", new_y="NEXT")

    counts = results_df["class"].value_counts()

    pdf.set_font("Helvetica", "", 10)
    with pdf.table(col_widths=(60, 30), text_align=("LEFT", "RIGHT")) as table:
        header = table.row()
        header.cell("Class")
        header.cell("Count")
        for class_name, count in counts.items():
            row = table.row()
            row.cell(str(class_name))
            row.cell(str(int(count)))


# ---------------------------------------------------------------------
# One page per processed image
# ---------------------------------------------------------------------

def _add_image_page(pdf, image_name, image_rows, annotated_bgr):
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, image_name, new_x="LMARGIN", new_y="NEXT")

    if "reference_source" in image_rows.columns and len(image_rows):
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(0, 5, f"Reference board: {image_rows['reference_source'].iloc[0]}",
                new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

    if annotated_bgr is not None:
        pil_image = _bgr_to_pil(annotated_bgr)
        # Fit to the page width, minus margins on both sides.
        page_width_mm = pdf.w - 2 * PAGE_MARGIN_MM
        pdf.image(pil_image, w=page_width_mm)
    else:
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 6, "(annotated image not available for this entry)",
                new_x="LMARGIN", new_y="NEXT")

    pdf.ln(4)
    _add_object_table(pdf, image_rows)


def _add_object_table(pdf, image_rows):
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, f"{len(image_rows)} object(s) found", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 9)
    column_widths = (35, 25, 30, 30, 35)
    with pdf.table(col_widths=column_widths,
                  text_align=("LEFT", "RIGHT", "RIGHT", "LEFT", "RIGHT")) as table:
        header = table.row()
        for title in ["Class", "Confidence", "Box size (px)", "Contour", "Area (px)"]:
            header.cell(title)

        for _, detected_object in image_rows.iterrows():
            row = table.row()
            row.cell(str(detected_object["class"]))
            confidence = detected_object["confidence"]
            row.cell(f"{confidence:.2f}" if confidence is not None else "-")
            row.cell(f"{detected_object['box_w']} x {detected_object['box_h']}")
            row.cell(detected_object["contour_method"])
            row.cell(f"{detected_object['contour_area_px']:.0f}"
                    if detected_object["contour_found"] else "-")


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _bgr_to_pil(image_bgr, max_width=MAX_EMBEDDED_IMAGE_WIDTH_PX):
    """Convert a BGR numpy image to a size-capped PIL image for embedding.

    PCB photos in this project are typically 2000-3000 px wide. Embedding
    them at full resolution makes the PDF large for no visible benefit at
    report scale, so anything wider than max_width is downscaled first.
    """
    height, width = image_bgr.shape[:2]
    if width > max_width:
        scale = max_width / width
        image_bgr = cv2.resize(image_bgr, (max_width, int(height * scale)))
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)
