# PCB Defect Inspection - Dashboard & GUI

Covers four shared functional requirements in one application:

- **Object Detection** - bounding boxes (trained YOLO model) + contours
  (classical segmentation inside each box)
- **Data Analysis Dashboard** - per-image and per-batch summary of findings
- **GUI** - the application itself, with single or batch/folder upload
- **Reporting** (extra effort) - one-click PDF export of the current
  session's results

Preprocessing (grayscale Gaussian / Canny / morphological / combined) is
included as a preview and detection-input step.

**Scope note:** this project accepts still images only. Real-time video
ingestion (the other extra-effort item in the brief) was assessed and
deliberately left out - every input in this dataset, and every input this
app is built to accept, is a single photograph.

## Setup

```
pip install streamlit opencv-python numpy pandas ultralytics fpdf2 pillow
```

## Run

```
streamlit run app.py
```

Opens in your browser. No trained weights required to try it - without a
model file, a classical reference-differencing detector stands in, so every
tab works end to end. Upload a real trained `.pt` file in the sidebar to
switch to genuine YOLO detection.

## Files

```
app.py                  the Streamlit application
core/preprocessing.py   grayscale Gaussian / Canny / morphological / combined
core/contours.py        contour extraction (reference-based + fallback)
core/report.py          PDF report generation
core/discovery.py       auto-detects reference boards + trained weights
                        from the surrounding project folder
```

Every file has a module-level docstring explaining its role, and every
function has a docstring covering what it does and - where the reason
is not obvious - why. `app.py`'s docstring also explains Streamlit's
execution model (the whole script reruns on every interaction), since
that is the one thing about this file that is not self-evident to someone
who has not used Streamlit before.

## How to use it

The app auto-detects two things from the surrounding project folder, with
zero clicks required:

- **Trained weights.** On first load, it searches `runs/` for a folder
  whose name matches the confirmed winning configuration (`A3B2` -
  `WINNING_RUN_NAME` in `core/discovery.py`) and loads its `best.pt`
  automatically. If several matching runs exist (e.g. from the seed
  verification study), the plain run is preferred over its seed-suffixed
  variants, since that is the one that appeared in the main comparison
  table. Every other run found is still listed and selectable in a
  sidebar expander, in case the automatic guess is wrong.
- **Reference boards.** Each uploaded image is matched against
  `data/raw/PCB_USED/` using this project's `<board_id>_<defect>_<n>.jpg`
  naming convention - so a mixed batch spanning several board designs gets
  the *correct* reference for each image automatically, not one global
  reference applied to everything.

Both fall back to manual upload when auto-detection finds nothing - a new
board design not in `PCB_USED`, or no `runs/` folder at all (e.g. before
training, or the app copied somewhere standalone). A manual weights upload
always overrides the auto-detected one; to go back to it, restart the app.

1. **Upload one or more board images** - select multiple files or an entire
   folder's contents at once.
2. **Preprocessing tab** - preview any of the ten grayscale techniques.
   Defaults to the winning configuration (see below).
3. **Detection & Contours tab** - see boxes (red) and contours (yellow) per
   image, plus a results table. Each image's caption shows which reference
   board was used and where it came from.
4. **Dashboard tab** - aggregate counts, charts, CSV export, and a
   **Download PDF report** button: a summary page (headline numbers +
   per-class counts) followed by one page per processed image (its
   annotated view + its own results table, including the reference source).

## Design notes worth knowing before extending this

**The app defaults to the team's confirmed winning configuration**, not to
"no preprocessing". The sidebar's Technique selector opens on
`A3 heavy Gaussian + B2 medium Canny` - seed-verified across multiple
training runs to outperform every individual technique on its own - and
says so directly in the sidebar, so this isn't just an implementation
detail buried in code. The dropdown stays fully switchable (to `none`, or
to any individual technique) for demos and side-by-side comparison; what
changed is which option loads automatically, so the deployed system uses
the study's actual result unless someone deliberately picks something else.
`DEFAULT_TECHNIQUE` in `core/preprocessing.py` is the single place this is
set - change it there if a later result supersedes A3+B2.

**A manual weights upload used to be able to crash the whole app.** Caught
while testing the auto-detection feature above: uploading a corrupted or
non-model `.pt` file raised an unhandled exception all the way up through
Streamlit. The fix wraps that load in a try/except, matching what the
auto-load path already did - an invalid upload now shows a sidebar error
and leaves whatever model was active beforehand in place, rather than
taking the whole app down.

**Contour extraction always reads raw grayscale pixels, never a
preprocessed (edge-detected or morphologically enhanced) image**, even when
a technique is selected in the sidebar. This was a real bug caught during
testing: diffing two independently Canny-processed images - even with
identical settings on both sides - produces thousands of spurious
mismatches from ordinary sub-pixel misalignment, because edge detection
amplifies exactly the kind of tiny discrepancy that raw photographic pixels
absorb. The selected technique only ever feeds the *YOLO model* (matching
what it was trained on); contour math is deliberately kept separate.

**The classical fallback detector floors its threshold rather than trusting
Otsu blindly.** On a real test image (a faint `mouse_bite`), Otsu chose a
threshold of 1 out of 255, which classified 34% of the entire board as
"different" and returned one contour the size of the whole image. The fix
takes the *strictest* of three candidates (Otsu, a percentile-based value,
and a fixed floor) rather than the most permissive - a union of masks would
only have made the failure worse, since the problem was Otsu already being
too permissive.

**The classical fallback is not a substitute for a trained model.** Verified
against real ground truth: it recovers `missing_hole` defects exactly (3/3)
but only partially for fainter classes, and even after the threshold fix,
one small edge-registration artifact still showed up in one test case
(clearly labelled, not silently hidden). This is inherent to global
thresholding on a single image pair, not a parameter that can be tuned
away - it's the reason a trained detector is the primary path.

**The PDF report is built fresh from session state on every click**, not
cached, so it always reflects everything processed so far in the session -
including images added after an earlier download.

**Not yet built:** the Image Calibration requirement (pixel-to-physical-unit
scaling). The dashboard and PDF report both show contour and box areas in
pixels only. The Dashboard tab has a caption marking where a physical-units
column would attach once a mm-per-pixel scale is available - the board's
corner mounting holes are a natural calibration reference already present
in every image in this dataset.

