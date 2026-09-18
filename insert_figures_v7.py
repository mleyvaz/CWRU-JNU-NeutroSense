# -*- coding: utf-8 -*-
"""
Inserts PNG figures into CWRU_JNU_NeutroSense_INGENIUS_v3.docx
replacing [Insert FigX_....png] placeholders.
"""
import os, sys, copy
from docx import Document
from docx.shared import Cm, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from lxml import etree

OUT = os.path.dirname(os.path.abspath(__file__))

if "--submission" in sys.argv:
    IN_DOC  = os.path.join(OUT, "CWRU_JNU_NeutroSense_v7_SUBMISSION.docx")
    OUT_DOC = os.path.join(OUT, "CWRU_JNU_NeutroSense_v7_SUBMISSION_FINAL.docx")
else:
    IN_DOC  = os.path.join(OUT, "CWRU_JNU_NeutroSense_v7.docx")
    OUT_DOC = os.path.join(OUT, "CWRU_JNU_NeutroSense_v7_FINAL.docx")

# Map placeholder text → (image file, width in cm)
FIGURES = {
    "Fig1_Architecture.png":            ("Fig1_Architecture.png",            15.0),
    "Fig2_ConfusionMatrices_v4.png":    ("Fig2_ConfusionMatrices_v4.png",    16.0),
    "Fig3_FBands_JNU_v7.png":           ("Fig3_FBands_JNU_v7.png",           14.0),
    "Fig4_Decile_JNU_v7.png":           ("Fig4_Decile_JNU_v7.png",           14.0),
    "Fig5_RiskCoverage_JNU_v7.png":     ("Fig5_RiskCoverage_JNU_v7.png",     13.0),
}

def find_placeholder(para):
    """Return the figure filename if this paragraph contains [Insert FigX...]"""
    text = para.text
    for key in FIGURES:
        if key in text:
            return key
    return None

def clean_caption(para, key):
    """Remove [Insert FigX_....png] from the caption paragraph text."""
    for run in para.runs:
        if key in run.text:
            run.text = run.text.replace(f"[Insert {key}]", "").strip()
        # also clean trailing whitespace
        run.text = run.text.rstrip()

def insert_image_before(doc, para_idx, img_path, width_cm):
    """
    Insert a centered image paragraph before paragraphs[para_idx].
    Returns the new paragraph.
    """
    # Add picture to a temporary paragraph at the end, then move it
    new_para = doc.add_paragraph()
    new_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    new_para.paragraph_format.space_before = Pt(6)
    new_para.paragraph_format.space_after  = Pt(2)
    run = new_para.add_run()
    run.add_picture(img_path, width=Cm(width_cm))

    # Move the new paragraph's XML element to the right position
    target_para = doc.paragraphs[para_idx]
    target_para._element.addprevious(new_para._element)
    return new_para

doc = Document(IN_DOC)

# We iterate in reverse so that inserting elements doesn't shift indices
# First pass: collect (index, key) pairs
to_insert = []
for i, para in enumerate(doc.paragraphs):
    key = find_placeholder(para)
    if key:
        to_insert.append((i, key))

print(f"Found {len(to_insert)} figure placeholders:")
for idx, key in to_insert:
    print(f"  para[{idx}]: {key}")

# Process in reverse order to preserve indices
for para_idx, key in reversed(to_insert):
    img_file, width = FIGURES[key]
    img_path = os.path.join(OUT, img_file)

    if not os.path.exists(img_path):
        print(f"  WARNING: {img_file} not found, skipping")
        continue

    # Clean caption text
    clean_caption(doc.paragraphs[para_idx], key)

    # Insert image paragraph before the caption
    insert_image_before(doc, para_idx, img_path, width)
    print(f"  Inserted {img_file} before para[{para_idx}]")

doc.save(OUT_DOC)
print(f"\nSaved: {OUT_DOC}")
