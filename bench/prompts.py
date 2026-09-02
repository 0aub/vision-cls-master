"""Class-descriptive text prompts for BiomedCLIP zero-shot (Phase B3).

Five clinically phrased prompts per class; the text embeddings are L2-normalised
and averaged into one class vector, which is what open_clip's zero-shot recipe
does. Every prompt is reproduced in log/bench-biomedclip-prompts.json and in the
report, as the brief requires.
"""

PROMPTS_5CLASS = {
    "AVM": [
        "a wireless capsule endoscopy image of small bowel angiodysplasia with a visible vascular lesion",
        "an endoscopy image showing an arteriovenous malformation of the small intestine",
        "capsule endoscopy frame with a bright red spider-like vascular ectasia on the mucosa",
        "small bowel mucosa with angiodysplasia, a cluster of dilated superficial vessels",
        "a photograph of intestinal angioectasia seen during capsule endoscopy",
    ],
    "Erosion": [
        "a wireless capsule endoscopy image of a small bowel mucosal erosion",
        "an endoscopy image showing a shallow superficial break in the intestinal mucosa",
        "capsule endoscopy frame with a small erosion and surrounding erythema",
        "small bowel mucosa with a superficial erosive lesion without deep excavation",
        "a photograph of mucosal erosion of the small intestine seen during capsule endoscopy",
    ],
    "Normal": [
        "a wireless capsule endoscopy image of normal small bowel mucosa",
        "an endoscopy image showing healthy intestinal villi with no lesion",
        "capsule endoscopy frame of unremarkable normal small bowel",
        "normal small intestinal mucosa without erosion, ulcer or vascular lesion",
        "a photograph of a healthy small bowel lumen seen during capsule endoscopy",
    ],
    "Ulcer": [
        "a wireless capsule endoscopy image of a small bowel ulcer",
        "an endoscopy image showing a deep mucosal ulceration with a white fibrinous base",
        "capsule endoscopy frame with an ulcer crater surrounded by inflamed mucosa",
        "small bowel mucosa with an excavated ulcerative lesion",
        "a photograph of small intestinal ulceration seen during capsule endoscopy",
    ],
    "Xanthoma": [
        "a wireless capsule endoscopy image of a small bowel xanthoma",
        "an endoscopy image showing a yellowish-white lipid-laden mucosal plaque",
        "capsule endoscopy frame with a yellow xanthomatous deposit on the intestinal mucosa",
        "small bowel mucosa with a pale yellow lipid island typical of xanthoma",
        "a photograph of intestinal xanthoma seen during capsule endoscopy",
    ],
}

# binary: Normal keeps its own prompt set; Lesion pools clinically phrased
# abnormality prompts rather than the union of the four class sets, so the two
# sides carry the same number of prompts.
PROMPTS_BINARY = {
    "Normal": PROMPTS_5CLASS["Normal"],
    "Lesion": [
        "a wireless capsule endoscopy image of abnormal small bowel mucosa with a lesion",
        "an endoscopy image showing a pathological small bowel finding such as an ulcer, erosion, angiodysplasia or xanthoma",
        "capsule endoscopy frame with a clearly abnormal mucosal lesion",
        "diseased small intestinal mucosa with a visible focal abnormality",
        "a photograph of a small bowel lesion requiring clinical attention, seen during capsule endoscopy",
    ],
}


# merged4: Erosion and Ulcer become one "mucosal break" class, so the prompt set
# spans both depths rather than pooling the two separate sets.
PROMPTS_MERGED4 = {
    "AVM": PROMPTS_5CLASS["AVM"],
    "ErosionUlcer": [
        "a wireless capsule endoscopy image of a small bowel mucosal break",
        "an endoscopy image showing an ulcer or erosion of the small intestine",
        "capsule endoscopy frame with a break in the mucosal surface, superficial or excavated",
        "small bowel mucosa with an ulcerative or erosive lesion and surrounding inflammation",
        "a photograph of a small intestinal mucosal break seen during capsule endoscopy",
    ],
    "Normal": PROMPTS_5CLASS["Normal"],
    "Xanthoma": PROMPTS_5CLASS["Xanthoma"],
}


def prompts_for(task):
    return {"5class": PROMPTS_5CLASS, "binary": PROMPTS_BINARY,
            "merged4": PROMPTS_MERGED4}[task]
