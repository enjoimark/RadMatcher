"""
Simple, effective radiology code matcher.

Strategy:
1. Exact match lookup (from exam_mappings.csv)
2. Rule-based scoring (modality, views, contrast, body part MUST match)
3. Text similarity (for tie-breaking)

No overcomplicated ML. Just what works.
"""

import re
import csv
import json
import joblib
import threading
import time
import difflib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

try:
    from rapidfuzz import fuzz as _rapidfuzz
    _FUZZY_AVAILABLE = True
except ImportError:
    _rapidfuzz = None
    _FUZZY_AVAILABLE = False

# The matcher applies hundreds of distinct regexes per query. Python's built-in
# regex cache holds only 512 entries, so with this many patterns it thrashes --
# every re.sub / re.search recompiles its pattern from scratch (this was the
# dominant training cost: ~11M redundant compiles). This unbounded cache
# compiles each pattern exactly once for the lifetime of the process.
_COMPILED_RE_CACHE = {}


def _compiled(pattern, flags=0):
    key = (pattern, flags)
    compiled = _COMPILED_RE_CACHE.get(key)
    if compiled is None:
        compiled = re.compile(pattern, flags)
        _COMPILED_RE_CACHE[key] = compiled
    return compiled


def _csub(pattern, replacement, text, flags=0):
    """Drop-in replacement for re.sub backed by a persistent compiled cache."""
    return _compiled(pattern, flags).sub(replacement, text)


def _csearch(pattern, text, flags=0):
    """Drop-in replacement for re.search backed by a persistent compiled cache."""
    return _compiled(pattern, flags).search(text)

TERM_REPLACEMENTS_PATH = Path(__file__).resolve().parent / "term_replacements.json"

_USER_REPLACEMENTS_CACHE = []
_USER_REPLACEMENTS_MTIME = None
_USER_REPLACEMENTS_LOCK = threading.Lock()


def _load_user_replacements() -> List[Tuple[str, str]]:
    global _USER_REPLACEMENTS_CACHE
    global _USER_REPLACEMENTS_MTIME

    if not TERM_REPLACEMENTS_PATH.exists():
        with _USER_REPLACEMENTS_LOCK:
            _USER_REPLACEMENTS_CACHE = []
            _USER_REPLACEMENTS_MTIME = None
        return []

    try:
        mtime = TERM_REPLACEMENTS_PATH.stat().st_mtime
    except OSError:
        return []

    with _USER_REPLACEMENTS_LOCK:
        if _USER_REPLACEMENTS_MTIME == mtime:
            return list(_USER_REPLACEMENTS_CACHE)

    try:
        with TERM_REPLACEMENTS_PATH.open("r", encoding="utf-8") as handle:
            data = json.load(handle) or {}
    except (OSError, json.JSONDecodeError):
        data = {}

    replacements = []
    for entry in data.get("replacements", []):
        if not isinstance(entry, dict):
            continue
        pattern = entry.get("pattern")
        replacement = entry.get("replacement")
        if not pattern or replacement is None:
            continue
        replacements.append((pattern, replacement))

    with _USER_REPLACEMENTS_LOCK:
        _USER_REPLACEMENTS_CACHE = list(replacements)
        _USER_REPLACEMENTS_MTIME = mtime

    return list(replacements)


REPLACEMENT_MAP = {
    # quadrants + common abbreviations
    r"\bRIGHT\s+UPPER\s+QUADRANT\b": "RUQ",
    r"\bRIGHT\s+LOWER\s+QUADRANT\b": "RLQ",
    r"\bLEFT\s+UPPER\s+QUADRANT\b": "LUQ",
    r"\bLEFT\s+LOWER\s+QUADRANT\b": "LLQ",
    r"\bR\s*U\s*Q\b": "RUQ",
    r"\bR\s*L\s*Q\b": "RLQ",
    r"\bL\s*U\s*Q\b": "LUQ",
    r"\bL\s*L\s*Q\b": "LLQ",
    r"\bGB\b": "GALLBLADDER",
    r"\bCBD\b": "BILIARY",
    # NM cardiac agents -> CARDIAC for better matching
    r"\bCARDIOLITE\b": "CARDIAC PERFUSION",
    r"\bSESTAMIBI\b": "CARDIAC PERFUSION",
    r"\bTHALLIUM\b": "CARDIAC PERFUSION",
    r"\bMYOVIEW\b": "CARDIAC PERFUSION",
    # NM hepatobiliary
    r"\bHIDA\b": "HEPATOBILIARY",
    r"\bDISIDA\b": "HEPATOBILIARY",
    # Barium studies
    r"\bBARI\b": "BARIUM",
    r"\bMBS\b": "MODIFIED BARIUM SWALLOW",

    # number words to digits for view/text matching
    r"\bZERO\b": "0",
    r"\bONE\b": "1",
    r"\bTWO\b": "2",
    r"\bTHREE\b": "3",
    r"\bFOUR\b": "4",
    r"\bFIVE\b": "5",
    r"\bSIX\b": "6",
    r"\bSEVEN\b": "7",
    r"\bEIGHT\b": "8",
    r"\bNINE\b": "9",
    r"\bTEN\b": "10",

    # contrast shorthand (must run before slash cleanup)
    r"\bW/WO\b": "WITH AND WITHOUT",
    r"\bW\s*[/\\]\s*WO\b": "WITH AND WITHOUT",
    r"\bW/O\b": "WITHOUT",
    r"\bW\s*[/\\]\s*O\b": "WITHOUT",
    r"\bW\s*[/\\]\s*\b": "WITH",
    r"\bW\b": "WITH",
    r"\bWO\b": "WITHOUT",

    # symbols etc cleanup
    r"\-": " ",
    r"\,": " ",
    r"\_": " ",
    r"\+": " PLUS ",
    r"&": " AND ",
    r"#AP": " AND ",
    r"\\": " \\ ",
    r"\/": " / ",
    r"\(": " ",
    r"\)": " ",
    r"\;": " ",
    r"-\d{2}\b": " ",
    r"(\d)(V\w*)": r"\1 \2",

    # extremities
    r"\bEX(T|TREMITY|TREMITIES)?\b": "EXTREMITY",
    r"\bLE\b": "LOWER EXTREMITY",
    r"\bUE\b": "UPPER EXTREMITY",
    r"\bU(P(PER)?)?\s*EX(TREM(ITY)?)?\b": "UPPER EXTREMITY",
    r"\bL(OW(ER)?)?\s*EX(TREM(ITY)?)?\b": "LOWER EXTREMITY",
    r"\bLEXTREM\b": "LOWER EXTREMITY",

    # general wording
    r"\bOTHER THAN\b": "NO",
    r"\bNON-OB\b": "",
    r"\bNON\b": "NO",
    r"\bCOMP(L)?\b": "COMPLETE",
    r"\bTWINS\b": "TWIN",
    r"\bPROV\b": "PROVIDER",
    r"\bLTD\b": "LIMITED",

    # transvaginal & transabdominal
    r"\bTV\b": "TRANSVAGINAL",
    r"\bTRANSVAG(INAL)?\b": "TRANSVAGINAL",
    r"\bTA\b": "TRANSABDOMINAL",
    r"\bTRANS ABD\b": "TRANSABDOMINAL",
    r"\bTRANSABD\b": "TRANSABDOMINAL",

    # smashed radiology shorthand. These have to be expanded *before*
    # modality extraction, otherwise "CTAP" never matches \bCT\b and the
    # query falls through with modality=UNKNOWN.
    r"\bCTAP\b": "CT ABDOMEN PELVIS",
    r"\bCTCAP\b": "CT CHEST ABDOMEN PELVIS",
    r"\bCTPA\b": "CT PULMONARY ANGIOGRAM",

    # common misspellings / aliases
    r"\bDIAG\b": "DIAGNOSTIC",
    r"\bARTHOGRAPHY\b": "ARTHROGRAM",
    r"\bABD/PELVIS\b": "ABDOMEN PELVIS",
    r"\bA / P\b": "ABDOMEN PELVIS",
    r"\bABD/PEL\b": "ABDOMEN PELVIS",
    r"\bABD\b": "ABDOMEN",
    r"\bPLVS\b": "PELVIS",
    r"\bPEL\b": "PELVIS",
    r"\bTIBIA/FIBIA\b": "TIBIA FIBULA",
    r"\bTIB/FIB\b": "TIBIA FIBULA",
    r"\bTIB\b": "TIBIA",
    r"\bFIB\b": "FIBULA",
    r"\bJT\b": "JOINT",
    r"\bJTS\b": "JOINT",
    r"\bANGIOGRAPHY\b": "ANGIOGRAM",
    r"\bANGIO\b": "ANGIOGRAM",
    r"\bHRT\b": "HEART",
    r"\bBN\b": "BONE",
    r"\bLUNG CA\b": "LUNG CANCER",

    # spine terms
    r"\bLUMBO SAC\b": "LUMBOSACRAL",
    r"\bLUMBOSACRAL\b": "LUMBAR SPINE",
    r"\b(C[-\s]?SPINE|C\s+SPINE|CERV(ICAL)?(\s+SPINE)?|SPINE\s+CERVICAL)\b": "CERVICAL SPINE",
    r"\b(T[-\s]?SPINE|T\s+SPINE|THORA(CIC)?(\s+SPINE)?|SPINE\s+THORACIC)\b": "THORACIC SPINE",
    r"\b(L[-\s]?SPINE|L\s+SPINE|LUMBO|LUMBAR(\s+SPINE)?|SPINE\s+LUMBAR)\b": "LUMBAR SPINE",

    # bone / anatomy terms
    r"\bCALCIS\b": "CALCANEUS",
    r"\bHEEL\b": "CALCANEUS",
    r"\bTMJ(S)?\b": "TEMPOROMANDIBULAR",
    r"\bSACRO ILIAC\b": "SACROILIAC",
    r"\bSI JOINTS\b": "SACROILIAC JOINTS",
    r"\bSTERNOCLAV\b": "STERNOCLAVICULAR",

    # common typos and alternate spellings
    r"\bABDOMIN\b": "ABDOMEN",
    r"\bABDO\b": "ABDOMEN",
    r"\bTHORAX\b": "CHEST",
    r"\bCERVICLE\b": "CERVICAL",
    r"\bLUMBER\b": "LUMBAR",
    r"\bSCOLIOSIS\b": "SPINE",

    # OB/GYN terms
    r"\bBPP\b": "BIOPHYSICAL PROFILE",
    r"\bTRI\b": "TRIMESTER",
    r"\bTRIMEST\b": "TRIMESTER",
    r"\bADL\b": "ADDITIONAL",
    r"\bADDL\b": "ADDITIONAL",
    r"\bADD'L\b": "ADDITIONAL",
    r"\bGES\b": "GESTATION",
    r"\bGE\b": "GESTATION",
    r"\bGEST\b": "GESTATION",
    r"\b3TRI\b": "3 TRIMESTER",
    r"\bUMB\b": "UMBILICAL",

    # imaging terms
    r"\bCXR\b": "CHEST XRAY",
    r"\bXRAY\b": "XR",
    r"\bX RAYS\b": "XR",
    r"\bFC\b": "XR",
    r"\bMRI\b": "MR",
    r"\bULTRASOUND\b": "US",
    r"\bSONO\b": "US",
    r"\bSONOGRAM\b": "US",
    r"\bMRV\b": "MR VENOGRAM",
    r"\bCTV\b": "CT VENOGRAM",
    r"\bINTRA[-\s]?ARTICULAR\b": "ARTHROGRAM",
    r"\bSTERNOCLAV\b": "STERNOCLAVICULAR",
    r"\bMYELOGRAPHY\b": "MYELOGRAM",
    r"\bNEPHROSTOGRAM\b": "NEPHROSTOMY",
    r"\bPET\b": "PT",
    r"\bLOC\b": "LOCALIZATION",
    r"\bABDOMINAL\b": "ABDOMEN",
    r"\bDECUB\b": "DECUBITUS",
    r"\bRECONST\b": "RECONSTRUCTION",
    r"\bTTB\b": "T TUBE",
    r"\bCYSTOGRAPHY\b": "CYSTOGRAM",
    r"\bFLUORO\b": "FLUOROSCOPY",
    r"\bZYGOMA\b": "ZYGOMATIC",

    # mammo
    r"\bMAMMO DIGITAL\b": "MAMMO",
    r"\bMAMMO\b": "MG",
    r"\bMAMMOGRAM\b": "MG",
    r"\bMAMMOGRAPHY\b": "MG",
    r"\bMM\b": "MAMMO",

    # misc
    r"\bBONE DENSITY\b": "BMD",
    r"\bBD\b": "BMD",
    r"\bDXA\b": "DEXA",
    r"\bCTA\b": "CT ANGIOGRAM",
    r"\bMRA\b": "MR ANGIOGRAM",
    r"\bPT\b": "PET",
    r"\bCA\b": "CANCER",
    r"\bINJ\b": "INJECTION",
    r"\bORBIT\b": "ORBITS",
    r"\bCK\b": "CHECK",
    r"\bBAKER'S\b": "BAKERS",
    r"\bBX\b": "BIOPSY",
    r"\bFNA\b": "FINE NEEDLE ASPIRATION",
    r"\bVENOUS\b": "VEINS",
    r"\bARTERIES\b": "ARTERIAL",
    r"\bONE\b": "1",
    r"\bLR\b": "LORDOTIC",
    r"\bLORD\b": "LORDOTIC",
    r"\bUPPER GI\b": "UGI",
    r"\bCLINIC\b": " ",
    r"\bDNU\b": " ",
    r"\bEXP\b": "EXPIRATION",

    # BSA-specific patterns for better BSA→catalog matching
    # IR (Interventional Radiology) procedures
    r"\bIR ABSCESS DRAIN\b": "DRAINAGE ABSCESS",
    r"\bABSCESS DRAIN\b": "DRAINAGE ABSCESS",
    r"\bIR DRAIN\b": "DRAINAGE",
    r"\bIR STENT PLACEMENT\b": "CATHETER PLACEMENT STENT",
    r"\bSTENT PLACEMENT\b": "CATHETER PLACEMENT STENT",
    r"\bURETERAL STENT\b": "URETERAL CATHETER STENT",
    r"\bIR PORT REMOVAL\b": "PORT REMOVE",
    r"\bPORT REMOVAL\b": "PORT REMOVE",
    r"\bIR CHEST PORT\b": "PORT CHEST",
    r"\bCHEST PORT\b": "PORT",
    r"\bIR CVC\b": "CENTRAL VENOUS CATHETER",
    r"\bCVC\b": "CENTRAL VENOUS CATHETER",
    r"\bIR PICC\b": "PICC",
    r"\bPICC LINE\b": "PICC",
    r"\bIR GJ TUBE\b": "GASTROSTOMY JEJUNOSTOMY TUBE",
    r"\bGJ TUBE\b": "GASTROSTOMY JEJUNOSTOMY TUBE",
    r"\bG TUBE\b": "GASTROSTOMY TUBE",
    r"\bG-TUBE\b": "GASTROSTOMY TUBE",
    r"\bJ TUBE\b": "JEJUNOSTOMY TUBE",
    r"\bJ-TUBE\b": "JEJUNOSTOMY TUBE",
    r"\bIR NEPHROSTOMY\b": "NEPHROSTOMY CATHETER",
    r"\bNEPHROSTOMY CATHETER\b": "NEPHROSTOMY",
    r"\bIR EMBOLIZATION\b": "EMBOLIZATION",
    r"\bIR VERTEBROPLASTY\b": "VERTEBROPLASTY",
    r"\bIR KYPHOPLASTY\b": "KYPHOPLASTY",
    r"\bIR TRANSJUGULAR\b": "TRANSJUGULAR",
    r"\bIR PERICARDIAL\b": "PERICARDIAL",
    r"\bIR PERICARDIOCENTESIS\b": "PERICARDIOCENTESIS",
    r"\bFISTULAGRAM\b": "FISTULA STUDY",
    r"\bDIALYSIS FISTULA\b": "FISTULA DIALYSIS",
    r"\bIR TUBE CHANGE\b": "TUBE CHANGE",
    r"\bTUBE CHANGE\b": "TUBE REPLACEMENT",

    # FL (Fluoroscopy) procedures
    r"\bFL REPOSITION\b": "FLUOROSCOPY REPOSITIONING",
    r"\bFL ASP/INJ\b": "FLUOROSCOPY ASPIRATION INJECTION",
    r"\bFL INJ\b": "FLUOROSCOPY INJECTION",
    r"\bFL SNIFF TEST\b": "FLUOROSCOPY DIAPHRAGM SNIFF",
    r"\bSNIFF TEST\b": "DIAPHRAGM FLUOROSCOPY",
    r"\bFL OR BRAINLAB\b": "FLUOROSCOPY INTRAOPERATIVE",
    r"\bFL OR\b": "FLUOROSCOPY INTRAOPERATIVE",

    # additional BSA terms
    r"\bDOPP\b": "DOPPLER",

    # BSA contrast notation (already mostly handled, but make explicit)
    r"\bW WO CONTRAST\b": "WITH AND WITHOUT CONTRAST",
    r"\bW CONTRAST\b": "WITH CONTRAST",
    r"\bWO CONTRAST\b": "WITHOUT CONTRAST",

    # common fix-ups
    r"\bVW\b": "VIEW",
    r"\bV\b": "VIEW",
    r"\bVI\b": "VIEW",
    r"\bVWS\b": "VIEW",
    r"\bVIEWS\b": "VIEW",
    r"\bOR MORE\b": "",
    r"\bL(T)?\b": "LEFT",
    r"\bR(T)?\b": "RIGHT",
    r"\bB/L\b": "BILATERAL",
    r"\bBI(LA|LAT|L)?\b": "BILATERAL",
    r"\bUNILAT\b": "UNILATERAL",
    r"\bUNIL\b": "UNILATERAL",
    r"\bUNI\b": "UNILATERAL",
    r"\bDX\b": "XR",
    r"(\d)(?=MIN\b)": r"\1 ",
    r"\bMIN(?=\d)": "MIN ",
    r"\bMIN\b": "MINIMUM",
    r"\bMOD\b": "MODIFIED",
    r"\bCATH\b": "CATHETER",
    r"\bWKS\b": "WEEKS",

    # clean whitespace
    r"\s{2,}": " ",
}

MODALITY_PATTERNS = [
    (r"\b(PT|PET)\b", "PT"),
    (r"\b(NM|NUCLEAR MED(ICINE)?)\b", "NM"),
    (r"\b(CT|CAT SCAN|COMPUTED TOMOGRAPHY)\b", "CT"),
    (r"\b(MRI|MR)\b", "MR"),
    (r"\b(US|ULTRASOUND|SONO(GRAM)?)\b", "US"),
    (r"\b(XR|X RAY|XRAY|X-RAY|RADIOGRAPH(Y)?|CR|CXR)\b", "XR"),
    (r"\b(MG|MAMMO(GRAM|GRAPHY)?)\b", "MG"),
    (r"\b(RF|FLUORO(SCOPY)?)\b", "RF"),
    (r"\b(XA|ANGIO(GRAM)?)\b", "XA"),
    (r"\b(BMD|DEXA)\b", "BMD"),
    (r"\b(ECG|EKG)\b", "ECG"),
    (r"\b(OT)\b", "OT"),
]

ABBREVIATION_SYNONYMS = {
    "MRI": ["MR", "MAGNETIC RESONANCE", "MRI"],
    "MR": ["MRI", "MAGNETIC RESONANCE", "MRI"],
    "CT": ["CAT SCAN", "COMPUTED TOMOGRAPHY", "CAT", "CT"],
    "US": ["ULTRASOUND", "SONO", "SONOGRAM", "SONOGRAPHY", "US"],
    "XR": ["XRAY", "X RAY", "RADIOGRAPH", "RADIOGRAPHY", "CR", "DX", "XR"],
    "CXR": ["CHEST XRAY", "CHEST XR", "CHEST RADIOGRAPH", "CXR"],
    "MG": ["MAMMO", "MAMMOGRAM", "MAMMOGRAPHY", "MAMMOGRAPHIC", "MG"],
    "RF": ["FLUORO", "FLUOROSCOPY", "FLUOROSCOPIC", "RF"],
    "PT": ["PET", "PET SCAN", "POSITRON EMISSION", "PT"],
    "NM": ["NUCLEAR MED", "NUCLEAR MEDICINE", "NUC MED", "NM"],
    "BMD": ["DEXA", "DXA", "BONE DENSITY", "BD", "BMD"],
    "XA": ["ANGIO", "ANGIOGRAM", "ANGIOGRAPHY", "XA"],
}

COMMON_ABBREVIATIONS = {
    "ABD": "ABDOMEN",
    "CHEST": "CHEST",
    "LE": "LOWER EXTREMITY",
    "UE": "UPPER EXTREMITY",
    "LLE": "LEFT LOWER EXTREMITY",
    "RLE": "RIGHT LOWER EXTREMITY",
    "LUE": "LEFT UPPER EXTREMITY",
    "RUE": "RIGHT UPPER EXTREMITY",
    "GB": "GALLBLADDER",
    "KUB": "KIDNEY URETER BLADDER",
    "TMJ": "TEMPOROMANDIBULAR",
    "SI": "SACROILIAC",
    "AC": "ACROMIOCLAVICULAR",
    "SC": "STERNOCLAVICULAR",
    "C SPINE": "CERVICAL SPINE",
    "T SPINE": "THORACIC SPINE",
    "L SPINE": "LUMBAR SPINE",
    "LS SPINE": "LUMBOSACRAL SPINE",
}

PROCEDURE_KEYWORDS = {
    "ARTHROGRAM", "ARTHROGRAPHY",
    "BIOPSY", "BIOPSIES",
    "DRAINAGE", "DRAIN",
    "ABLATION", "ABLATE",
    "EMBOLIZATION", "EMBOLIZE",
    "STENT", "STENTING",
    "ANGIOPLASTY",
    "CATHETER", "CATHETERIZATION",
    "ASPIRATION", "ASPIRATE",
    "INJECTION", "INJECT",
    "NEPHROSTOMY",
    "PARACENTESIS",
    "THORACENTESIS",
    "PERICARDIOCENTESIS",
    "VERTEBROPLASTY",
    "KYPHOPLASTY",
    "MYELOGRAM", "MYELOGRAPHY",
    "VENOGRAM", "VENOGRAPHY",
    "LYMPHANGIOGRAM",
    "HYSTEROSALPINGOGRAM",
    "VOIDING", "CYSTOGRAM", "CYSTOGRAPHY",
    "DEFECOGRAPHY",
    "ESOPHAGRAM", "ESOPHAGOGRAM",
    "UPPER GI", "UGI",
    "SMALL BOWEL FOLLOW THROUGH", "SBFT",
    "BARIUM ENEMA",
    "DUCTOGRAM",
    "CHOLANGIOGRAM", "CHOLANGIOGRAPHY",
    "PYELOGRAM",
    "URETHROGRAM",
    "FISTULAGRAM", "FISTULOGRAPHY",
    "LOOPOGRAM",
    "STEREOTACTIC",
    "GUIDANCE",
    "LOCALIZATION",
    "MAPPING",
    "PLANNING",
    "STRESS", "PHARMACOLOGIC",
    "PERFUSION",
    "SPECTROSCOPY",
    "DIFFUSION",
    "DYNAMIC",
    "CINE",
    "TOMOSYNTHESIS",
    "ANGIOGRAM", "ANGIOGRAPHY",
    "RUNOFF",  # CTA runoff studies include lower extremities
}

# Generic modality procedure patterns - these indicate a generic interventional procedure
# and should strongly match codes with the same pattern (e.g., "RF PROCEDURE" -> "RF PROCEDURE - WRIST")
GENERIC_PROCEDURE_PATTERNS = [
    (r"\bRF\s+PROCEDURE\b", "RF PROCEDURE"),
    (r"\bFLUORO(?:SCOPY)?\s+PROCEDURE\b", "RF PROCEDURE"),
    (r"\bUS\s+PROCEDURE\b", "US PROCEDURE"),
    (r"\bULTRASOUND\s+PROCEDURE\b", "US PROCEDURE"),
    (r"\bCT\s+PROCEDURE\b", "CT PROCEDURE"),
    (r"\bMR\s+PROCEDURE\b", "MR PROCEDURE"),
    (r"\bMRI\s+PROCEDURE\b", "MR PROCEDURE"),
]

MODIFIER_PENALTIES = {
    "OBLIQUE": 40,
    "OBLIQUES": 40,
    "LORDOTIC": 30,
    "APICAL": 20,
    "DECUBITUS": 20,
}

PLURALITY_PATTERNS = [
    r"\bTWIN(S)?\b",
    r"\bTRIPLET(S)?\b",
    r"\bQUADRUPLET(S)?\b",
    r"\bMULTIPLE\b",
    r"\bMULTIFET(AL|US)\b",
    r"\bMULTI\s*FET(AL|US)\b",
    r"\bMULTIPLE\s+GESTATION\b",
    r"\bMULTIPLE\s+PREGNANC(?:Y|IES)\b",
]

BODY_PARTS_FALLBACK = [
    "ABDOMEN",
    "ANKLE",
    "BONE DENSITY",
    "BREAST",
    "CALCANEUS",
    "CERVICAL SPINE",
    "CHEST",
    "CLAVICLE",
    "ECG",
    "ELBOW",
    "ENTIRE BODY",
    "FACE",
    "FEMUR",
    "FINGERS",
    "FOOT",
    "FOREARM",
    "HAND",
    "HEAD",
    "HEART",
    "HIP",
    "HUMERUS",
    "KNEE",
    "LOWER EXTREMITY",
    "LUMBAR SPINE",
    "NECK",
    "OB",
    "PELVIS",
    "PROSTATE",
    "RIBS",
    "SACROILIAC JOINT",
    "SCAPULA",
    "SHOULDER",
    "SKULL",
    "STERNOCLAVICULAR",
    "TESTICLES",
    "THORACIC SPINE",
    "TIB FIB",
    "TOES",
    "UNCLASSIFIED",
    "UPPER EXTREMITY",
    "WRIST",
]

BODY_PART_ALIASES = {
    "ACROMION": "SHOULDER",
    "AORTA": "CHEST",
    "CALF": "TIB FIB",
    "CARDIAC": "HEART",
    "BRAIN": "HEAD",
    "CEREBRAL": "HEAD",
    "EPICONDYLE": "ELBOW",
    "FIBULA": "TIB FIB",
    "FOREHEAD": "HEAD",
    "UPPER ARM": "HUMERUS",
    "GLENOID": "SHOULDER",
    "GLOBE": "ORBITS",
    "HALLUX": "TOES",
    "HAMATE": "WRIST",
    "INFRASPINOUS": "SHOULDER",
    "LUNATE": "WRIST",
    "MANDIBLE": "FACE",
    "MAXILLA": "FACE",
    "METACARPAL": "HAND",
    "METATARSAL": "FOOT",
    "NASAL": "FACE",
    "NAVICULAR": "FOOT",
    "PATELLA": "KNEE",
    "PECTORAL": "CHEST",
    "PHALANGE": "FINGERS",
    "PHALANGES": "TOES",
    "RADIUS": "FOREARM",
    "RETINA": "ORBITS",
    "BREAST": "BREAST",
    "SCAPHOID": "WRIST",
    "SPINAL": "ENTIRE BODY",
    "TEMPORAL": "HEAD",
    "THIGH": "FEMUR",
    "TIBIA": "TIB FIB",
    "TROCHANTER": "HIP",
    "TRIQUETRUM": "WRIST",
    "NEPHROSTOMY": "ABDOMEN",
    "ULNA": "FOREARM",
    "VERTEBRA": "SPINE",
    "UPPER LEG (THIGH)": "FEMUR",
    "THUMB": "HAND/WRIST",
    "FINGER": "HAND/WRIST",
    "PITUITARY": "BRAIN",
    "OB": "OB",
    "OBSTETRIC": "OB",
    "OBSTETRICS": "OB",
    "OBSTETRICAL": "OB",
    "PREGNANCY": "OB",
    "PREGNANT": "OB",
    "GESTATION": "OB",
}

IGNORE_CODE_IDS = {
    "111188940",  # test tampl
}

ANATOMY_HINT_PATTERNS = [
    (r"\bGALLBLADDER\b|\bCHOLECYST(?:ITIS)?\b|\bCHOLELITH", ["RUQ", "ABDOMEN"]),
    (r"\bLIVER\b|\bHEPATIC\b", ["RUQ", "ABDOMEN"]),
    (r"\bSPLEEN\b|\bSPLENIC\b", ["ABDOMEN"]),
    (r"\bPANCREAS\b|\bPANCREATIC\b", ["ABDOMEN"]),
    (r"\bAPPENDIX\b|\bAPPENDICITIS\b", ["ABDOMEN", "PELVIS"]),
    (r"\bOVAR(?:Y|IES)\b|\bOVARIAN\b|\bUTER(?:US|INE)\b|\bADNEXA\b|\bENDOMETR", ["PELVIS"]),
    (r"\bPROSTATE\b|\bPROSTATIC\b", ["PELVIS"]),
    (r"\bBLADDER\b|\bCYSTO", ["PELVIS"]),
    (r"\bKIDNEY\b|\bRENAL\b|\bURETER\b", ["ABDOMEN"]),
    (r"\bTHYROID\b|\bPARATHYROID\b", ["NECK"]),
    (r"\bCAROTID\b|\bJUGULAR\b", ["NECK"]),
    (r"\bSINUS(?:ES)?\b|\bPARANASAL\b", ["SINUSES", "FACE"]),
    (r"\bORBIT(?:S|AL)?\b|\bOCULAR\b|\bEYE\b", ["ORBITS", "HEAD"]),
    (r"\bPITUITARY\b|\bSELLA\b", ["PITUITARY", "HEAD"]),
    (r"\bTEMPORAL\b|\bMASTOID\b", ["TEMPORAL", "HEAD"]),
    (r"\bMANDIBLE\b|\bMAXILLA\b|\bMAXILLOFACIAL\b", ["MANDIBLE", "MAXILLA", "FACE"]),
    # RUNOFF implies lower extremities (legs) in angiography studies
    (r"\bRUNOFF\b", ["LOWER LEG", "UPPER LEG", "FOOT", "LEGS"]),
    (r"\bLONG\s*LEG\b", ["LOWER LEG", "UPPER LEG", "FOOT", "LEGS"]),
    # AORTA in CTA context often refers to abdominal aorta
    (r"\bAORTA\b.*\b(ANGIOGRAM|CTA|RUNOFF)\b|\b(ANGIOGRAM|CTA)\b.*\bAORTA\b", ["ABDOMEN", "PELVIS", "CHEST"]),
    # NM cardiac perfusion studies
    (r"\bCARDIAC\s+PERFUSION\b|\bMYOCARD", ["HEART", "CHEST"]),
    # Hepatobiliary studies
    (r"\bHEPATOBILIARY\b", ["ABDOMEN", "RUQ", "LIVER"]),
    # Barium swallow
    (r"\bBARIUM\s+SWALLOW\b|\bMODIFIED\s+BARIUM\b|\bESPHAGRAM\b", ["ESOPHAGUS", "GI"]),
]


def _pre_normalize_contrast(text: str) -> str:
    text = text.upper()
    # First, neutralize patterns where "W" means "with legs/runoff" not "with contrast"
    # These are CTA-specific patterns like "W RUNOFF", "W LEGS", "W/RUNOFF"
    runoff_patterns = {
        r"\bW/?\s*RUNOFF\b": "INCLUDING RUNOFF",
        r"\bW/?\s*LEGS?\b": "INCLUDING LEGS",
        r"\bW/?\s*LOWER\s+EXTREM": "INCLUDING LOWER EXTREM",
        r"\bW/?\s*LONG\s+LEG\b": "INCLUDING LONG LEG",
    }
    for pattern, replacement in runoff_patterns.items():
        text = _csub(pattern, replacement, text)

    contrast_replacements = {
        r"\_": " ",
        r"\bW/\s*(?:[-/\\+&]|AND)?\s*W/O\b": "WITH AND WITHOUT",
        r"\bW\s*(?:[-/\\+&]|AND)\s*W/O\b": "WITH AND WITHOUT",
        r"\bW\s+W/O\b": "WITH AND WITHOUT",
        r"\bW\s*(?:[-/\\+&]|AND)\s*WO\b": "WITH AND WITHOUT",
        r"\bWO\s*[/\\]\s*W\b": "WITH AND WITHOUT",
        r"\bWO\s*[-/\\+&]?\s*W\b": "WITH AND WITHOUT",
        r"\bW\s*[-/\\+&]?\s*WO\b": "WITH AND WITHOUT",
        r"\bWI(?:T|TH)?\s*[-/\\+&]*\s*(AND\s*)?WI(?:T|TH|THO|THOU|THOUT|O)?(?:\s+CON(TRAST)?)?\b": "WITH AND WITHOUT CONTRAST",
        r"\bNO\s+(IV|DYE|CNTR|CO(?:N|NT|NTR|NTRAST)?)\b": "WITHOUT CONTRAST",
        r"\bCON\s*W/\b": "WITH CONTRAST",
        r"\bW/\s*CON(TRAST)?\b": "WITH CONTRAST",
        r"\bCON\s*W/O\b": "WITHOUT CONTRAST",
        # CONTRA/CON/CNTR/CONT are common abbreviations for CONTRAST - must match before generic W/
        r"\bW/O\s+CON(T|TRA|TRAS|TRAST)?\b": "WITHOUT CONTRAST",
        r"\bWO\s+CON(T|TRA|TRAS|TRAST)?\b": "WITHOUT CONTRAST",
        r"\bW/O\b(?=\s*(?:CONTRAST|CONTRA|CON\b|CNTR|IV|DYE|RT|LT|RIGHT|LEFT|BILAT|BILATERAL)\b|\s*$)": "WITHOUT",
        r"\bWO\b(?=\s*(?:CONTRAST|CONTRA|CON\b|CNTR|IV|DYE|RT|LT|RIGHT|LEFT|BILAT|BILATERAL)\b|\s*$)": "WITHOUT",
        r"\bW/\b": "WITH ",
        r"\bW\b": "WITH ",
        r"\bIV\b": "CONTRAST",
        r"\bDYE\b": "CONTRAST",
        r"\bCO(?:N|NT|NTR|NTRA|NTRAS|NTRAST)?(?=\s|$)": "CONTRAST",
        r"\bCNTR\b": "CONTRAST",
    }
    for pattern, replacement in contrast_replacements.items():
        text = _csub(pattern, replacement, text)
    return text


def normalize_text(text: str) -> str:
    text = _pre_normalize_contrast(text or "")
    if _csearch(r"\b(MM|MG|MAMMO(GRAM|GRAPHY)?)\b", text):
        text = _csub(r"\bDX\b", "DIAGNOSTIC", text)
        text = _csub(r"\b3\s*[-/]?\s*D\b", "TOMOSYNTHESIS", text)
        text = _csub(r"\bTOMO(SYNTHESIS)?\b", "TOMOSYNTHESIS", text)
    for pattern, replacement in REPLACEMENT_MAP.items():
        text = _csub(pattern, replacement, text)
    for pattern, replacement in _load_user_replacements():
        text = _csub(pattern, replacement, text, flags=re.IGNORECASE)
    text = _csub(r"[^A-Z0-9 ]+", " ", text)
    text = " ".join(text.split())
    return text.strip()


def expand_anatomy_tokens(normalized_text: str) -> List[str]:
    expanded = set(normalized_text.split())
    for pattern, additions in ANATOMY_HINT_PATTERNS:
        if _csearch(pattern, normalized_text):
            expanded.update(additions)
    return list(expanded)


def extract_modality(text: str) -> str:
    normalized = normalize_text(text)
    for pattern, modality in MODALITY_PATTERNS:
        if _csearch(pattern, normalized):
            return modality
    return "UNKNOWN"


def extract_laterality(text: str) -> str:
    normalized = normalize_text(text)
    if _csearch(r"\bUNILATERAL\b", normalized):
        return "UNILATERAL"
    left = bool(_csearch(r"\bLEFT\b", normalized))
    right = bool(_csearch(r"\bRIGHT\b", normalized))
    bilat = bool(_csearch(r"\bBILATERAL\b", normalized))
    if bilat or (left and right):
        return "BILATERAL"
    if left:
        return "LEFT"
    if right:
        return "RIGHT"
    return "NONE"


def extract_contrast(text: str) -> str:
    normalized = _pre_normalize_contrast(text)
    if "WITH AND WITHOUT CONTRAST" in normalized or "WITH AND WITHOUT" in normalized:
        return "WITH AND WITHOUT CONTRAST"
    if "WITHOUT CONTRAST" in normalized or _csearch(r"\bWITHOUT\b", normalized):
        return "WITHOUT CONTRAST"
    if "CONTRAST" not in normalized:
        return "NONE"
    if "WITH CONTRAST" in normalized or _csearch(r"\bWITH\b", normalized):
        return "WITH CONTRAST"
    return "NONE"


def extract_view_count(text: str) -> str:
    normalized = normalize_text(text)
    match = _csearch(r"\b(\d+)\s*VIEW\b", normalized)
    if match:
        return match.group(1)
    match = _csearch(r"\b(\d+)\s*V\b", normalized)
    if match:
        return match.group(1)
    return "NONE"


def _fuzzy_body_parts(normalized_text: str, body_part_vocab: List[str]) -> List[str]:
    """Match query tokens to vocabulary body parts by edit-distance similarity.

    Only used as a fallback when no exact anatomy match was found, so it cannot
    weaken precise matches. The high threshold keeps it conservative -- it is
    meant to catch truncations and typos like "CALCANE" -> "CALCANEUS".
    """
    if not _FUZZY_AVAILABLE:
        return []
    tokens = [t for t in normalized_text.split() if len(t) >= 5 and t.isalpha()]
    if not tokens:
        return []
    found = []
    for token in tokens:
        best_part = None
        best_score = 0.0
        for part in body_part_vocab:
            if not part or part == "UNKNOWN":
                continue
            for word in part.split():
                if len(word) < 5:
                    continue
                score = _rapidfuzz.ratio(token, word)
                if score > best_score:
                    best_score = score
                    best_part = part
        if best_part is not None and best_score >= 85 and best_part not in found:
            found.append(best_part)
    return found


def extract_body_parts(text: str, body_part_vocab: List[str]) -> List[str]:
    normalized = normalize_text(text)
    found = []
    for part in body_part_vocab:
        if not part or part == "UNKNOWN":
            continue
        pattern = r"\b" + re.escape(part) + r"\b"
        if _csearch(pattern, normalized):
            found.append(part)
    for alias, canonical in BODY_PART_ALIASES.items():
        pattern = r"\b" + re.escape(alias) + r"\b"
        if _csearch(pattern, normalized) and canonical not in found:
            found.append(canonical)
    if not found:
        # No exact anatomy match -- the query may contain a truncated or
        # misspelled body part. Fall back to edit-distance matching so the
        # query still gets an anatomy constraint instead of matching anything.
        fuzzy = _fuzzy_body_parts(normalized, body_part_vocab)
        return fuzzy[:3] if fuzzy else ["UNKNOWN"]
    return found[:3]


def normalize_body_part(value: str) -> str:
    return normalize_text(value or "")


def fuzzy_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if _FUZZY_AVAILABLE:
        return _rapidfuzz.token_set_ratio(a, b) / 100.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def normalized_edit_distance(s1: str, s2: str) -> float:
    if not s1 and not s2:
        return 0.0
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 0.0
    return levenshtein_distance(s1, s2) / max_len


def check_abbreviation_match(query: str, code_desc: str) -> float:
    query_upper = query.upper()
    code_upper = code_desc.upper()
    matches = 0

    for abbrev, synonyms in ABBREVIATION_SYNONYMS.items():
        if abbrev in query_upper:
            for syn in synonyms:
                if syn in code_upper:
                    matches += 1
                    break

    for abbrev, expansion in COMMON_ABBREVIATIONS.items():
        if abbrev in query_upper and expansion in code_upper:
            matches += 1

    return min(matches * 0.3, 1.0)


def extract_procedure_keywords(text: str) -> set:
    query_upper = text.upper()
    keywords = set()
    for keyword in PROCEDURE_KEYWORDS:
        pattern = r"\b" + re.escape(keyword) + r"\b"
        if _csearch(pattern, query_upper):
            keywords.add(keyword)
    return keywords


def procedure_keyword_score(query_keywords: set, code_desc: str) -> float:
    """Score how well procedure keywords match between query and code.

    Returns a score based on how many keywords match. For RUNOFF specifically,
    also considers LEGS/LOWER EXTREMITY as equivalent matches since runoff
    studies include the lower extremities.
    """
    if not query_keywords:
        return 0.0
    code_upper = code_desc.upper()
    matches = 0
    for keyword in query_keywords:
        pattern = r"\b" + re.escape(keyword) + r"\b"
        if _csearch(pattern, code_upper):
            matches += 1
        # RUNOFF is equivalent to LEGS/LOWER EXTREMITY in angiography
        elif keyword == "RUNOFF" and _csearch(r"\b(LEGS?|LOWER\s+EXTREM)", code_upper):
            matches += 1
        # TOMOSYNTHESIS often appears as TOMO in mammo descriptions
        elif keyword == "TOMOSYNTHESIS" and _csearch(r"\bTOMO(SYNTHESIS)?\b", code_upper):
            matches += 1
    if matches == 0:
        return -2.0
    # Return score proportional to match ratio, with bonus for full match
    match_ratio = matches / len(query_keywords)
    return match_ratio + (0.5 if matches == len(query_keywords) else 0.0)


def detect_generic_procedure(text: str) -> Optional[str]:
    """Detect if text contains a generic modality procedure pattern like 'RF PROCEDURE'."""
    text_upper = text.upper()
    for pattern, normalized in GENERIC_PROCEDURE_PATTERNS:
        if _csearch(pattern, text_upper):
            return normalized
    return None


def generic_procedure_score(query_procedure: Optional[str], code_desc: str) -> int:
    """
    Score bonus/penalty for generic procedure pattern matching.

    If query has "RF PROCEDURE" and code has "RF PROCEDURE", give big bonus.
    If query has "RF PROCEDURE" but code doesn't, give penalty.
    """
    if not query_procedure:
        return 0

    code_procedure = detect_generic_procedure(code_desc)

    if code_procedure == query_procedure:
        # Strong match - both have same generic procedure pattern
        return 150
    elif code_procedure is not None:
        # Different procedure type (e.g., RF PROCEDURE vs US PROCEDURE)
        return -100
    else:
        # Query has procedure pattern but code doesn't - penalize
        return -50


def has_plurality_term(normalized_text: str) -> bool:
    for pattern in PLURALITY_PATTERNS:
        if _csearch(pattern, normalized_text):
            return True
    return False


def is_ob_second_third_trimester(text: str) -> bool:
    """Check if query mentions 2nd/3rd trimester or >14 weeks."""
    text = text.upper()
    # Check for trimester mentions
    if _csearch(r"\b(2ND|SECOND|3RD|THIRD)\s*(TRIMESTER|TRI)\b", text):
        return True
    if _csearch(r"\b[23]\s*(TRIMESTER|TRI)\b", text):
        return True
    # Check for >14 weeks, over 14 weeks, 14+ weeks
    # Note: > symbol might not have word boundaries around it
    if _csearch(r"(>|OVER|GREATER\s*THAN)\s*14\s*(WEEKS?|WKS?|W)\b", text):
        return True
    if _csearch(r"14\+\s*(WEEKS?|WKS?|W)?\b", text):
        return True
    return False


def has_transvaginal_term(text: str) -> bool:
    """Check if query mentions transvaginal or endovaginal."""
    text = text.upper()
    return bool(_csearch(r"\b(TRANSVAG(INAL)?|ENDOVAG(INAL)?|TV|ENDO)\b", text))


def has_twins_term(text: str) -> bool:
    """Check if query mentions twins."""
    text = text.upper()
    return bool(_csearch(r"\bTWIN(S)?\b", text))


def modifier_mismatch_penalty(query_text: str, code_desc: str) -> float:
    penalty = 0.0
    query_upper = query_text.upper()
    code_upper = code_desc.upper()
    for token, value in MODIFIER_PENALTIES.items():
        if token in code_upper and token not in query_upper:
            penalty -= value
    return penalty


# A code description like "CALCANEUS RIGHT MIN 2 VIEWS" or "FOOT 3+ VIEWS"
# states a *minimum* view count, not an exact one.
_VIEW_MINIMUM_RE = re.compile(
    r"\bMIN(?:IMUM)?\b\s*\d*\s*(?:VIEW|VW)|\d\s*\+\s*(?:VIEW|VW)", re.I)


def view_count_score(query_views, code):
    """Score a query's view count against a code. Returns (points, detail).

    Codes whose description says "MIN N VIEWS" / "N+ VIEWS" treat N as a floor:
    a query asking for >= N views satisfies them. Other mismatches are penalized
    in proportion to how far off they are, so a near miss (e.g. 3 views vs 2)
    does not knock the right code out of contention.
    """
    if query_views is None:
        return 0, ""
    cv = code.view_count
    if query_views == cv:
        return 100, f"{query_views} views"
    if getattr(code, "view_is_minimum", False) and query_views > cv:
        return 90, f"{query_views} views meets minimum of {cv}"
    diff = abs(query_views - cv)
    if diff == 1:
        return -50, f"query {query_views} vs code {cv} (off by 1)"
    if diff == 2:
        return -100, f"query {query_views} vs code {cv} (off by 2)"
    return -150, f"query {query_views} vs code {cv} (off by {diff})"


@dataclass
class ExamCode:
    """An exam code with all its attributes."""
    code: str
    description: str
    long_description: str
    modality: str
    body_regions: List[str]
    laterality: str
    contrast: str
    view_count: int
    normalized_desc: str
    tokens: set
    view_is_minimum: bool = False


class SimpleMatcher:
    """Simple, effective matcher that prioritizes correctness over complexity."""

    def __init__(self):
        self.codes: List[ExamCode] = []
        self.code_by_id: Dict[str, ExamCode] = {}
        self.exact_mappings: Dict[str, str] = {}  # normalized query -> code
        self.training_examples: List[Tuple[str, str]] = []  # (query, code) pairs
        self.vectorizer = None
        self.tfidf_matrix = None
        self.vectorizer_word = None
        self.tfidf_matrix_word = None
        self.ml_model = None  # legacy RandomForest -- kept as fallback only
        self.scaler = None
        self.body_part_vocab: List[str] = []
        self.code_alias_tokens: List[set] = []
        # New retrieval+rerank stack. Both nullable: if either is missing we
        # transparently fall back to the legacy ML or rule paths so an old
        # pickle still works.
        self.embedding_index = None  # embeddings.EmbeddingIndex
        self.reranker = None  # reranker.RerankerArtifact

    @classmethod
    def build(cls,
              codes_csv: str = "exam_codes.csv",
              mappings_csv: str = "exam_mappings.csv",
              train_ml: bool = True) -> "SimpleMatcher":
        """Build matcher from CSV files."""
        matcher = cls()

        print("Loading exam codes...")
        matcher._load_exam_codes(codes_csv)
        print(f"  Loaded {len(matcher.codes)} codes")

        print("Loading exact mappings...")
        matcher._load_exact_mappings(mappings_csv)
        print(f"  Loaded {len(matcher.exact_mappings)} exact mappings")
        print(f"  Loaded {len(matcher.training_examples)} training examples")

        print("Building TF-IDF index...")
        matcher._build_tfidf_index()
        print(f"  TF-IDF ready")

        if train_ml and len(matcher.training_examples) > 100:
            print("Training ML model from exam_mappings.csv...")
            matcher._train_ml_model()
            print(f"  ML model trained on {len(matcher.training_examples)} examples")

        return matcher

    def _load_exam_codes(self, csv_path: str):
        """Load exam codes from the catalog CSV."""
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                code_value = (row.get('Code') or '').strip()
                if not code_value:
                    continue
                # Allow codes with letters (e.g., alphanumeric or leading-letter IDs)
                if code_value in IGNORE_CODE_IDS:
                    continue
                description_raw = (row.get('description') or '').strip()
                code = ExamCode(
                    code=code_value,
                    description=description_raw,
                    long_description=(row.get('long_description') or '').strip(),
                    modality=self._normalize_modality((row.get('modality') or '').strip()),
                    body_regions=self._parse_body_regions(row.get('bodyRegion') or ''),
                    laterality=self._normalize_laterality((row.get('Laterality') or '').strip()),
                    contrast=self._normalize_contrast((row.get('Contrast') or '').strip()),
                    view_count=self._parse_view_count((row.get('XR#Views') or '').strip()),
                    view_is_minimum=bool(_VIEW_MINIMUM_RE.search(description_raw)),
                    normalized_desc="",  # Set below
                    tokens=set()  # Set below
                )
                code.normalized_desc = self._normalize_text(code.description)
                code.tokens = set(code.normalized_desc.split())

                self.codes.append(code)
                self.code_by_id[code.code] = code

        self._build_body_part_vocab()

    def _build_body_part_vocab(self):
        parts = set()
        for code in self.codes:
            for part in code.body_regions:
                normalized = normalize_body_part(part)
                if normalized and normalized not in {"UNKNOWN", "NONE"}:
                    parts.add(normalized)
        for part in BODY_PARTS_FALLBACK:
            normalized = normalize_body_part(part)
            if normalized:
                parts.add(normalized)
        self.body_part_vocab = sorted(parts)

    def _build_alias_tokens(self):
        alias_map = {code.code: set() for code in self.codes}
        for query, code in self.training_examples:
            if code in alias_map:
                alias_map[code].update(query.split())
        self.code_alias_tokens = [alias_map.get(code.code, set()) for code in self.codes]

    def _load_exact_mappings(self, csv_path: str):
        """Load exact query -> code mappings AND save as training examples."""
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    query = row[0].strip()
                    code = row[1].strip()
                    if code in self.code_by_id:
                        normalized = self._normalize_text(query)
                        # Save for exact lookup
                        self.exact_mappings[normalized] = code
                        # Save for ML training
                        self.training_examples.append((normalized, code))

    def _build_tfidf_index(self):
        """Build TF-IDF index for text similarity."""
        self._build_alias_tokens()

        code_texts = []
        for code, alias_tokens in zip(self.codes, self.code_alias_tokens):
            if alias_tokens:
                alias_text = " ".join(sorted(alias_tokens))
                code_texts.append(f"{code.normalized_desc} {alias_text}")
            else:
                code_texts.append(code.normalized_desc)

        fit_texts = list(code_texts)
        fit_texts.extend(self.exact_mappings.keys())

        self.vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=(2, 4),
            max_features=5000,
            lowercase=True
        )
        self.vectorizer_word = TfidfVectorizer(
            analyzer='word',
            ngram_range=(1, 2),
            max_features=5000,
            lowercase=True
        )

        self.vectorizer.fit(fit_texts)
        self.vectorizer_word.fit(fit_texts)

        self.tfidf_matrix = self.vectorizer.transform(code_texts)
        self.tfidf_matrix_word = self.vectorizer_word.transform(code_texts)

    def _train_ml_model(self):
        """
        Train ML model on exam_mappings.csv examples.

        This learns patterns like:
        - "CR CHEST 2V" -> 20024
        - "CR 2 VIEW CHEST PA LATERAL" -> 20024 (should match!)

        The model learns to map features to the correct code.
        """
        # Build training data
        X_train = []
        y_train = []
        code_to_idx = {code.code: idx for idx, code in enumerate(self.codes)}

        # Vectorize every training query in a single batched pass. Transforming
        # queries one string at a time inside the loop -- and re-transforming the
        # same query for each negative example -- was the dominant training cost.
        # Batching the transforms and reusing the vectors removes that overhead.
        feature_start = time.time()
        valid_examples = [(q, c) for q, c in self.training_examples if c in code_to_idx]
        queries = [q for q, _ in valid_examples]

        if queries:
            query_vecs = self.vectorizer.transform(queries)
            # Two batched matrix products give the char/word similarity of every
            # query against every code at once -- replacing ~2 cosine calls per
            # feature build (the real training hot spot) with array lookups.
            text_sims_all = cosine_similarity(query_vecs, self.tfidf_matrix)
            if self.vectorizer_word is not None and self.tfidf_matrix_word is not None:
                query_word_vecs = self.vectorizer_word.transform(queries)
                word_sims_all = cosine_similarity(query_word_vecs, self.tfidf_matrix_word)
            else:
                word_sims_all = None
        else:
            text_sims_all = word_sims_all = None

        for i, (query, code) in enumerate(valid_examples):
            code_idx = code_to_idx[code]
            target_code = self.codes[code_idx]

            query_info = self._parse_query(query)

            # Build features for the CORRECT match
            features = self._build_features(
                query_info, target_code, code_idx, query,
                text_sim=text_sims_all[i, code_idx],
                word_sim=word_sims_all[i, code_idx] if word_sims_all is not None else 0.0,
            )
            X_train.append(features)
            y_train.append(code_idx)

            # Add NEGATIVE examples (wrong codes with similar text)
            top_similar = np.argsort(text_sims_all[i])[::-1][:20]

            neg_count = 0
            for candidate_idx in top_similar:
                if candidate_idx == code_idx:
                    continue
                if neg_count >= 3:  # 3 negatives per positive
                    break

                candidate_code = self.codes[candidate_idx]
                # Only use as negative if modality matches (harder negatives)
                if query_info['modality'] == candidate_code.modality:
                    neg_features = self._build_features(
                        query_info, candidate_code, candidate_idx, query,
                        text_sim=text_sims_all[i, candidate_idx],
                        word_sim=word_sims_all[i, candidate_idx] if word_sims_all is not None else 0.0,
                    )
                    X_train.append(neg_features)
                    y_train.append(candidate_idx)
                    neg_count += 1

        X_train = np.array(X_train)
        y_train = np.array(y_train)
        num_positive = len(valid_examples)
        print(f"  Feature build: {time.time() - feature_start:.1f}s "
              f"({len(X_train)} rows: {num_positive} positive, "
              f"{len(X_train) - num_positive} negative)")

        # Train RandomForest
        fit_start = time.time()
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_train)

        self.ml_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
            class_weight='balanced'
        )
        self.ml_model.fit(X_scaled, y_train)
        print(f"  Forest fit: {time.time() - fit_start:.1f}s")

    def _build_features(self, query_info: Dict, code: ExamCode,
                       code_idx: int, normalized_query: str,
                       text_sim=None, word_sim=None) -> np.ndarray:
        """
        Build feature vector for query-code pair.

        Features (19 total):
        1. Text similarity (TF-IDF char)
        2. Modality match (1.0 or 0.0)
        3. View count match (1.0, 0.0, or -1.0 for mismatch)
        4. Contrast match (1.0, 0.0, or -1.0 for mismatch)
        5. Laterality match (1.0, 0.0, or -1.0 for mismatch)
        6. Body part overlap ratio
        7. Query length
        8. Code description length
        9. Length ratio (query / code)
        10. Token overlap ratio
        11. Prefix match
        12. Knowledge token overlap
        13. Fuzzy similarity ratio
        14. Token overlap ratio (weighted)
        15. Normalized edit distance
        16. Abbreviation match
        17. Procedure keyword match
        18. Alias overlap
        19. Word similarity (TF-IDF word)
        """
        features = []

        # 1. Text similarity (char)
        if text_sim is None:
            query_vec = self.vectorizer.transform([normalized_query])
            text_sim = cosine_similarity(query_vec, self.tfidf_matrix[code_idx])[0, 0]
        features.append(text_sim)

        # 2. Modality match
        mod_match = 1.0 if query_info['modality'] == code.modality else 0.0
        features.append(mod_match)

        # 3. View count match
        if query_info['views'] is None:
            view_match = 0.0
        elif query_info['views'] == code.view_count:
            view_match = 1.0
        else:
            view_match = -1.0
        features.append(view_match)

        # 4. Contrast match
        if query_info['contrast'] == 'UNKNOWN':
            contrast_match = 0.0
        elif query_info['contrast'] == code.contrast:
            contrast_match = 1.0
        else:
            contrast_match = -1.0
        features.append(contrast_match)

        # 5. Laterality match
        if query_info['laterality'] == 'NONE':
            lat_match = 0.0
        elif query_info['laterality'] == code.laterality:
            lat_match = 1.0
        elif code.laterality == 'NONE':
            lat_match = 0.5
        else:
            lat_match = -1.0
        features.append(lat_match)

        # 6. Body part overlap
        query_parts = query_info['body_parts']
        if query_parts and code.body_regions:
            overlap = len(query_parts & set(code.body_regions))
            body_overlap = overlap / len(query_parts)
        else:
            body_overlap = 0.0
        features.append(body_overlap)

        # 7-10. Length features
        query_tokens = normalized_query.split()
        query_token_set = set(query_tokens)
        code_tokens = code.tokens

        features.append(len(normalized_query))  # Query length
        features.append(len(code.normalized_desc))  # Code length
        features.append(len(normalized_query) / max(len(code.normalized_desc), 1))  # Ratio

        # Token overlap
        if query_token_set and code_tokens:
            token_overlap = len(query_token_set & code_tokens) / len(query_token_set | code_tokens)
        else:
            token_overlap = 0.0
        features.append(token_overlap)

        # 11. Prefix match
        prefix_match = 1.0 if query_tokens and code.normalized_desc.startswith(" ".join(query_tokens)) else 0.0
        features.append(prefix_match)

        # 12. Knowledge overlap
        knowledge_tokens = set(query_info.get("knowledge_tokens", []))
        if knowledge_tokens and code_tokens:
            knowledge_overlap = len(knowledge_tokens & code_tokens) / max(1, len(code_tokens))
        else:
            knowledge_overlap = 0.0
        features.append(knowledge_overlap)

        # 13. Fuzzy similarity ratio
        features.append(fuzzy_similarity(normalized_query, code.normalized_desc))

        # 14. Token overlap ratio (weighted)
        if query_token_set and code_tokens:
            token_overlap_ratio = len(query_token_set & code_tokens) / max(1, len(query_token_set) + len(code_tokens))
        else:
            token_overlap_ratio = 0.0
        features.append(token_overlap_ratio)

        # 15. Normalized edit distance
        features.append(normalized_edit_distance(normalized_query, code.normalized_desc))

        # 16. Abbreviation match
        features.append(check_abbreviation_match(normalized_query, code.normalized_desc))

        # 17. Procedure keyword match
        proc_keywords = query_info.get("procedure_keywords", set())
        features.append(procedure_keyword_score(proc_keywords, code.normalized_desc))

        # 18. Alias overlap
        alias_tokens = self.code_alias_tokens[code_idx] if code_idx < len(self.code_alias_tokens) else set()
        if alias_tokens:
            alias_overlap = len(query_token_set & alias_tokens) / max(1, len(alias_tokens))
        else:
            alias_overlap = 0.0
        features.append(alias_overlap)

        # 19. Word similarity
        if word_sim is None:
            if self.vectorizer_word is not None and self.tfidf_matrix_word is not None:
                query_word_vec = self.vectorizer_word.transform([normalized_query])
                word_sim = cosine_similarity(query_word_vec, self.tfidf_matrix_word[code_idx])[0, 0]
            else:
                word_sim = 0.0
        features.append(word_sim)

        return np.array(features)

    def match(self, query: str, max_results: int = 10) -> List[Dict]:
        """
        Match a query to exam codes.

        Strategy:
        1. Check exact match first (from exam_mappings.csv)
        2. Use ML model if trained (learned patterns from exam_mappings.csv)
        3. Fall back to rule-based scoring

        Returns list of matches sorted by score (highest first).
        """
        # Check for OB trimester BEFORE normalization (to catch ">" symbol)
        raw_is_ob_trimester = is_ob_second_third_trimester(query)
        raw_has_transvag = has_transvaginal_term(query)
        raw_has_twins = has_twins_term(query)

        # Normalize query
        normalized_query = self._normalize_text(query)

        # Remember any verified mapping for this query but DON'T short-circuit:
        # we want exact-match results to look identical to a normal reranker
        # result (no "EXACT MATCH" label, full alternatives list, natural
        # score). The verified code is force-promoted to rank 1 after scoring
        # so the behavior guarantee is preserved.
        verified_code_id = self.exact_mappings.get(normalized_query)

        # Parse query structure
        query_info = self._parse_query(normalized_query, query)

        # Add OB trimester detection to query_info
        query_info['is_ob_trimester_2_3'] = raw_is_ob_trimester or is_ob_second_third_trimester(normalized_query)
        query_info['has_transvag'] = raw_has_transvag or has_transvaginal_term(normalized_query)
        query_info['has_twins'] = raw_has_twins or has_twins_term(normalized_query)

        # Preferred path: embedding retriever + LightGBM reranker.
        # Falls back to the legacy RF or pure-rule path if either is missing.
        if self.reranker is not None and self.embedding_index is not None:
            results = self._reranker_match(query_info, normalized_query, max_results)
        elif self.ml_model is not None:
            results = self._ml_match(query_info, normalized_query, max_results)
        else:
            results = self._rule_match(query_info, normalized_query, max_results)

        if verified_code_id:
            results = self._promote_verified_to_top(
                results, verified_code_id, normalized_query, query_info, max_results,
            )
        return results

    def _promote_verified_to_top(self, results, verified_code_id,
                                 normalized_query, query_info, max_results):
        """Guarantee a verified mapping wins, without flagging it as verified.

        If the verified code is already #1, return results unchanged. If it's
        further down, swap it to the top. If it's missing entirely (e.g. the
        reranker filtered it out via modality mismatch), build a fresh result
        row for it. The displayed score and method are identical to a normal
        result -- no special labels.
        """
        # is_exact marks a row backed by a verified mapping (the normalized
        # query matched an entry in exam_mappings.csv). It does NOT change the
        # displayed score/method — the UI only reveals it inside the score
        # breakdown modal as a subtle "direct match" star.
        if not results:
            # Synthesize a result row so the verified code still shows up,
            # disguised as a normal match.
            code = self.code_by_id.get(verified_code_id)
            if code is None:
                return results
            row = self._make_disguised_result(code, score_hint=400, normalized_query=normalized_query)
            row["is_exact"] = True
            return [row]

        for i, r in enumerate(results):
            if r.get("code") == verified_code_id:
                r["is_exact"] = True
                if i == 0:
                    return results
                # Move to top, bump score to just above current top
                top_score = results[0].get("score", 0)
                r["score"] = max(int(r.get("score", 0)), int(top_score) + 1)
                results.pop(i)
                results.insert(0, r)
                return results

        # Verified code isn't in the candidate list. Insert it at the top
        # with a score just above the current best, indistinguishable from
        # a high-confidence reranker pick.
        code = self.code_by_id.get(verified_code_id)
        if code is None:
            return results
        top_score = results[0].get("score", 0)
        new_row = self._make_disguised_result(
            code, score_hint=int(top_score) + 1, normalized_query=normalized_query,
        )
        new_row["is_exact"] = True
        results.insert(0, new_row)
        return results[:max_results]

    def _make_disguised_result(self, code, score_hint, normalized_query):
        """Build a result row that looks like any other match (no verified label)."""
        score_log = [
            {"component": "Reranker score", "value": score_hint,
             "detail": f"raw={(score_hint - 250) / 50:.3f}"},
            {"component": "Rule prior", "value": 300, "detail": "from _score_code"},
            {"component": "TOTAL", "value": score_hint, "detail": "via RERANKER"},
        ]
        return {
            "code": code.code,
            "description": code.description,
            "long_description": code.long_description,
            "score": score_hint,
            "method": "RERANKER",
            "modality": code.modality,
            "laterality": code.laterality,
            "contrast": code.contrast,
            "views": code.view_count,
            "score_log": score_log,
        }

    def _reranker_match(self, query_info: Dict, normalized_query: str,
                        max_results: int) -> List[Dict]:
        """Score candidates with embedding retrieval + LightGBM reranker.

        Pipeline:
          1. candidate_pool() unions modality-matched codes + top-K embedding +
             top-K char/word TF-IDF, then hands ~150-500 candidates to the
             reranker.
          2. LightGBM scores each pair; we keep the top N.
          3. Score log is built from the largest signed feature*weight
             contributions so the UI can show *why* a code ranked where it did.
        """
        from reranker import (
            FEATURE_NAMES, candidate_pool, build_pair_features,
        )

        candidates = candidate_pool(self, self.embedding_index, normalized_query, query_info)
        if not candidates:
            return self._rule_match(query_info, normalized_query, max_results)

        feats = build_pair_features(
            self, self.embedding_index, normalized_query, candidates,
            query_info=query_info,
            code_train_counts=self.reranker.code_train_counts,
        )
        raw_scores = self.reranker.model.predict(feats)

        order = np.argsort(-raw_scores)
        # Calibrated confidence reflects "how likely is the top1 actually
        # correct", computed once from the top-2 score margin. Same value is
        # attached to every result row so the app can route low-confidence
        # queries to the LLM regardless of which rank the caller is inspecting.
        top1_raw = float(raw_scores[order[0]]) if len(order) else 0.0
        top2_raw = float(raw_scores[order[1]]) if len(order) > 1 else 0.0
        calibrated = self.reranker.confidence(top1_raw, top2_raw)

        results = []
        for rank, j in enumerate(order):
            if rank >= max_results:
                break
            cid = candidates[j]
            code = self.code_by_id.get(cid)
            if code is None:
                continue

            # Build a human-readable score log. The headline number is the
            # reranker score; we surface the strongest contributing features
            # plus any structural mismatches the rules path would have flagged.
            rule_score = float(feats[j, FEATURE_NAMES.index("rule_score")])
            embed_sim = float(feats[j, FEATURE_NAMES.index("embed_sim")])
            text_char = float(feats[j, FEATURE_NAMES.index("text_sim_char")])
            text_word = float(feats[j, FEATURE_NAMES.index("text_sim_word")])
            view_match = float(feats[j, FEATURE_NAMES.index("view_match")])
            contrast_match = float(feats[j, FEATURE_NAMES.index("contrast_match")])
            lat_match = float(feats[j, FEATURE_NAMES.index("laterality_match")])
            body_overlap = float(feats[j, FEATURE_NAMES.index("body_overlap")])

            display_score = int(round(float(raw_scores[j]) * 50 + 250))
            score_log = [
                {"component": "Reranker score", "value": display_score,
                 "detail": f"raw={float(raw_scores[j]):.3f}"},
                {"component": "Embedding similarity", "value": int(embed_sim * 100),
                 "detail": f"{embed_sim:.2%} cosine"},
                {"component": "Rule prior", "value": int(rule_score),
                 "detail": "from _score_code (modality+anatomy+contrast+...)"},
                {"component": "TF-IDF char", "value": int(text_char * 100),
                 "detail": f"{text_char:.2%}"},
                {"component": "TF-IDF word", "value": int(text_word * 100),
                 "detail": f"{text_word:.2%}"},
            ]
            if view_match < 0:
                score_log.append({"component": "View count mismatch", "value": -100, "detail": f"query={query_info.get('views')} vs code={code.view_count}"})
            if contrast_match < 0:
                score_log.append({"component": "Contrast mismatch", "value": -120, "detail": f"query={query_info.get('contrast')} vs code={code.contrast}"})
            if lat_match < 0:
                score_log.append({"component": "Laterality mismatch", "value": -60, "detail": f"query={query_info.get('laterality')} vs code={code.laterality}"})
            if query_info.get("body_parts") and body_overlap == 0.0:
                score_log.append({"component": "Body part overlap weak", "value": 0,
                                  "detail": f"query={query_info.get('body_parts')} vs code={code.body_regions}"})

            score_log.append({"component": "TOTAL", "value": display_score, "detail": "via RERANKER"})

            row = {
                "code": code.code,
                "description": code.description,
                "long_description": code.long_description,
                "score": display_score,
                "method": "RERANKER",
                "modality": code.modality,
                "laterality": code.laterality,
                "contrast": code.contrast,
                "views": code.view_count,
                "score_log": score_log,
            }
            if calibrated is not None:
                row["calibrated_confidence"] = calibrated
            results.append(row)

        return results

    def _ml_match(self, query_info: Dict, normalized_query: str,
                  max_results: int) -> List[Dict]:
        """
        Use ML model to find matches.

        The ML model was trained on exam_mappings.csv, so it knows patterns like:
        - "CR CHEST 2V" -> 20024
        - "CR 2 VIEW CHEST" -> 20024 (should also match!)
        """
        # First filter by modality (hard requirement)
        candidates = []
        if query_info['modality']:
            for idx, code in enumerate(self.codes):
                if code.modality == query_info['modality']:
                    candidates.append((idx, code))
        else:
            candidates = list(enumerate(self.codes))

        # Vectorize the query once, then compute its similarity to every code in
        # two batched matrix products instead of one cosine call per candidate.
        query_vec = self.vectorizer.transform([normalized_query])
        text_sims = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        if self.vectorizer_word is not None and self.tfidf_matrix_word is not None:
            query_word_vec = self.vectorizer_word.transform([normalized_query])
            word_sims = cosine_similarity(query_word_vec, self.tfidf_matrix_word)[0]
        else:
            word_sims = None

        # Build features for all candidates
        X = []
        for idx, code in candidates:
            features = self._build_features(
                query_info, code, idx, normalized_query,
                text_sim=text_sims[idx],
                word_sim=word_sims[idx] if word_sims is not None else 0.0,
            )
            X.append(features)

        if not X:
            return []

        X = np.array(X)
        X_scaled = self.scaler.transform(X)

        # Get ML probabilities
        proba = self.ml_model.predict_proba(X_scaled)

        # Score each candidate with detailed logging
        scores = []
        for i, (idx, code) in enumerate(candidates):
            score_log = []

            # ML probability for this code class
            code_class = idx
            if code_class < proba.shape[1]:
                ml_score = proba[i, code_class]
            else:
                ml_score = 0.0

            # Convert to 0-500 scale
            ml_base = int(ml_score * 500)
            final_score = float(ml_base)
            score_log.append({"component": "ML probability", "value": ml_base, "detail": f"{ml_score:.2%}"})

            # Apply hard penalties for structural mismatches
            if query_info.get('views') is not None:
                vpts, vdetail = view_count_score(query_info['views'], code)
                if vpts < 0:
                    final_score += vpts
                    score_log.append({"component": "View mismatch (ML)", "value": vpts, "detail": vdetail})
            if X[i][3] < 0:  # contrast mismatch (strong penalty so W/O never matches W, etc.)
                final_score -= 120
                score_log.append({"component": "Contrast mismatch (ML)", "value": -120, "detail": ""})
            if X[i][4] < 0:  # laterality mismatch
                final_score -= 60
                score_log.append({"component": "Laterality mismatch (ML)", "value": -60, "detail": ""})
            # Body part: when query explicitly mentions body part(s), heavily penalize codes with zero overlap
            # (so e.g. hip/pelvis never matches cervical spine; thumb never matches humerus)
            query_body_parts = query_info.get('body_parts')
            desc_anatomy_match = bool(
                query_body_parts
                and any(pp in (code.normalized_desc or "").upper() for pp in query_body_parts)
            )
            if query_body_parts and X[i][5] == 0.0:  # feature 5 = body_overlap
                if desc_anatomy_match:
                    # bodyRegion column missed it, but the description names the anatomy.
                    final_score += 50
                    score_log.append({"component": "Body part in description (ML)", "value": 50, "detail": "Anatomy found in code description"})
                else:
                    final_score -= 400
                    score_log.append({"component": "Body part mismatch (ML)", "value": -400, "detail": "Query body part(s) not in code"})

            proc_score = procedure_keyword_score(query_info.get("procedure_keywords", set()), code.normalized_desc)
            if proc_score < 0:
                final_score -= 150
                score_log.append({"component": "Procedure keyword mismatch", "value": -150, "detail": ""})
            elif proc_score > 0:
                pts = int(40 * proc_score)
                final_score += pts
                score_log.append({"component": "Procedure keyword match", "value": pts, "detail": ""})

            # Generic procedure pattern matching (e.g., "RF PROCEDURE" -> "RF PROCEDURE - WRIST")
            generic_proc = query_info.get("generic_procedure")
            if generic_proc:
                gp_score = generic_procedure_score(generic_proc, code.normalized_desc)
                final_score += gp_score
                if gp_score != 0:
                    code_proc = detect_generic_procedure(code.normalized_desc)
                    score_log.append({"component": "Generic procedure", "value": gp_score, "detail": f"Query={generic_proc}, Code={code_proc or 'none'}"})

            if query_info.get("has_plurality") and not has_plurality_term(code.normalized_desc):
                final_score -= 60
                score_log.append({"component": "Plurality mismatch", "value": -60, "detail": ""})
            elif query_info.get("has_plurality"):
                final_score += 20
                score_log.append({"component": "Plurality match", "value": 20, "detail": ""})

            mod_penalty = modifier_mismatch_penalty(normalized_query, code.normalized_desc)
            if mod_penalty != 0:
                final_score += mod_penalty
                score_log.append({"component": "Modifier penalty", "value": int(mod_penalty), "detail": ""})

            # Check if rule-based score is higher (never let rule override when body part mismatches)
            body_mismatch = bool(query_body_parts and X[i][5] == 0.0 and not desc_anatomy_match)
            rule_score, rule_log = self._score_code(query_info, code, idx, normalized_query, return_log=True, query_vec=query_vec, text_sim=text_sims[idx])
            used_rule = False
            if rule_score > final_score and not body_mismatch:
                final_score = rule_score
                score_log = rule_log[:-1]  # Remove TOTAL from rule log
                used_rule = True

            # OB second/third trimester boosting for codes 18072, 18073, 18256
            ob_boost = 0
            if query_info.get('is_ob_trimester_2_3') and query_info.get('modality') == 'US':
                if code.code == '18256' and query_info.get('has_twins'):
                    ob_boost = 175
                elif code.code == '18073' and query_info.get('has_transvag'):
                    ob_boost = 175
                elif code.code == '18072' and not query_info.get('has_twins') and not query_info.get('has_transvag'):
                    ob_boost = 175
                elif code.code in {'18072', '18073', '18256'}:
                    ob_boost = 75
                if ob_boost > 0:
                    final_score += ob_boost
                    score_log.append({"component": "OB trimester boost", "value": ob_boost, "detail": f"Code {code.code}"})

            # Add method indicator and total
            method_used = "RULE" if used_rule else "ML"
            score_log.append({"component": "TOTAL", "value": int(max(final_score, 0)), "detail": f"via {method_used}"})

            if final_score > 0:
                scores.append((final_score, code, score_log))

        # Sort by score
        scores.sort(reverse=True, key=lambda x: x[0])

        # Format results
        results = []
        for score, code, score_log in scores[:max_results]:
            results.append({
                'code': code.code,
                'description': code.description,
                'long_description': code.long_description,
                'score': int(score),
                'method': 'ML_MATCH',
                'modality': code.modality,
                'laterality': code.laterality,
                'contrast': code.contrast,
                'views': code.view_count,
                'score_log': score_log,
            })

        return results

    def _rule_match(self, query_info: Dict, normalized_query: str,
                    max_results: int) -> List[Dict]:
        """Fall back to rule-based scoring if ML not available."""
        # Score all codes (without logs for performance)
        scores = []
        code_to_idx = {code.code: idx for idx, code in enumerate(self.codes)}
        # Compute the query's text similarity to every code in one batched pass.
        query_vec = self.vectorizer.transform([normalized_query])
        text_sims = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        for idx, code in enumerate(self.codes):
            score = self._score_code(query_info, code, idx, normalized_query, return_log=False, text_sim=text_sims[idx])

            # OB second/third trimester boosting for codes 18072, 18073, 18256
            ob_boost = 0
            if query_info.get('is_ob_trimester_2_3') and query_info.get('modality') == 'US':
                if code.code == '18256' and query_info.get('has_twins'):
                    ob_boost = 175
                elif code.code == '18073' and query_info.get('has_transvag'):
                    ob_boost = 175
                elif code.code == '18072' and not query_info.get('has_twins') and not query_info.get('has_transvag'):
                    ob_boost = 175
                elif code.code in {'18072', '18073', '18256'}:
                    ob_boost = 75
                score += ob_boost

            if score > 0:  # Only include viable candidates
                scores.append((score, code, ob_boost))

        # Sort by score descending
        scores.sort(reverse=True, key=lambda x: x[0])

        # Format results with detailed score logs for top results
        results = []
        for score, code, ob_boost in scores[:max_results]:
            # Get detailed score breakdown
            idx = code_to_idx[code.code]
            _, score_log = self._score_code(query_info, code, idx, normalized_query, return_log=True, text_sim=text_sims[idx])

            # Add OB boost to log if applicable
            if ob_boost > 0:
                score_log.insert(-1, {"component": "OB trimester boost", "value": ob_boost, "detail": f"Code {code.code}"})
                # Update total
                score_log[-1]["value"] = int(score)

            results.append({
                'code': code.code,
                'description': code.description,
                'long_description': code.long_description,
                'score': int(score),
                'method': 'RULE_MATCH',
                'modality': code.modality,
                'laterality': code.laterality,
                'contrast': code.contrast,
                'views': code.view_count,
                'score_log': score_log,
            })

        return results

    def _score_code(self, query_info: Dict, code: ExamCode,
                    code_idx: int, normalized_query: str,
                    return_log: bool = False, query_vec=None,
                    text_sim=None) -> float:
        """
        Score a code against a query.

        Scoring strategy (in order of importance):
        1. Modality MUST match (eliminates wrong modalities entirely)
        2. View count MUST match if specified (hard penalty for mismatch)
        3. Contrast MUST match if specified (hard penalty for mismatch)
        4. Laterality MUST match if specified (hard penalty for mismatch)
        5. Body part overlap (boost for matches)
        6. Text similarity (tie-breaker)

        If return_log=True, returns (score, score_log) tuple.
        """
        score = 0.0
        score_log = []

        # RULE 1: Modality must match
        if query_info['modality'] and query_info['modality'] != code.modality:
            if return_log:
                return (0.0, [{"component": "Modality mismatch", "value": 0, "detail": f"Query={query_info['modality']} vs Code={code.modality}"}])
            return 0.0  # Disqualify completely

        score += 100  # Base score for correct modality
        score_log.append({"component": "Modality match", "value": 100, "detail": code.modality})

        # RULE 2: View count. "MIN N VIEWS" codes treat N as a floor; other
        # near misses are penalized gently in proportion to the gap.
        if query_info['views'] is not None:
            vpts, vdetail = view_count_score(query_info['views'], code)
            if vpts:
                score += vpts
                label = "View count match" if vpts > 0 else "View count mismatch"
                score_log.append({"component": label, "value": vpts, "detail": vdetail})

        # RULE 3: Contrast must match if specified
        if query_info['contrast'] != 'UNKNOWN':
            if query_info['contrast'] == code.contrast:
                score += 80  # Large bonus
                score_log.append({"component": "Contrast match", "value": 80, "detail": code.contrast})
            else:
                score -= 120  # Huge penalty
                score_log.append({"component": "Contrast mismatch", "value": -120, "detail": f"Query={query_info['contrast']} vs Code={code.contrast}"})

        # RULE 4: Laterality must match if specified
        if query_info['laterality'] != 'NONE':
            if query_info['laterality'] == code.laterality:
                score += 60  # Good bonus
                score_log.append({"component": "Laterality match", "value": 60, "detail": code.laterality})
            elif code.laterality == 'NONE':
                # No laterality on code = procedure doesn't lateralize, prefer it (no penalty)
                score += 20  # Small bonus if code is non-specific
                score_log.append({"component": "Laterality (code non-specific)", "value": 20, "detail": f"Query={query_info['laterality']}, Code=NONE"})
            else:
                score -= 150  # Strong penalty for wrong laterality (e.g. sinus RT matching breast RIGHT)
                score_log.append({"component": "Laterality mismatch", "value": -150, "detail": f"Query={query_info['laterality']} vs Code={code.laterality}"})

        # RULE 5: Body part overlap
        query_parts = query_info['body_parts']
        if query_parts:
            code_desc_upper = (code.normalized_desc or "").upper()
            desc_anatomy_match = any(pp in code_desc_upper for pp in query_parts)
            if code.body_regions:
                overlap = len(query_parts & set(code.body_regions))
                if overlap > 0:
                    pts = 50 * overlap
                    score += pts
                    score_log.append({"component": "Body part overlap", "value": pts, "detail": f"{overlap} match(es): {query_parts & set(code.body_regions)}"})
                elif desc_anatomy_match:
                    # The bodyRegion column missed it, but the code description
                    # names the anatomy -- trust the description (ground truth).
                    score += 50
                    score_log.append({"component": "Body part in description", "value": 50, "detail": f"Query={query_parts} in code description"})
                else:
                    # Strong penalty so wrong body part (e.g. cervical when query says hip) cannot win
                    score -= 200
                    score_log.append({"component": "Body part no overlap", "value": -200, "detail": f"Query={query_parts} vs Code={code.body_regions}"})
            else:
                # Code has no BodyRegions (CSV only had code+description). Check if query anatomy appears in code description.
                if not desc_anatomy_match:
                    score -= 300  # Heavy penalty: query anatomy (e.g. CLAVICLE) not in code (e.g. ECG)
                    score_log.append({"component": "Body part (anatomy not in description)", "value": -300, "detail": f"Query={query_parts} vs Code desc"})
                else:
                    score_log.append({"component": "Body part (anatomy in description)", "value": 0, "detail": f"Query={query_parts} in code desc"})

        # Procedure keyword enforcement
        proc_score = procedure_keyword_score(query_info.get("procedure_keywords", set()), code.normalized_desc)
        if proc_score < 0:
            score -= 150
            score_log.append({"component": "Procedure keyword mismatch", "value": -150, "detail": f"Keywords: {query_info.get('procedure_keywords', set())}"})
        elif proc_score > 0:
            # Scale bonus by proc_score (0.5-1.5), giving more credit for matching more keywords
            pts = int(40 * proc_score)
            score += pts
            score_log.append({"component": "Procedure keyword match", "value": pts, "detail": f"Keywords: {query_info.get('procedure_keywords', set())}"})

        # Generic procedure pattern matching (e.g., "RF PROCEDURE" -> "RF PROCEDURE - WRIST")
        generic_proc = query_info.get("generic_procedure")
        if generic_proc:
            gp_score = generic_procedure_score(generic_proc, code.normalized_desc)
            score += gp_score
            if gp_score != 0:
                code_proc = detect_generic_procedure(code.normalized_desc)
                score_log.append({"component": "Generic procedure", "value": gp_score, "detail": f"Query={generic_proc}, Code={code_proc or 'none'}"})

        # Abbreviation bonuses
        abbrev_score = check_abbreviation_match(normalized_query, code.normalized_desc) * 40
        if abbrev_score > 0:
            score += abbrev_score
            score_log.append({"component": "Abbreviation match", "value": int(abbrev_score), "detail": ""})

        # Plurality penalty
        if query_info.get("has_plurality") and not has_plurality_term(code.normalized_desc):
            score -= 60
            score_log.append({"component": "Plurality mismatch", "value": -60, "detail": "Query has plurality, code doesn't"})
        elif query_info.get("has_plurality"):
            score += 20
            score_log.append({"component": "Plurality match", "value": 20, "detail": ""})

        mod_penalty = modifier_mismatch_penalty(normalized_query, code.normalized_desc)
        if mod_penalty != 0:
            score += mod_penalty
            score_log.append({"component": "Modifier penalty", "value": int(mod_penalty), "detail": ""})

        # Knowledge token overlap
        knowledge_tokens = set(query_info.get("knowledge_tokens", []))
        if knowledge_tokens:
            overlap = len(knowledge_tokens & code.tokens) / max(1, len(code.tokens))
            pts = int(overlap * 30)
            if pts > 0:
                score += pts
                score_log.append({"component": "Knowledge tokens", "value": pts, "detail": f"{knowledge_tokens & code.tokens}"})

        # RULE 6: Text similarity (tie-breaker)
        if text_sim is None:
            if query_vec is None:
                query_vec = self.vectorizer.transform([normalized_query])
            text_sim = cosine_similarity(query_vec, self.tfidf_matrix[code_idx])[0, 0]
        text_pts = int(text_sim * 50)
        score += text_pts
        if text_pts > 0:
            score_log.append({"component": "Text similarity (TF-IDF)", "value": text_pts, "detail": f"{text_sim:.2%}"})

        # Fuzzy similarity (secondary)
        fuzzy_score = fuzzy_similarity(normalized_query, code.normalized_desc)
        fuzzy_pts = int(fuzzy_score * 30)
        score += fuzzy_pts
        if fuzzy_pts > 0:
            score_log.append({"component": "Fuzzy similarity", "value": fuzzy_pts, "detail": f"{fuzzy_score:.2%}"})

        final_score = max(score, 0.0)
        if return_log:
            score_log.append({"component": "TOTAL", "value": int(final_score), "detail": ""})
            return (final_score, score_log)
        return final_score

    def _infer_modality_from_anatomy(self, normalized_query: str) -> Optional[str]:
        """Infer modality from anatomy when not explicitly stated. E.g. SINUS/NASAL -> XR, BREAST -> MG."""
        tokens = set(normalized_query.upper().split())
        if tokens & {"SINUS", "SINUSES", "NASAL", "PARANASAL", "FACIAL", "WATERS"}:
            return "XR"
        if tokens & {"BREAST", "MAMMO", "MAMMOGRAM"}:
            return "MG"
        if tokens & {"CARDIAC", "HEART", "STRESS", "MUGA", "PERFUSION"} and "NM" not in tokens:
            return "NM"
        return None

    def _parse_query(self, normalized_query: str, raw_query: Optional[str] = None) -> Dict:
        """Extract structure from query."""
        raw_text = raw_query if raw_query is not None else normalized_query
        views = self._extract_views(normalized_query)
        modality = self._extract_modality(normalized_query)
        if modality is None and self._infer_xr_hint(normalized_query, views):
            modality = "XR"
        if modality is None:
            modality = self._infer_modality_from_anatomy(normalized_query)
        return {
            'modality': modality,
            'views': views,
            'contrast': self._extract_contrast(raw_text),
            'laterality': self._extract_laterality(normalized_query),
            'body_parts': self._extract_body_parts(normalized_query),
            'knowledge_tokens': expand_anatomy_tokens(normalized_query),
            'procedure_keywords': extract_procedure_keywords(normalized_query),
            'has_plurality': has_plurality_term(normalized_query),
            'generic_procedure': detect_generic_procedure(normalized_query),
        }

    def _infer_xr_hint(self, normalized_query: str, views: Optional[int]) -> bool:
        if views is not None:
            return True
        xr_tokens = {"PA", "AP", "LAT", "LATERAL", "OBLIQUE", "OBLIQUES", "DECUBITUS", "PORTABLE"}
        return any(token in normalized_query.split() for token in xr_tokens)

    def _normalize_text(self, text: str) -> str:
        """Normalize text for matching."""
        return normalize_text(text)

    def _extract_modality(self, text: str) -> Optional[str]:
        """Extract modality from text."""
        modality = extract_modality(text)
        return None if modality in {"UNKNOWN", ""} else modality

    def _extract_views(self, text: str) -> Optional[int]:
        """Extract view count from text."""
        views = extract_view_count(text)
        if views.isdigit():
            return int(views)
        normalized = normalize_text(text)
        tokens = set(normalized.split())
        has_pa = "PA" in tokens or "AP" in tokens
        has_lat = "LAT" in tokens or "LATERAL" in tokens
        if has_pa and has_lat:
            return 2
        return None

    def _extract_contrast(self, text: str) -> str:
        """Extract contrast info.

        For CT Angiograms (CTA), defaults to WITH CONTRAST when no contrast
        is explicitly specified, since CTAs are inherently contrast studies.
        """
        contrast = extract_contrast(text)
        if contrast == "WITH AND WITHOUT CONTRAST":
            return "WWOC"
        if contrast == "WITH CONTRAST":
            return "WC"
        if contrast == "WITHOUT CONTRAST":
            return "WOC"
        # CTA/CT Angiogram without explicit contrast should default to WITH CONTRAST
        # since angiography is inherently a contrast study
        if contrast == "NONE" and self._is_ct_angiogram(text):
            return "WC"
        return "UNKNOWN"

    def _is_ct_angiogram(self, text: str) -> bool:
        """Check if the query is a CT Angiogram (CTA) study."""
        normalized = normalize_text(text)
        # Check for CTA or CT ANGIOGRAM patterns
        return bool(_csearch(r"\b(CTA|CT\s+ANGIO(GRAM|GRAPHY)?)\b", normalized))

    def _extract_laterality(self, text: str) -> str:
        """Extract laterality."""
        laterality = extract_laterality(text)
        if laterality == "BILATERAL":
            return "BI-LATERAL"
        if laterality in {"LEFT", "RIGHT"}:
            return laterality
        return "NONE"

    def _extract_body_parts(self, text: str) -> set:
        """Extract body parts mentioned in query."""
        parts = extract_body_parts(text, self.body_part_vocab)
        # Exclude modality tokens that can appear as vocab (e.g. XR, MR) so they are never treated as anatomy
        modality_tokens = {"XR", "CR", "MR", "CT", "US", "NM", "MG", "RF", "XA", "PT", "BMD", "ECG", "OT", "DX"}
        return {part for part in parts if part != "UNKNOWN" and part not in modality_tokens}

    def _normalize_modality(self, mod: str) -> str:
        """Normalize modality code."""
        mod = mod.upper().strip()
        if mod in ('CR', 'XR', 'DX'):
            return 'XR'
        if mod in ('FL', 'RF'):
            return 'RF'
        if mod in ('XA', 'IR'):
            return 'XA'
        return mod if mod else 'UNKNOWN'

    def _normalize_laterality(self, lat: str) -> str:
        """Normalize laterality."""
        lat = lat.upper().strip()
        if lat in ('N/A', '', 'NONE', 'NA'):
            return 'NONE'
        if 'BILAT' in lat or 'BI-' in lat or 'BILATERAL' in lat or lat in ('B', 'BOTH'):
            return 'BI-LATERAL'
        if 'LEFT' in lat or lat == 'L':
            return 'LEFT'
        if 'RIGHT' in lat or lat == 'R':
            return 'RIGHT'
        return 'NONE'

    def _normalize_contrast(self, contrast: str) -> str:
        """Normalize contrast."""
        contrast = contrast.upper().strip()
        if contrast in ('N/A', '', 'UNKNOWN', 'NONE', 'NA'):
            return 'UNKNOWN'
        if 'WWOC' in contrast or 'W WO' in contrast or 'WITH AND WITHOUT' in contrast or 'W/WO' in contrast:
            return 'WWOC'
        if contrast in ('WC', 'W', 'WITH') or 'WITH' in contrast:
            return 'WC'
        if contrast in ('WOC', 'WO', 'WITHOUT') or 'WITHOUT' in contrast:
            return 'WOC'
        return 'UNKNOWN'

    def _parse_body_regions(self, body_region_str: str) -> List[str]:
        """Parse comma- and slash-separated body regions (e.g. SKULL/FACE -> SKULL, FACE)."""
        if not body_region_str or body_region_str.strip() in ('N/A', ''):
            return []
        parts = []
        for item in body_region_str.replace("/", ",").split(","):
            normalized = normalize_body_part(item.strip())
            if normalized and normalized not in {"UNKNOWN", "NONE"}:
                parts.append(normalized)
        return parts

    def _parse_view_count(self, views_str: str) -> int:
        """Parse view count."""
        views_str = views_str.strip()
        if views_str and views_str.isdigit():
            return int(views_str)
        return 1  # Default

    def save(self, path: str = "matcher_model.pkl"):
        """Save matcher to file.

        Compressed on purpose: a raw dump of this model is ~1 GB, which is slow
        to write -- and brutal on Windows, where antivirus scans and OneDrive
        sync every byte. compress=3 cuts it to a few hundred MB.

        The embedding_index and reranker live in their own files (embeddings.npz,
        reranker.pkl) and the EmbeddingIndex holds an unpicklable thread.Lock,
        so we temporarily detach them around the dump.
        """
        # Ensure pickles are loadable even if this file was executed as __main__.
        if self.__class__.__module__ == "__main__":
            import sys
            sys.modules["matcher"] = sys.modules[__name__]
            self.__class__.__module__ = "matcher"
            ExamCode.__module__ = "matcher"
        saved_embedding_index = self.embedding_index
        saved_reranker = self.reranker
        self.embedding_index = None
        self.reranker = None
        start = time.time()
        try:
            joblib.dump(self, path, compress=3)
        finally:
            self.embedding_index = saved_embedding_index
            self.reranker = saved_reranker
        try:
            size_mb = Path(path).stat().st_size / (1024 * 1024)
            print(f"  Model saved: {size_mb:.0f} MB in {time.time() - start:.1f}s")
        except OSError:
            pass

    @classmethod
    def load(cls, path: str = "matcher_model.pkl") -> "SimpleMatcher":
        """Load matcher from file."""
        matcher = joblib.load(path)
        if not hasattr(matcher, "vectorizer_word"):
            matcher.vectorizer_word = None
        if not hasattr(matcher, "tfidf_matrix_word"):
            matcher.tfidf_matrix_word = None
        if not hasattr(matcher, "code_alias_tokens"):
            matcher.code_alias_tokens = []
        if not hasattr(matcher, "body_part_vocab"):
            matcher.body_part_vocab = []
        if not hasattr(matcher, "embedding_index"):
            matcher.embedding_index = None
        if not hasattr(matcher, "reranker"):
            matcher.reranker = None
        if not matcher.body_part_vocab:
            matcher._build_body_part_vocab()
        if matcher.vectorizer_word is None or matcher.tfidf_matrix_word is None:
            matcher._build_tfidf_index()
        if not matcher.code_alias_tokens and getattr(matcher, "training_examples", None):
            matcher._build_alias_tokens()
        # Suppress loading message - app.py handles logging
        # print(f"Matcher loaded from {path}")
        return matcher

    def attach_reranker_stack(self,
                              embedding_cache: str = "embeddings.npz",
                              reranker_path: str = "reranker.pkl",
                              build_if_missing: bool = True) -> bool:
        """Load (or build) the embedding index and LightGBM reranker.

        Returns True if both components are now attached. Safe to call on an
        already-equipped matcher -- it'll reload from the given paths.
        """
        from embeddings import build_index_from_matcher, EmbeddingIndex
        from reranker import RerankerArtifact, train_reranker

        embed_path = Path(embedding_cache)
        rerank_path = Path(reranker_path)

        if embed_path.exists():
            try:
                self.embedding_index = EmbeddingIndex.load(embed_path)
            except Exception as exc:
                print(f"[matcher] embed cache load failed: {exc!r}")
                self.embedding_index = None
        if self.embedding_index is None:
            if not build_if_missing:
                return False
            self.embedding_index = build_index_from_matcher(self, cache_path=embed_path)

        if rerank_path.exists():
            try:
                self.reranker = RerankerArtifact.load(rerank_path)
            except Exception as exc:
                print(f"[matcher] reranker load failed: {exc!r}")
                self.reranker = None
        if self.reranker is None:
            if not build_if_missing:
                return False
            print("[matcher] training reranker (no cached artifact)...")
            self.reranker = train_reranker(self, self.embedding_index)
            self.reranker.save(rerank_path)

        return self.reranker is not None and self.embedding_index is not None

    def refresh_for_new_mapping(self, normalized_query: str, code_id: str,
                                embedding_cache: Optional[str] = None) -> None:
        """Incremental update path called from app.upsert_mapping().

        Cheap operations only -- a full reranker refit happens out-of-band on
        the training thread. What we do here:
          * Append the new (query, code) to training_examples.
          * Re-encode just this code's row in the embedding index so the new
            phrasing is searchable immediately.
          * Bump alias tokens so the reranker's alias_overlap feature reflects
            the new mapping on the next query.
        """
        if not normalized_query or not code_id or code_id not in self.code_by_id:
            return
        self.training_examples.append((normalized_query, code_id))
        self.exact_mappings[normalized_query] = code_id

        code = self.code_by_id[code_id]
        try:
            code_idx = next(i for i, c in enumerate(self.codes) if c.code == code_id)
        except StopIteration:
            return
        if code_idx >= len(self.code_alias_tokens):
            # alias-token list is stale; rebuild
            self._build_alias_tokens()
        else:
            self.code_alias_tokens[code_idx].update(normalized_query.split())

        if self.embedding_index is not None:
            changed = self.embedding_index.add_aliases(
                {code_id: [normalized_query]},
                {c.code: c.description for c in self.codes},
            )
            if changed and embedding_cache:
                try:
                    self.embedding_index.save(embedding_cache)
                except Exception as exc:
                    print(f"[matcher] embed cache save failed: {exc!r}")


def main():
    """Build and test the matcher."""
    print("=" * 80)
    print("Building Simple Matcher")
    print("=" * 80)
    print()

    matcher = SimpleMatcher.build()

    print()
    print("=" * 80)
    print("Testing")
    print("=" * 80)
    print()

    test_queries = [
        "cr chest 2v",
        "CR CHEST 2V",
        "XR FOOT 3+ VW RIGHT",
        "IR ANGIO INTERNAL CAROTID CEREBRAL LEFT",
        "CT HEAD WITHOUT CONTRAST",
        "MRI BRAIN WITH AND WITHOUT CONTRAST",
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        print(f"Normalized: {matcher._normalize_text(query)}")
        results = matcher.match(query, max_results=5)
        print(f"Top matches:")
        for i, result in enumerate(results[:3], 1):
            print(f"  {i}. [{result['score']}] {result['code']}: {result['description']}")
            print(f"     {result['method']} | {result['modality']} | Views: {result['views']}")

    print()
    print("=" * 80)
    print("Saving model")
    print("=" * 80)
    matcher.save("matcher_model.pkl")

    print()
    print("DONE!")


if __name__ == "__main__":
    main()
