"""
RadMatcher v3 - UI-compatible Flask app using the v3 matcher.

Self-bootstrapping: running `python app.py` will create a local virtual
environment (.venv), install every dependency, and relaunch itself inside it.
Just hit it and go.
"""

import csv
import hashlib
import io
import json
import os
import re
import secrets
import sqlite3
import subprocess
import sys
import threading
import time
from datetime import datetime

# RadMatcher's pinned numpy/torch stack needs Python 3.11+. On older versions pip
# resolves a different, incompatible combo that yields NaN/Inf in the embedding
# similarity matmul. Fail fast with a clear message instead of a broken model.
if sys.version_info < (3, 11):
    sys.exit(
        "RadMatcher requires Python 3.11 or newer (you are running "
        f"{sys.version_info.major}.{sys.version_info.minor}).\n"
        "Install a newer Python and run, e.g.:  python3.12 app.py"
    )

# ---------------------------------------------------------------------------
# Bootstrap: create .venv and install dependencies before importing 3rd-party
# packages, then relaunch inside the venv. Runs only when executed directly.
# ---------------------------------------------------------------------------

_BOOT_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_VENV_DIR = os.path.join(_BOOT_APP_DIR, ".venv")


def _venv_python():
    if os.name == "nt":
        return os.path.join(_VENV_DIR, "Scripts", "python.exe")
    return os.path.join(_VENV_DIR, "bin", "python")


def _requirements_file():
    win = os.path.join(_BOOT_APP_DIR, "requirements-windows.txt")
    if os.name == "nt" and os.path.exists(win):
        return win
    return os.path.join(_BOOT_APP_DIR, "requirements.txt")


def _install_requirements(venv_python):
    """Install requirements into the venv, skipping if already up to date."""
    req = _requirements_file()
    if not os.path.exists(req):
        return
    with open(req, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    stamp = os.path.join(_VENV_DIR, ".deps-installed")
    if os.path.exists(stamp):
        try:
            if open(stamp, encoding="utf-8").read().strip() == digest:
                return  # dependencies already match this requirements file
        except OSError:
            pass
    print("[setup] Installing dependencies (first run may take several minutes)...")
    try:
        subprocess.check_call([venv_python, "-m", "pip", "install", "-r", req])
    except subprocess.CalledProcessError as exc:
        print(f"[setup] Dependency installation failed: {exc}")
        print(f"[setup] Try manually: {venv_python} -m pip install -r {req}")
        sys.exit(1)
    with open(stamp, "w", encoding="utf-8") as handle:
        handle.write(digest)


def _bootstrap():
    """Ensure a .venv with all dependencies exists, then run inside it."""
    venv_python = _venv_python()

    if not os.path.exists(venv_python):
        print("[setup] Creating virtual environment in .venv ...")
        import venv as _venv

        _venv.EnvBuilder(with_pip=True, upgrade_deps=True).create(_VENV_DIR)

    # A venv's bin/python is a symlink to the base interpreter, so comparing
    # resolved executables gives false positives. sys.prefix is the reliable
    # signal: inside a venv it points at the venv directory, otherwise it does not.
    in_venv = os.path.realpath(sys.prefix) == os.path.realpath(_VENV_DIR)
    _install_requirements(venv_python)

    if not in_venv:
        print("[setup] Launching RadMatcher inside .venv ...")
        os.execv(venv_python, [venv_python, os.path.abspath(__file__), *sys.argv[1:]])


if __name__ == "__main__":
    _bootstrap()


def _ensure_runtime_packages():
    """Safety net for any launcher that bypasses the __main__ bootstrap.

    The .venv bootstrap above only fires when app.py is executed directly.
    If someone runs us under gunicorn / uwsgi / a different venv that was
    set up before requirements.txt was updated, the bootstrap stamp can
    match an older requirements file and lightgbm/sentence-transformers
    end up missing -- which silently downgrades the matcher to rules-only.
    This check forces a pip install of the current requirements.txt the
    moment any required package is absent.
    """
    required_modules = ["lightgbm", "sentence_transformers"]
    missing = []
    for name in required_modules:
        try:
            __import__(name)
        except ImportError:
            missing.append(name)
    if not missing:
        return

    req = _requirements_file()
    print(f"[runtime] Required packages missing: {missing}.")
    print(f"[runtime] Installing from {req} into {sys.executable} ...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req])
    except subprocess.CalledProcessError as exc:
        print(f"[runtime] pip install failed ({exc}).")
        print(f"[runtime] Install manually: {sys.executable} -m pip install -r {req}")
        print(f"[runtime] On macOS, lightgbm also needs libomp: 'brew install libomp'")
        return  # don't crash -- matcher.py will fall back to rules-only

    # Refresh the bootstrap stamp so subsequent starts skip the reinstall.
    try:
        with open(req, "rb") as handle:
            digest = hashlib.sha256(handle.read()).hexdigest()
        stamp = os.path.join(_VENV_DIR, ".deps-installed")
        if os.path.isdir(_VENV_DIR):
            with open(stamp, "w", encoding="utf-8") as handle:
                handle.write(digest)
    except OSError:
        pass


_ensure_runtime_packages()


from flask import Flask, render_template, request, redirect, flash, session, jsonify, send_file, Response

from matcher import SimpleMatcher
from llm_providers import suggest as llm_suggest
from llm_review import add_pending as llm_add_pending, get_pending as llm_get_pending, agree as llm_agree, disagree as llm_disagree, clear_all as llm_clear_all

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = APP_DIR  # All files are now in the same directory

VERSION = "1.0.0"

PROFILE = "synth"

# Training Mode password. Override with the RADMATCHER_PASSWORD environment
# variable; the default below is only meant for local/first-run use.
PASSWORD = os.environ.get("RADMATCHER_PASSWORD", "radmatcher")

MIN_MATCH_SCORE_DEFAULT = 220
MIN_CONFIDENCE_DEFAULT = 0.5  # calibrated P(top1 correct); below this -> LLM fallback
MAX_RESULTS_DEFAULT = 15
MODEL_PATH = os.path.join(APP_DIR, "matcher_model.pkl")
EMBED_CACHE_PATH = os.path.join(APP_DIR, "embeddings.npz")
RERANKER_PATH = os.path.join(APP_DIR, "reranker.pkl")

MAPPINGS_FILE = os.path.join(PROJECT_ROOT, "exam_mappings.csv")
CODES_FILE = os.path.join(PROJECT_ROOT, "exam_codes.csv")
TERM_REPLACEMENTS_FILE = os.path.join(PROJECT_ROOT, "term_replacements.json")
SETTINGS_FILE = os.path.join(PROJECT_ROOT, "app_settings.json")
SCHEDULE_FILE = os.path.join(PROJECT_ROOT, "training_schedule.json")
MATCH_HISTORY_DB = os.path.join(APP_DIR, "match_history.db")
LOGO_PATH = os.path.join(PROJECT_ROOT, "logo.svg")

SETTINGS_DEFAULT = {
    "training": {
        "use_augmentation": True,
        "use_synthetic": True,
        "use_balancing": True,
        "use_cross_validation": False,
        "n_jobs": -1,
    },
    "matching": {
        "max_results": MAX_RESULTS_DEFAULT,
        "min_match_score": MIN_MATCH_SCORE_DEFAULT,
    },
    "wizard": {
        "auto_approve_enabled": True,
        "auto_approve_score": 400,
    },
    # Optional LLM fallback for low-confidence matches. Disabled by default and
    # provider-agnostic: works with hosted APIs (OpenAI, Anthropic/Claude,
    # Google Gemini) and self-hosted OpenAI-compatible servers (Ollama,
    # LM Studio, vLLM, etc.). See llm_providers.py.
    "llm": {
        "enabled": False,
        "provider": "openai",   # openai | anthropic | gemini | openai_compatible
        "model": "gpt-4o",
        "api_key": "",
        "base_url": "",         # required for openai_compatible (e.g. http://localhost:11434/v1)
    },
}

SCHEDULE_DEFAULT = {"enabled": False, "time": "03:00", "last_run": None}

MATCHER_LOCK = threading.Lock()
matcher = None

CODE_LEGEND_LOCK = threading.Lock()
_code_legend_cache = None
_code_legend_mtime = None
_code_map_cache = None
_code_modality_cache = None

# Column order of the exam-code catalog (exam_codes.csv). Used by the Code Set
# Manager for reading, writing, CSV upload validation, and the CSV template.
CODE_FIELDS = ["Code", "Laterality", "Contrast", "XR#Views", "description", "modality", "bodyRegion"]
CODES_WRITE_LOCK = threading.Lock()

TRAINING_LOCK = threading.Lock()
TRAINING_STATUS = {
    "is_training": False,
    "progress": 0,
    "message": "",
    "message_base": None,
    "stage": None,
    "stage_started_at": None,
    "queued": False,
    "started_at": None,
    "completed_at": None,
    "error": None,
    "last_metrics": None,
    "cancelled": False,
}

app = Flask(__name__)
# Set RADMATCHER_SECRET to keep sessions valid across restarts; otherwise a
# random key is generated per process (training-mode logins reset on restart).
app.secret_key = os.environ.get("RADMATCHER_SECRET") or secrets.token_hex(32)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def sanitize_mapping_query(value: str) -> str:
    if not value:
        return ""
    cleaned = re.sub(r"[\r\n\t]+", " ", value)
    cleaned = cleaned.replace(",", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def normalize_mapping_key(value):
    return sanitize_mapping_query(value).upper()


def get_confidence_level(score):
    if score >= 350:
        return "HIGH"
    if score >= 200:
        return "MEDIUM"
    return "LOW"


def normalize_schedule_time(value):
    if not value:
        return None
    try:
        parts = value.strip().split(":")
        if len(parts) != 2:
            return None
        hour = int(parts[0])
        minute = int(parts[1])
        if hour < 0 or hour > 23 or minute < 0 or minute > 59:
            return None
        return f"{hour:02d}:{minute:02d}"
    except (ValueError, TypeError):
        return None


def _merge_settings(defaults, data):
    if not isinstance(data, dict):
        data = {}
    merged = {}
    for key, value in defaults.items():
        if isinstance(value, dict):
            merged[key] = _merge_settings(value, data.get(key, {}))
        else:
            merged[key] = data.get(key, value)
    return merged


def _normalize_int(value, fallback):
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


VALID_LLM_PROVIDERS = ("openai", "anthropic", "gemini", "openai_compatible")


def llm_is_ready(cfg):
    """True if the LLM fallback is enabled and configured enough to call.

    Self-hosted OpenAI-compatible servers (Ollama, LM Studio, ...) usually
    need no API key but do need a base URL; hosted providers need a key.
    """
    if not isinstance(cfg, dict) or not cfg.get("enabled"):
        return False
    provider = (cfg.get("provider") or "openai").lower()
    if provider == "openai_compatible":
        return bool(cfg.get("base_url"))
    return bool(cfg.get("api_key"))


def load_app_settings():
    data = {}
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as handle:
                data = json.load(handle) or {}
        except (OSError, json.JSONDecodeError):
            data = {}
    # Backward compatibility: older settings stored the LLM config under
    # "openai". Migrate it into the provider-agnostic "llm" block on load.
    if isinstance(data, dict) and "llm" not in data and isinstance(data.get("openai"), dict):
        legacy = data.pop("openai")
        data["llm"] = {
            "enabled": bool(legacy.get("enabled", False)),
            "provider": "openai",
            "model": legacy.get("model", "gpt-4o"),
            "api_key": legacy.get("api_key", ""),
            "base_url": "",
        }
    return _merge_settings(SETTINGS_DEFAULT, data)


def save_app_settings(payload):
    settings = load_app_settings()
    training = payload.get("training") if isinstance(payload, dict) else None
    matching = payload.get("matching") if isinstance(payload, dict) else None
    wizard = payload.get("wizard") if isinstance(payload, dict) else None

    if isinstance(training, dict):
        settings["training"]["use_augmentation"] = bool(
            training.get("use_augmentation", settings["training"]["use_augmentation"])
        )
        settings["training"]["use_synthetic"] = bool(
            training.get("use_synthetic", settings["training"]["use_synthetic"])
        )
        settings["training"]["use_balancing"] = bool(
            training.get("use_balancing", settings["training"]["use_balancing"])
        )
        settings["training"]["use_cross_validation"] = bool(
            training.get("use_cross_validation", settings["training"]["use_cross_validation"])
        )
        settings["training"]["n_jobs"] = _normalize_int(
            training.get("n_jobs"), settings["training"]["n_jobs"]
        )

    if isinstance(matching, dict):
        settings["matching"]["max_results"] = _normalize_int(
            matching.get("max_results"), settings["matching"]["max_results"]
        )
        settings["matching"]["min_match_score"] = _normalize_int(
            matching.get("min_match_score"), settings["matching"]["min_match_score"]
        )

    if isinstance(wizard, dict):
        settings["wizard"]["auto_approve_enabled"] = bool(
            wizard.get("auto_approve_enabled", settings["wizard"]["auto_approve_enabled"])
        )
        settings["wizard"]["auto_approve_score"] = _normalize_int(
            wizard.get("auto_approve_score"), settings["wizard"]["auto_approve_score"]
        )

    # Accept the LLM config under "llm" (preferred) or legacy "openai".
    llm_payload = None
    if isinstance(payload, dict):
        if isinstance(payload.get("llm"), dict):
            llm_payload = payload.get("llm")
        elif isinstance(payload.get("openai"), dict):
            llm_payload = payload.get("openai")
    if isinstance(llm_payload, dict):
        settings["llm"]["enabled"] = bool(
            llm_payload.get("enabled", settings["llm"]["enabled"])
        )
        provider = str(llm_payload.get("provider", settings["llm"]["provider"])).strip().lower()
        if provider in VALID_LLM_PROVIDERS:
            settings["llm"]["provider"] = provider
        # Never persist the masked placeholder sent back by the UI.
        if "api_key" in llm_payload and llm_payload["api_key"] not in ("", "***"):
            settings["llm"]["api_key"] = str(llm_payload.get("api_key", ""))
        settings["llm"]["model"] = str(
            llm_payload.get("model", settings["llm"]["model"])
        ).strip() or "gpt-4o"
        settings["llm"]["base_url"] = str(
            llm_payload.get("base_url", settings["llm"]["base_url"])
        ).strip()

    os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True)
    with open(SETTINGS_FILE, "w", encoding="utf-8") as handle:
        json.dump(settings, handle, indent=2)
    return settings


def load_training_schedule():
    if not os.path.exists(SCHEDULE_FILE):
        return dict(SCHEDULE_DEFAULT)
    try:
        with open(SCHEDULE_FILE, "r", encoding="utf-8") as handle:
            data = json.load(handle) or {}
    except (OSError, json.JSONDecodeError):
        return dict(SCHEDULE_DEFAULT)

    schedule = dict(SCHEDULE_DEFAULT)
    schedule["enabled"] = bool(data.get("enabled", schedule["enabled"]))
    time_value = normalize_schedule_time(data.get("time"))
    if time_value:
        schedule["time"] = time_value
    if data.get("last_run"):
        schedule["last_run"] = data.get("last_run")
    return schedule


def save_training_schedule(schedule):
    os.makedirs(os.path.dirname(SCHEDULE_FILE), exist_ok=True)
    with open(SCHEDULE_FILE, "w", encoding="utf-8") as handle:
        json.dump(schedule, handle, indent=2)


def load_term_replacements():
    if not os.path.exists(TERM_REPLACEMENTS_FILE):
        return []
    try:
        with open(TERM_REPLACEMENTS_FILE, "r", encoding="utf-8") as handle:
            data = json.load(handle) or {}
    except (OSError, json.JSONDecodeError):
        return []
    if isinstance(data, dict):
        replacements = data.get("replacements", [])
    else:
        replacements = data
    return replacements if isinstance(replacements, list) else []


def save_term_replacements(replacements):
    os.makedirs(os.path.dirname(TERM_REPLACEMENTS_FILE), exist_ok=True)
    payload = {"replacements": replacements}
    with open(TERM_REPLACEMENTS_FILE, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def apply_term_replacements(text):
    updated = text
    for entry in load_term_replacements():
        if not isinstance(entry, dict):
            continue
        pattern = entry.get("pattern")
        replacement = entry.get("replacement")
        if not pattern or replacement is None:
            continue
        try:
            updated = re.sub(pattern, replacement, updated, flags=re.IGNORECASE)
        except re.error:
            continue
    return updated


def load_matcher():
    if os.path.exists(MODEL_PATH):
        try:
            m = SimpleMatcher.load(MODEL_PATH)
        except Exception:
            m = None
    else:
        m = None
    if m is None:
        # train_ml=False because the new reranker stack replaces the legacy RF
        # (per eval: RF was hurting top-1 by ~6pp and adding 15x latency).
        m = SimpleMatcher.build(codes_csv=CODES_FILE, mappings_csv=MAPPINGS_FILE, train_ml=False)
        try:
            m.save(MODEL_PATH)
        except Exception as exc:
            print(f"[load_matcher] save failed: {exc!r}")
    try:
        ok = m.attach_reranker_stack(
            embedding_cache=EMBED_CACHE_PATH,
            reranker_path=RERANKER_PATH,
            build_if_missing=True,
        )
        if ok:
            print("[load_matcher] reranker stack attached")
        else:
            print("[load_matcher] reranker stack unavailable -- using rules fallback")
    except Exception as exc:
        print(f"[load_matcher] attach_reranker_stack failed: {exc!r}")
    return m


def get_matcher():
    global matcher
    needs_reload = False
    with MATCHER_LOCK:
        if matcher is None:
            matcher = load_matcher()
            needs_reload = True
    if needs_reload:
        reload_exact_mappings()
    return matcher


def reload_exact_mappings(rebuild_index: bool = False):
    current = get_matcher()
    exact_mappings = {}
    training_examples = []

    if os.path.exists(MAPPINGS_FILE):
        with open(MAPPINGS_FILE, "r", newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            for row in reader:
                if len(row) < 2:
                    continue
                query = row[0].strip()
                code = row[1].strip()
                if code not in current.code_by_id:
                    continue
                normalized = current._normalize_text(query)
                exact_mappings[normalized] = code
                training_examples.append((normalized, code))

    with MATCHER_LOCK:
        current.exact_mappings = exact_mappings
        current.training_examples = training_examples
        if rebuild_index:
            current._build_tfidf_index()


# Third value is the expected duration (seconds) for the progress estimator.
# Tuned after the regex-cache optimization cut ML training from ~80s to ~6s.
TRAINING_STAGE_PROGRESS = {
    "starting": (5, 10, 2),
    "loading_codes": (10, 25, 3),
    "loading_mappings": (25, 40, 2),
    "building_tfidf": (40, 60, 3),
    "training_ml": (60, 92, 10),
    "saving_model": (92, 99, 12),
}


def _format_elapsed(seconds: float) -> str:
    total = max(0, int(seconds))
    minutes, sec = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {sec:02d}s"
    return f"{sec}s"


def _set_training_stage(stage: str, progress: int, message: str) -> bool:
    with TRAINING_LOCK:
        if not TRAINING_STATUS["is_training"]:
            return False
        current_progress = TRAINING_STATUS.get("progress", 0)
        TRAINING_STATUS["stage"] = stage
        TRAINING_STATUS["stage_started_at"] = time.time()
        TRAINING_STATUS["progress"] = max(current_progress, progress)
        TRAINING_STATUS["message_base"] = message
        TRAINING_STATUS["message"] = message
    return True


def _compute_stage_progress(stage: str, stage_elapsed: float, current_progress: int) -> int:
    config = TRAINING_STAGE_PROGRESS.get(stage)
    if not config:
        return current_progress
    start, end, expected = config
    if end <= start:
        return current_progress
    max_step = max(0, end - start - 1)
    if expected <= 0:
        return max(current_progress, start)
    if stage_elapsed <= expected:
        progress = start + int(max_step * (stage_elapsed / expected))
    else:
        creep = int((stage_elapsed - expected) / 8)
        progress = start + max_step + creep
    if stage == "training_ml":
        stage_cap = min(93, end + 1)
    else:
        stage_cap = min(99, end + 5)
    progress = min(stage_cap, progress)
    return max(current_progress, progress)


def _queue_training_request() -> bool:
    with TRAINING_LOCK:
        if TRAINING_STATUS.get("queued"):
            return False
        TRAINING_STATUS["queued"] = True
    return True


def _consume_training_queue() -> bool:
    with TRAINING_LOCK:
        if not TRAINING_STATUS.get("queued"):
            return False
        TRAINING_STATUS["queued"] = False
    return True


def _start_queued_training_if_needed() -> None:
    if not _consume_training_queue():
        return
    if start_training():
        return
    with TRAINING_LOCK:
        TRAINING_STATUS["queued"] = True


def start_training():
    with TRAINING_LOCK:
        if TRAINING_STATUS["is_training"]:
            return False
        TRAINING_STATUS.update(
            {
                "is_training": True,
                "progress": 5,
                "message": "Training started",
                "message_base": "Training started",
                "stage": "starting",
                "stage_started_at": time.time(),
                "queued": False,
                "started_at": datetime.now().isoformat(),
                "completed_at": None,
                "error": None,
                "cancelled": False,
            }
        )

    def worker():
        global matcher
        try:
            print("TRAINING STARTED")

            # Update progress periodically in a separate thread
            def progress_updater():
                start_time = time.time()
                while True:
                    with TRAINING_LOCK:
                        if not TRAINING_STATUS["is_training"]:
                            break
                        stage = TRAINING_STATUS.get("stage") or "starting"
                        stage_started_at = TRAINING_STATUS.get("stage_started_at") or start_time
                        message_base = TRAINING_STATUS.get("message_base") or "Training..."
                        current_progress = TRAINING_STATUS.get("progress", 0)

                    now = time.time()
                    stage_elapsed = now - stage_started_at
                    progress = _compute_stage_progress(stage, stage_elapsed, current_progress)
                    message = f"{message_base} (elapsed {_format_elapsed(stage_elapsed)})"

                    with TRAINING_LOCK:
                        if TRAINING_STATUS["is_training"]:
                            if progress > TRAINING_STATUS["progress"]:
                                TRAINING_STATUS["progress"] = progress
                            TRAINING_STATUS["message"] = message

                    time.sleep(1)  # Update every second

            progress_thread = threading.Thread(target=progress_updater, daemon=True)
            progress_thread.start()

            # Actually build the model (suppress output)
            import sys
            import io
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()

            try:
                new_matcher = SimpleMatcher()
                _set_training_stage("loading_codes", 10, "Loading codes")
                new_matcher._load_exam_codes(CODES_FILE)

                _set_training_stage(
                    "loading_mappings",
                    25,
                    f"Loading mappings ({len(new_matcher.codes)} codes)",
                )
                new_matcher._load_exact_mappings(MAPPINGS_FILE)

                _set_training_stage(
                    "building_tfidf",
                    40,
                    f"Building TF-IDF ({len(new_matcher.exact_mappings)} mappings)",
                )
                new_matcher._build_tfidf_index()

                # Legacy RandomForest path is skipped: eval showed it hurt
                # top-1 accuracy by ~6pp vs rules-only. The reranker stack
                # below replaces it.

                _set_training_stage(
                    "embeddings",
                    55,
                    f"Encoding {len(new_matcher.codes)} codes (embeddings)",
                )
                from embeddings import build_index_from_matcher
                new_matcher.embedding_index = build_index_from_matcher(
                    new_matcher, cache_path=EMBED_CACHE_PATH, force=True,
                )

                if len(new_matcher.training_examples) > 100:
                    _set_training_stage(
                        "training_reranker",
                        70,
                        f"Training reranker ({len(new_matcher.training_examples)} examples)",
                    )
                    # Hold out 10% deterministically for calibrator. Reranker
                    # sees the other 90% so calibration is honest.
                    import random as _r
                    rng = _r.Random(42)
                    all_examples = list(new_matcher.training_examples)
                    rng.shuffle(all_examples)
                    n_cal = max(50, int(0.1 * len(all_examples)))
                    cal_examples = all_examples[:n_cal]
                    train_examples = all_examples[n_cal:]
                    saved_examples = new_matcher.training_examples
                    new_matcher.training_examples = train_examples
                    try:
                        from reranker import train_reranker, fit_calibrator
                        new_matcher.reranker = train_reranker(
                            new_matcher, new_matcher.embedding_index,
                        )
                        _set_training_stage(
                            "calibrator", 88,
                            f"Calibrating confidence on {len(cal_examples)} held-out",
                        )
                        fit_calibrator(
                            new_matcher, new_matcher.embedding_index,
                            new_matcher.reranker, cal_examples, kind="margin",
                        )
                    finally:
                        # Restore the full training set so the live matcher
                        # has the most knowledge available for exact match
                        # lookup and alias features.
                        new_matcher.training_examples = saved_examples
                    new_matcher.reranker.save(RERANKER_PATH)
                else:
                    _set_training_stage(
                        "training_reranker",
                        70,
                        f"Skipping reranker ({len(new_matcher.training_examples)} examples < 100)",
                    )

                _set_training_stage("saving_model", 92, "Finalizing and saving")
                new_matcher.save(MODEL_PATH)
            finally:
                sys.stdout = old_stdout

            # Check if cancelled before swapping model
            cancelled = False
            with TRAINING_LOCK:
                if TRAINING_STATUS.get("cancelled"):
                    TRAINING_STATUS.update(
                        {
                            "is_training": False,
                            "progress": 0,
                            "message": "Training cancelled",
                            "message_base": None,
                            "stage": None,
                            "stage_started_at": None,
                            "completed_at": datetime.now().isoformat(),
                            "error": "Cancelled by user",
                        }
                    )
                    cancelled = True
            if cancelled:
                print("TRAINING CANCELLED BY USER\n")
                _start_queued_training_if_needed()
                return

            with MATCHER_LOCK:
                matcher = new_matcher
            with TRAINING_LOCK:
                TRAINING_STATUS.update(
                    {
                        "is_training": False,
                        "progress": 100,
                        "message": "Training completed",
                        "message_base": None,
                        "stage": None,
                        "stage_started_at": None,
                        "completed_at": datetime.now().isoformat(),
                        "error": None,
                        "last_metrics": {
                            "num_codes": len(new_matcher.codes),
                            "num_mappings": len(new_matcher.exact_mappings),
                        },
                    }
                )
            print(f"TRAINING COMPLETED - {len(new_matcher.codes)} codes, {len(new_matcher.exact_mappings)} mappings\n")
            _start_queued_training_if_needed()
        except Exception as exc:
            print(f"TRAINING FAILED: {exc}\n")
            with TRAINING_LOCK:
                TRAINING_STATUS.update(
                    {
                        "is_training": False,
                        "progress": 0,
                        "message": "Training failed",
                        "message_base": None,
                        "stage": None,
                        "stage_started_at": None,
                        "completed_at": datetime.now().isoformat(),
                        "error": str(exc),
                    }
                )
            _start_queued_training_if_needed()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return True


def init_match_history_db():
    conn = sqlite3.connect(MATCH_HISTORY_DB)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS match_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            query TEXT,
            normalized TEXT,
            modality TEXT,
            code TEXT,
            description TEXT,
            score INTEGER,
            method TEXT,
            corrected_code TEXT,
            corrected_at TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def _history_connection():
    conn = sqlite3.connect(MATCH_HISTORY_DB, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def log_match_history(query, result, top_match):
    try:
        init_match_history_db()
        created_at = datetime.now().isoformat()
        normalized = result.get("normalized") if isinstance(result, dict) else ""
        modality = result.get("modality") if isinstance(result, dict) else ""
        code = top_match.get("code", "") if top_match else ""
        description = top_match.get("description", "") if top_match else ""
        score = int(top_match.get("score", 0)) if top_match else 0
        method = top_match.get("method", "") if top_match else ""
        conn = _history_connection()
        try:
            conn.execute(
                """
                INSERT INTO match_history (
                    created_at, query, normalized, modality,
                    code, description, score, method
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (created_at, query, normalized, modality, code, description, score, method),
            )
            conn.execute(
                """
                DELETE FROM match_history
                WHERE id NOT IN (
                    SELECT id FROM match_history
                    ORDER BY id DESC
                    LIMIT 3000
                )
                """
            )
            conn.commit()
        finally:
            conn.close()
    except Exception as exc:
        print(f"[HISTORY] Failed to log match: {exc}")


def scheduled_training_worker():
    while True:
        schedule = load_training_schedule()
        if schedule.get("enabled"):
            time_value = normalize_schedule_time(schedule.get("time"))
            if time_value:
                now = datetime.now()
                hour, minute = map(int, time_value.split(":"))
                if now.hour == hour and now.minute == minute:
                    last_run = schedule.get("last_run", "")
                    today_key = now.date().isoformat()
                    if not last_run.startswith(today_key):
                        if start_training():
                            schedule["last_run"] = now.isoformat()
                            save_training_schedule(schedule)
        time.sleep(30)


_scheduler_thread = threading.Thread(target=scheduled_training_worker, daemon=True)
_scheduler_thread.start()


def csv_file_watcher():
    """Watch exam_mappings.csv for external modifications and auto-retrain."""
    last_mtime = None
    if os.path.exists(MAPPINGS_FILE):
        last_mtime = os.path.getmtime(MAPPINGS_FILE)

    while True:
        time.sleep(5)  # Check every 5 seconds
        try:
            if os.path.exists(MAPPINGS_FILE):
                current_mtime = os.path.getmtime(MAPPINGS_FILE)
                if last_mtime and current_mtime > last_mtime:
                    print(f"[CSV Watcher] Detected change in {MAPPINGS_FILE}")
                    reload_exact_mappings()
                    # Auto-trigger training after external CSV modification
                    if start_training():
                        print("[CSV Watcher] Started background retraining")
                last_mtime = current_mtime
        except Exception as e:
            print(f"[CSV Watcher] Error: {e}")


_csv_watcher_thread = threading.Thread(target=csv_file_watcher, daemon=True)
_csv_watcher_thread.start()


def upsert_mapping(original, selected_code, note):
    original_clean = sanitize_mapping_query(original)
    if not original_clean:
        return False
    original_key = normalize_mapping_key(original_clean)
    rows = []
    updated = False

    if os.path.exists(MAPPINGS_FILE):
        with open(MAPPINGS_FILE, "r", newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            rows = list(reader)

        for row in rows:
            if not row:
                continue
            if normalize_mapping_key(row[0]) == original_key:
                row[0] = original_clean
                if len(row) < 2:
                    row.append(selected_code)
                else:
                    row[1] = selected_code
                if len(row) < 3:
                    row.append(note)
                else:
                    row[2] = note
                updated = True
                break

    if not updated:
        rows.append([original_clean, selected_code, note])

    with open(MAPPINGS_FILE, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)

    reload_exact_mappings()

    # Incremental refresh: re-encode just this code's embedding row with the
    # new alias so the next query benefits *immediately*, without waiting for
    # the full background retrain to finish.
    try:
        current = get_matcher()
        normalized_query = current._normalize_text(original_clean)
        current.refresh_for_new_mapping(
            normalized_query, selected_code, embedding_cache=EMBED_CACHE_PATH,
        )
    except Exception as exc:
        print(f"[upsert_mapping] incremental refresh failed: {exc!r}")

    # Hard-negative signal: if the user's most recent match for this query
    # picked a *different* code, retroactively mark that row as corrected. The
    # next reranker train can then sample those wrong picks as hard negatives.
    try:
        init_match_history_db()
        with _history_connection() as conn:
            row = conn.execute(
                """
                SELECT id, code FROM match_history
                WHERE normalized = ? AND (corrected_code IS NULL OR corrected_code = '')
                ORDER BY created_at DESC LIMIT 1
                """,
                (normalized_query,),
            ).fetchone()
            if row and row["code"] and row["code"] != selected_code:
                conn.execute(
                    "UPDATE match_history SET corrected_code = ?, corrected_at = ? WHERE id = ?",
                    (selected_code, datetime.now().isoformat(), row["id"]),
                )
                conn.commit()
    except Exception as exc:
        print(f"[upsert_mapping] correction backfill failed: {exc!r}")

    # Log the training correction
    action = "UPDATED" if updated else "ADDED"
    print(f"TRAINING {action}: '{original_clean}' -> {selected_code}")

    # Full background retrain still kicks off so the LightGBM weights catch up
    # with the new mapping. The incremental embed update above keeps the live
    # matcher useful while that retrain runs.
    if not start_training():
        _queue_training_request()

    return updated


def get_code_legend():
    global _code_legend_cache, _code_legend_mtime, _code_map_cache, _code_modality_cache
    try:
        mtime = os.path.getmtime(CODES_FILE)
    except OSError:
        return []

    with CODE_LEGEND_LOCK:
        if _code_legend_cache is not None and _code_legend_mtime == mtime:
            return _code_legend_cache

        legend = []
        code_map = {}
        code_modality = {}
        with open(CODES_FILE, "r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                code = row.get("Code", "").strip()
                description = row.get("description", "").strip()
                modality = row.get("modality", "").strip()
                if not code:
                    continue
                legend.append(
                    {
                        "code": code,
                        "description": description,
                        "modality": modality,
                    }
                )
                code_map[code] = description
                code_modality[code] = modality

        _code_legend_cache = legend
        _code_legend_mtime = mtime
        _code_map_cache = code_map
        _code_modality_cache = code_modality
        return legend


def get_code_map():
    if _code_map_cache is None:
        get_code_legend()
    return _code_map_cache or {}


def get_code_modality(code):
    """Return the modality (XR/CT/MR/...) for an exam code, or ''.

    Prefers the trained model so it matches regular match results exactly, and
    falls back to the live CSV legend for codes newer than the model.
    """
    if not code:
        return ""
    code = str(code).strip()
    matcher = get_matcher()
    code_obj = matcher.code_by_id.get(code)
    if code_obj:
        return code_obj.modality
    if _code_modality_cache is None:
        get_code_legend()
    raw = (_code_modality_cache or {}).get(code, "")
    return matcher._normalize_modality(raw) if raw else ""


def _invalidate_code_cache():
    """Force get_code_legend() to re-read exam_codes.csv on the next call."""
    global _code_legend_cache, _code_legend_mtime, _code_map_cache, _code_modality_cache
    with CODE_LEGEND_LOCK:
        _code_legend_cache = None
        _code_legend_mtime = None
        _code_map_cache = None
        _code_modality_cache = None


def load_codes_full():
    """Read exam_codes.csv into a list of dicts with a stable row id each."""
    rows = []
    if not os.path.exists(CODES_FILE):
        return rows
    with open(CODES_FILE, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader):
            entry = {field: (row.get(field) or "").strip() for field in CODE_FIELDS}
            if not entry["Code"]:
                continue
            entry["id"] = idx
            rows.append(entry)
    return rows


def write_codes_full(rows):
    """Rewrite exam_codes.csv from a list of dicts (header always first)."""
    with CODES_WRITE_LOCK:
        with open(CODES_FILE, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CODE_FIELDS)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: (row.get(field) or "") for field in CODE_FIELDS})
    _invalidate_code_cache()


def _clean_code_row(payload):
    """Normalize a code payload from the UI into the catalog field set."""
    row = {field: str(payload.get(field, "") or "").strip() for field in CODE_FIELDS}
    # Sensible defaults so the matcher always has values to work with.
    if not row["Laterality"]:
        row["Laterality"] = "N/A"
    if not row["Contrast"]:
        row["Contrast"] = "N/A"
    if not row["XR#Views"]:
        row["XR#Views"] = "1"
    row["modality"] = row["modality"].upper()
    row["bodyRegion"] = row["bodyRegion"].upper()
    return row


def build_result(query_raw, manual_modality=None, max_results=MAX_RESULTS_DEFAULT):
    raw_query = query_raw.strip()
    if not raw_query:
        return None

    manual_modality = (manual_modality or "").strip().upper()
    query_for_match = apply_term_replacements(raw_query)
    if manual_modality and manual_modality not in query_for_match.upper():
        query_for_match = f"{manual_modality} {query_for_match}"

    current = get_matcher()
    normalized = current._normalize_text(query_for_match)
    query_info = current._parse_query(normalized)
    if manual_modality:
        query_info["modality"] = manual_modality

    matches = current.match(query_for_match, max_results=max_results)
    if manual_modality:
        matches = [match for match in matches if match.get("modality") == manual_modality]

    normalized_matches = []
    for match in matches:
        method = match.get("method", "")
        if method == "EXACT_MATCH":
            method = "VERIFIED"
        normalized_matches.append(
            {
                **match,
                "method": method,
                "view_count": match.get("views") if match.get("views") is not None else "N/A",
                "confidence": get_confidence_level(int(match.get("score", 0))),
                "score_log": match.get("score_log", []),
            }
        )

    result = {
        "original": raw_query,
        "normalized": normalized,
        "modality": query_info.get("modality") or "UNKNOWN",
        "views": query_info.get("views") if query_info.get("views") is not None else "NONE",
        "body_parts": sorted(query_info.get("body_parts") or []),
        "matches": normalized_matches,
    }

    # Log the match
    if normalized_matches:
        top_match = normalized_matches[0]
        print(f"QUERY: '{raw_query}' -> {top_match['code']} ({top_match['method']}, score={top_match['score']})")
    else:
        print(f"QUERY: '{raw_query}' -> No matches found")

    return result


def apply_min_score(result, min_score=None):
    if min_score is None:
        min_score = MIN_MATCH_SCORE_DEFAULT
    if not isinstance(result, dict):
        return result
    matches = result.get("matches") or []
    if not matches:
        result["min_score"] = min_score
        return result
    top = matches[0]
    score = top.get("score", 0) if isinstance(top, dict) else 0
    try:
        score_value = float(score)
    except (TypeError, ValueError):
        score_value = 0
    if score_value < min_score:
        result = dict(result)
        result["matches"] = []
        result["min_score"] = min_score
        result["rejected_score"] = score_value
    return result


def annotate_min_score(result, min_score=None):
    if min_score is None:
        min_score = MIN_MATCH_SCORE_DEFAULT
    if not isinstance(result, dict):
        return result
    matches = result.get("matches") or []
    result["min_score"] = min_score
    if not matches:
        result["top_ok"] = False
        return result
    top = matches[0] if isinstance(matches[0], dict) else {}
    score = top.get("score", 0)
    try:
        score_value = float(score)
    except (TypeError, ValueError):
        score_value = 0
    result["top_ok"] = score_value >= min_score
    if not result["top_ok"]:
        result["rejected_score"] = score_value
    return result


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    llm_fallback_enabled = False
    llm_query = ""
    llm_top_score = 0

    if request.method == "POST":
        csv_file = request.files.get("csv_file")
        if csv_file and csv_file.filename.endswith(".csv"):
            flash("To upload a code set, use the Code Set manager in the sidebar.", "info")
            return redirect("/")

        query_raw = request.form.get("exam_description", "")
        manual_modality = request.form.get("manual_modality", "")
        if query_raw.strip():
            settings = load_app_settings()
            max_results = settings.get("matching", {}).get("max_results", MAX_RESULTS_DEFAULT)
            min_match_score = settings.get("matching", {}).get("min_match_score", MIN_MATCH_SCORE_DEFAULT)
            result = build_result(query_raw, manual_modality=manual_modality, max_results=max_results)
            result = annotate_min_score(result, min_score=min_match_score)
            if not result.get("matches"):
                flash("No match found.", "info")
            elif not result.get("top_ok", True):
                flash(f"Top match below {min_match_score}. Review alternatives.", "info")

            top_match = result.get("matches")[0] if result.get("matches") else None
            log_match_history(query_raw, result, top_match)

            # LLM fallback: prefer calibrated confidence if the reranker
            # supplies it (calibrated_confidence = P(top1 is correct)). Falls
            # back to the legacy score threshold if the matcher path is rules
            # or legacy ML (which don't produce a calibrated probability).
            top_score = int(top_match.get("score", 0)) if top_match else int(result.get("rejected_score", 0) or 0)
            min_match_score = settings.get("matching", {}).get("min_match_score", MIN_MATCH_SCORE_DEFAULT)
            min_conf = settings.get("matching", {}).get("min_confidence", MIN_CONFIDENCE_DEFAULT)
            threshold = min_match_score if min_match_score > 0 else 250
            llm_cfg = settings.get("llm", {})
            confidence = top_match.get("calibrated_confidence") if top_match else None
            low_conf = (confidence is not None and confidence < min_conf)
            # Score floor is a hard safety net — fires even when the calibrator
            # is confident, because a wide margin between two bad candidates
            # (OOD queries) can fool the calibrator.
            low_score = (top_score < threshold)
            if (low_conf or low_score) and llm_is_ready(llm_cfg):
                llm_fallback_enabled = True
                llm_query = query_raw.strip()
                llm_top_score = top_score

    return render_template(
        "index.html",
        profile=PROFILE,
        result=result,
        version=VERSION,
        llm_fallback_enabled=llm_fallback_enabled,
        llm_query=llm_query,
        llm_top_score=llm_top_score,
    )


@app.route("/logo.svg")
def logo_svg():
    if not os.path.exists(LOGO_PATH):
        return "", 404
    return send_file(LOGO_PATH, mimetype="image/svg+xml")


@app.route("/confirm", methods=["POST"])
def confirm():
    original = request.form.get("original", "").strip()
    selected_code = request.form.get("selected_code", "").strip()
    manual_code = request.form.get("manual_code", "").strip()

    if manual_code:
        selected_code = manual_code

    if original and selected_code:
        note = "### added by code matcher manual review"
        if selected_code == "NONE":
            note = "### skipped by user: no suitable match"
        if selected_code != "NONE":
            updated = upsert_mapping(original, selected_code, note)
            action_msg = "Code updated" if updated else "Code trained"
            flash(f"{action_msg}. Background retraining started.", "success")
        else:
            flash("Skipped.", "info")
    else:
        flash("Missing data.", "error")

    return redirect("/")


@app.route("/set_training_mode", methods=["POST"])
def set_training_mode():
    if "enter_training" in request.form:
        if request.form.get("training_password") == PASSWORD:
            session["training_mode"] = True
            flash("Training mode enabled.", "success")
        else:
            flash("Incorrect password.", "error")
    if request.form.get("exit_training"):
        session.pop("training_mode", None)
        flash("Training mode disabled.", "info")
    return redirect("/")


@app.route("/api/match", methods=["POST"])
def api_match():
    data = request.get_json(silent=True) or {}
    query = (data.get("query") or data.get("exam_description") or "").strip()
    manual_modality = (data.get("modality") or "").strip().upper()
    if not query:
        return jsonify({"error": "Query required"}), 400

    settings = load_app_settings()
    max_results = settings.get("matching", {}).get("max_results", MAX_RESULTS_DEFAULT)
    min_score = data.get("min_score")
    if min_score is None:
        min_score = settings.get("matching", {}).get("min_match_score", MIN_MATCH_SCORE_DEFAULT)
    min_score = _normalize_int(min_score, MIN_MATCH_SCORE_DEFAULT)

    result = build_result(query, manual_modality=manual_modality, max_results=max_results)
    result = annotate_min_score(result, min_score=min_score)

    top_match = result.get("matches")[0] if result.get("matches") else None
    top_score = int(top_match.get("score", 0)) if top_match else int(result.get("rejected_score", 0) or 0)
    log_match_history(query, result, top_match)

    # LLM fallback: calibrated confidence is the primary signal; the score
    # threshold is a hard safety net that fires even when the calibrator
    # is confident, because a wide margin between two bad candidates (OOD
    # queries) can fool the calibrator.
    auto_suggest = data.get("auto_suggest_on_low_score", True)
    threshold = min_score if min_score > 0 else 250
    min_conf = settings.get("matching", {}).get("min_confidence", MIN_CONFIDENCE_DEFAULT)
    confidence = top_match.get("calibrated_confidence") if top_match else None
    low_conf = (confidence is not None and confidence < min_conf)
    low_score = (top_score < threshold)
    llm_cfg = settings.get("llm", {})
    if auto_suggest and (low_conf or low_score) and llm_is_ready(llm_cfg):
        try:
            codes = get_code_legend()
            code_list = [{"code": c.get("code", ""), "description": c.get("description", ""), "modality": c.get("modality", "")} for c in codes]
            matcher = get_matcher()
            norm = matcher._normalize_text(query)
            query_info = matcher._parse_query(norm)
            inferred_mod = query_info.get("modality") or manual_modality or None
            ai_result = llm_suggest(
                query,
                code_list,
                llm_cfg,
                inferred_modality=inferred_mod,
            )
            if ai_result and ai_result.get("suggested_code"):
                desc = (ai_result.get("suggested_description") or "").upper()
                if "UNCLASSIFIED" not in desc:
                    item = llm_add_pending(
                        query,
                        ai_result["suggested_code"],
                        ai_result.get("suggested_description"),
                        top_score,
                        confidence=ai_result.get("confidence"),
                    )
                    result["ai_suggestion"] = {
                        "id": item["id"],
                        "suggested_code": ai_result["suggested_code"],
                        "suggested_description": ai_result.get("suggested_description", ""),
                        "suggested_modality": get_code_modality(ai_result["suggested_code"]),
                        "confidence": ai_result.get("confidence", 0.5),
                    }
        except Exception:
            pass  # don't fail the match if AI errors

    # Add top-level code and description for Mirth HL7 compatibility
    if top_match:
        result["code"] = top_match.get("code", "")
        result["description"] = top_match.get("description", "")
        result["score"] = top_match.get("score", 0)
        result["method"] = top_match.get("method", "")

    return jsonify(result)


@app.route("/api/llm-review/add", methods=["POST"])
def api_llm_review_add():
    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    code = (data.get("suggested_code") or data.get("suggestedCode") or "").strip()
    if not query or not code:
        return jsonify({"error": "Query and suggested_code required"}), 400
    item = llm_add_pending(query, code, data.get("suggested_description") or data.get("suggestedDescription"), data.get("match_score") or 0, confidence=data.get("confidence"))
    return jsonify({"id": item["id"], "added_to_review": True})


@app.route("/api/llm-review/suggest-and-queue", methods=["POST"])
def api_llm_review_suggest_and_queue():
    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"error": "Query required"}), 400
    settings = load_app_settings()
    llm_cfg = settings.get("llm", {})
    if not llm_is_ready(llm_cfg):
        return jsonify({"error": "LLM not configured. Choose a provider and add credentials in Settings, then enable suggestions."})
    codes = get_code_legend()
    code_list = [{"code": c.get("code", ""), "description": c.get("description", ""), "modality": c.get("modality", "")} for c in codes]
    matcher = get_matcher()
    norm = matcher._normalize_text(query)
    query_info = matcher._parse_query(norm)
    inferred_mod = query_info.get("modality") or None
    ai_result = llm_suggest(
        query,
        code_list,
        llm_cfg,
        inferred_modality=inferred_mod,
    )
    if not ai_result or not ai_result.get("suggested_code"):
        reason = (ai_result or {}).get("error") or "OpenAI could not return a suggestion"
        return jsonify({"error": reason})
    item = llm_add_pending(query, ai_result["suggested_code"], ai_result.get("suggested_description"), data.get("match_score") or 0, confidence=ai_result.get("confidence"))
    return jsonify({
        "id": item["id"],
        "suggested_code": ai_result["suggested_code"],
        "suggested_description": ai_result.get("suggested_description", ""),
        "suggested_modality": get_code_modality(ai_result["suggested_code"]),
        "confidence": ai_result.get("confidence", 0.5),
        "added_to_review": True,
    })


@app.route("/api/llm-review/pending", methods=["GET"])
def api_llm_review_pending():
    items = llm_get_pending()
    return jsonify([{"id": x["id"], "query": x["query"], "suggested_code": x["suggested_code"], "suggested_description": x["suggested_description"], "suggested_modality": get_code_modality(x["suggested_code"]), "match_score": x["match_score"], "confidence": x.get("confidence"), "created_at": x["created_at"]} for x in items])


@app.route("/api/llm-review/agree", methods=["POST"])
def api_llm_review_agree():
    data = request.get_json(silent=True) or {}
    ids = data.get("ids") or []
    if not ids:
        return jsonify({"error": "Ids required"}), 400
    added = llm_agree(ids, upsert_mapping)
    return jsonify({"agreed": len(ids), "added_to_mappings": added})


@app.route("/api/llm-review/disagree", methods=["POST"])
def api_llm_review_disagree():
    data = request.get_json(silent=True) or {}
    item_id = (data.get("id") or "").strip()
    correct_code = (data.get("correct_code") or data.get("correctCode") or "").strip()
    if not item_id or not correct_code:
        return jsonify({"error": "id and correct_code required"}), 400
    ok = llm_disagree(item_id, correct_code, lambda q, c, n: upsert_mapping(q, c, n))
    return jsonify({"success": ok})


@app.route("/api/llm-review/clear", methods=["POST"])
def api_llm_review_clear():
    """Clear all pending AI suggestions (e.g. after fixing AI settings)."""
    n = llm_clear_all()
    return jsonify({"cleared": n})


@app.route("/api/match_history", methods=["GET"])
def api_match_history():
    limit = request.args.get("limit", 500)
    offset = request.args.get("offset", 0)
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 500
    try:
        offset = int(offset)
    except (TypeError, ValueError):
        offset = 0

    limit = max(1, min(limit, 2000))
    offset = max(0, offset)

    init_match_history_db()
    conn = _history_connection()
    try:
        rows = conn.execute(
            """
            SELECT id, created_at, query, normalized, modality,
                   code, description, score, method,
                   corrected_code, corrected_at
            FROM match_history
            ORDER BY id DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()
    finally:
        conn.close()

    data = [dict(row) for row in rows]
    return jsonify(data)


@app.route("/api/match_history/export_ml", methods=["GET"])
def export_ml_matches():
    """Export ML matches (non-exact matches) as CSV for review.

    Returns CSV with columns: query, code, description
    Optional query params:
        - limit: max rows to export (default 5000)
    """
    limit = request.args.get("limit", 5000)
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 5000
    limit = max(1, min(limit, 10000))

    init_match_history_db()
    conn = _history_connection()
    try:
        rows = conn.execute(
            """
            SELECT query, code, description
            FROM match_history
            WHERE method != 'EXACT_MATCH' AND method IS NOT NULL AND method != ''
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    finally:
        conn.close()

    # Build CSV content
    output = io.StringIO()
    writer = csv.writer(output)
    for row in rows:
        writer.writerow([row["query"], row["code"], row["description"]])

    csv_content = output.getvalue()
    output.close()

    return Response(
        csv_content,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=ml_matches.csv"}
    )


@app.route("/api/match_history/<int:row_id>", methods=["PUT"])
def update_match_history(row_id):
    payload = request.get_json(silent=True) or {}
    corrected_code = (payload.get("corrected_code") or "").strip()
    if not corrected_code:
        return jsonify({"success": False, "error": "Corrected code required."}), 400

    init_match_history_db()
    conn = _history_connection()
    try:
        row = conn.execute(
            "SELECT id, query FROM match_history WHERE id = ?",
            (row_id,),
        ).fetchone()
        if not row:
            return jsonify({"success": False, "error": "History entry not found."}), 404

        corrected_at = datetime.now().isoformat()
        conn.execute(
            """
            UPDATE match_history
            SET corrected_code = ?, corrected_at = ?
            WHERE id = ?
            """,
            (corrected_code, corrected_at, row_id),
        )
        conn.commit()
    finally:
        conn.close()

    original = row["query"]
    original_clean = sanitize_mapping_query(original)
    if original_clean:
        already_exact = False
        if os.path.exists(MAPPINGS_FILE):
            with open(MAPPINGS_FILE, "r", newline="", encoding="utf-8") as handle:
                reader = csv.reader(handle)
                for map_row in reader:
                    if len(map_row) < 2:
                        continue
                    if normalize_mapping_key(map_row[0]) == normalize_mapping_key(original_clean) and map_row[1].strip() == corrected_code:
                        already_exact = True
                        break
        if not already_exact:
            with open(MAPPINGS_FILE, "a", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow([original_clean, corrected_code, "### history correction"])
            reload_exact_mappings()

    return jsonify({"success": True, "corrected_code": corrected_code, "corrected_at": corrected_at})


@app.route("/api/code_legend")
def api_code_legend():
    return jsonify(get_code_legend())


# ---------------------------------------------------------------------------
# Code Set Manager — define your own catalog without touching the CSV by hand
# ---------------------------------------------------------------------------

@app.route("/api/codes", methods=["GET"])
def api_codes_list():
    """Full structured view of the exam-code catalog."""
    return jsonify({"fields": CODE_FIELDS, "codes": load_codes_full()})


@app.route("/api/codes", methods=["POST"])
def api_codes_add():
    payload = request.get_json(silent=True) or {}
    row = _clean_code_row(payload)
    if not row["Code"] or not row["description"]:
        return jsonify({"success": False, "error": "Code and description are required."}), 400

    rows = load_codes_full()
    if any(r["Code"].upper() == row["Code"].upper() for r in rows):
        return jsonify({"success": False, "error": f"Code '{row['Code']}' already exists."}), 409

    rows.append(row)
    write_codes_full(rows)
    return jsonify({"success": True, "code": row, "total": len(rows)})


@app.route("/api/codes/<int:row_id>", methods=["PUT"])
def api_codes_update(row_id):
    payload = request.get_json(silent=True) or {}
    rows = load_codes_full()
    if row_id < 0 or row_id >= len(rows):
        return jsonify({"success": False, "error": "Code not found."}), 404

    row = _clean_code_row(payload)
    if not row["Code"] or not row["description"]:
        return jsonify({"success": False, "error": "Code and description are required."}), 400
    # Guard against renaming onto another existing code.
    for idx, existing in enumerate(rows):
        if idx != row_id and existing["Code"].upper() == row["Code"].upper():
            return jsonify({"success": False, "error": f"Code '{row['Code']}' already exists."}), 409

    rows[row_id] = row
    write_codes_full(rows)
    return jsonify({"success": True, "code": row})


@app.route("/api/codes/<int:row_id>", methods=["DELETE"])
def api_codes_delete(row_id):
    rows = load_codes_full()
    if row_id < 0 or row_id >= len(rows):
        return jsonify({"success": False, "error": "Code not found."}), 404
    removed = rows.pop(row_id)
    write_codes_full(rows)
    return jsonify({"success": True, "removed": removed["Code"], "total": len(rows)})


@app.route("/api/codes/clear", methods=["POST"])
def api_codes_clear():
    """Delete every code in the catalog (keeps the CSV header).

    Existing mappings in exam_mappings.csv are left untouched.
    """
    removed = len(load_codes_full())
    write_codes_full([])
    return jsonify({"success": True, "removed": removed, "total": 0})


@app.route("/api/codes/template")
def api_codes_template():
    """Download a blank CSV template for the exam-code catalog."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(CODE_FIELDS)
    writer.writerow(["EXAMPLE1", "N/A", "WOC", "1", "CT Head wo IV Contrast", "CT", "HEAD"])
    writer.writerow(["EXAMPLE2", "LEFT", "N/A", "2", "XR Knee Left 2 Views", "XR", "KNEE"])
    data = buffer.getvalue()
    return Response(
        data,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=exam_codes_template.csv"},
    )


@app.route("/api/codes/upload", methods=["POST"])
def api_codes_upload():
    """Replace or append the catalog from an uploaded CSV.

    The CSV must have a header row containing at least 'Code' and
    'description'. Other catalog columns are optional and default sensibly.
    """
    upload = request.files.get("file") or request.files.get("csv_file")
    if not upload or not upload.filename:
        return jsonify({"success": False, "error": "No file uploaded."}), 400
    mode = (request.form.get("mode") or "replace").strip().lower()
    if mode not in ("replace", "append"):
        mode = "replace"

    try:
        text = upload.read().decode("utf-8-sig")
    except UnicodeDecodeError:
        return jsonify({"success": False, "error": "File must be UTF-8 encoded CSV."}), 400

    reader = csv.DictReader(io.StringIO(text))
    headers = [h.strip() for h in (reader.fieldnames or [])]
    if "Code" not in headers or "description" not in headers:
        return jsonify({
            "success": False,
            "error": "CSV must include at least 'Code' and 'description' columns. "
                     "Download the template for the expected format.",
        }), 400

    parsed = []
    seen = set()
    skipped = 0
    for raw in reader:
        row = _clean_code_row({k: raw.get(k, "") for k in CODE_FIELDS})
        if not row["Code"] or not row["description"]:
            skipped += 1
            continue
        key = row["Code"].upper()
        if key in seen:
            skipped += 1
            continue
        seen.add(key)
        parsed.append(row)

    if not parsed:
        return jsonify({"success": False, "error": "No valid rows found in the CSV."}), 400

    if mode == "append":
        existing = load_codes_full()
        existing_codes = {r["Code"].upper() for r in existing}
        added = [r for r in parsed if r["Code"].upper() not in existing_codes]
        skipped += len(parsed) - len(added)
        final_rows = existing + added
        added_count = len(added)
    else:
        final_rows = parsed
        added_count = len(parsed)

    write_codes_full(final_rows)
    return jsonify({
        "success": True,
        "mode": mode,
        "added": added_count,
        "skipped": skipped,
        "total": len(final_rows),
        "retrain_recommended": True,
    })


@app.route("/api/codes/generate_variations", methods=["POST"])
def api_codes_generate_variations():
    """Seed exam_mappings.csv with non-LLM text variations of each code.

    Uses the rule-based generator in seed_mappings.py (modality aliases,
    anatomy synonyms, contrast/laterality/view phrasing). Only genuinely new
    (variant, code) pairs are appended, so it is safe to run repeatedly.
    """
    from seed_mappings import variants_for

    payload = request.get_json(silent=True) or {}
    try:
        max_per_code = int(payload.get("max_per_code", 8))
    except (TypeError, ValueError):
        max_per_code = 8
    max_per_code = max(1, min(max_per_code, 25))

    codes = load_codes_full()
    if not codes:
        return jsonify({"success": False, "error": "No codes in the catalog yet."}), 400

    # Existing (variant, code) pairs so we never write duplicates.
    seen = set()
    if os.path.exists(MAPPINGS_FILE):
        with open(MAPPINGS_FILE, "r", newline="", encoding="utf-8") as handle:
            for row in csv.reader(handle):
                if len(row) >= 2:
                    seen.add((row[0].strip().lower(), row[1].strip()))

    new_rows = []
    for entry in codes:
        code = entry["Code"]
        desc = entry["description"]
        if not code or not desc:
            continue
        candidates = [desc] + sorted(variants_for(desc, max_per_code=max_per_code))
        for variant in candidates:
            variant = variant.strip()
            if not variant:
                continue
            key = (variant.lower(), code)
            if key in seen:
                continue
            seen.add(key)
            new_rows.append((variant, code, "### generated variation"))

    if new_rows:
        with open(MAPPINGS_FILE, "a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerows(new_rows)
        reload_exact_mappings()

    return jsonify({
        "success": True,
        "added": len(new_rows),
        "codes_processed": len(codes),
        "retrain_recommended": len(new_rows) > 0,
    })


@app.route("/api/training_data")
def api_training_data():
    code_map = get_code_map()
    data = []
    if os.path.exists(MAPPINGS_FILE):
        with open(MAPPINGS_FILE, "r", newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            for idx, row in enumerate(reader):
                if len(row) >= 2:
                    code = row[1].strip()
                    data.append(
                        {
                            "id": idx,
                            "original": row[0].strip(),
                            "code": code,
                            "description": code_map.get(code, ""),
                            "note": row[2].strip() if len(row) > 2 else "",
                        }
                    )
    return jsonify(data)


@app.route("/api/training_data/<int:row_id>", methods=["DELETE"])
def delete_training_data(row_id):
    if not os.path.exists(MAPPINGS_FILE):
        return jsonify({"success": False, "error": "Mappings file missing."}), 404

    with open(MAPPINGS_FILE, "r", newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))

    if row_id < 0 or row_id >= len(rows):
        return jsonify({"success": False, "error": "Entry not found."}), 404

    rows.pop(row_id)
    with open(MAPPINGS_FILE, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)

    reload_exact_mappings()
    return jsonify({"success": True})


@app.route("/api/training_data/<int:row_id>", methods=["PUT"])
def update_training_data(row_id):
    payload = request.get_json(silent=True) or {}
    new_code = (payload.get("code") or "").strip()
    if not new_code:
        return jsonify({"success": False, "error": "Code required."}), 400

    if not os.path.exists(MAPPINGS_FILE):
        return jsonify({"success": False, "error": "Mappings file missing."}), 404

    with open(MAPPINGS_FILE, "r", newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))

    if row_id < 0 or row_id >= len(rows):
        return jsonify({"success": False, "error": "Entry not found."}), 404

    row = rows[row_id]
    if len(row) < 2:
        return jsonify({"success": False, "error": "Entry missing code."}), 400

    row[1] = new_code
    rows[row_id] = row

    with open(MAPPINGS_FILE, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)

    reload_exact_mappings()
    return jsonify({"success": True})


@app.route("/api/training_schedule", methods=["GET"])
def api_training_schedule():
    schedule = load_training_schedule()
    return jsonify(schedule)


@app.route("/api/training_schedule", methods=["POST"])
def update_training_schedule():
    payload = request.get_json(silent=True) or {}
    enabled = bool(payload.get("enabled"))
    time_value = normalize_schedule_time(payload.get("time"))
    if not time_value:
        return jsonify({"success": False, "error": "Invalid time format. Use HH:MM."}), 400

    schedule = load_training_schedule()
    changed = schedule.get("enabled") != enabled or schedule.get("time") != time_value
    schedule["enabled"] = enabled
    schedule["time"] = time_value
    if changed:
        schedule["last_run"] = None
    save_training_schedule(schedule)
    return jsonify({"success": True, "schedule": schedule})


def _settings_for_api():
    """Return settings with API key masked for API response."""
    s = load_app_settings()
    if s.get("llm", {}).get("api_key"):
        s = dict(s)
        s["llm"] = dict(s["llm"])
        s["llm"]["api_key"] = "***"
    return s


@app.route("/api/settings", methods=["GET"])
def api_settings():
    return jsonify(_settings_for_api())


@app.route("/api/settings", methods=["POST"])
def update_settings():
    payload = request.get_json(silent=True) or {}
    saved = save_app_settings(payload)
    return jsonify({"success": True, "settings": saved})


@app.route("/api/term-replacements", methods=["GET"])
def api_get_term_replacements():
    replacements = load_term_replacements()
    return jsonify({"replacements": replacements})


@app.route("/api/term-replacements", methods=["POST"])
def api_save_term_replacements():
    payload = request.get_json(silent=True) or {}
    entries = payload.get("replacements", [])
    if not isinstance(entries, list):
        return jsonify({"success": False, "error": "Invalid replacements payload."}), 400

    normalized = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        pattern = (entry.get("pattern") or "").strip()
        replacement = (entry.get("replacement") or "").strip()
        if not pattern or not replacement:
            continue
        try:
            re.compile(pattern)
        except re.error:
            continue
        normalized.append({"pattern": pattern, "replacement": replacement})

    save_term_replacements(normalized)
    return jsonify({"success": True, "replacements": normalized})


@app.route("/api/training_status", methods=["GET"])
def api_training_status():
    with TRAINING_LOCK:
        return jsonify(dict(TRAINING_STATUS))


@app.route("/api/cancel_training", methods=["POST"])
def api_cancel_training():
    with TRAINING_LOCK:
        if not TRAINING_STATUS["is_training"]:
            return jsonify({"success": False, "error": "No training in progress"})
        TRAINING_STATUS["cancelled"] = True
        TRAINING_STATUS["is_training"] = False
        TRAINING_STATUS["message"] = "Cancelling..."
        TRAINING_STATUS["message_base"] = None
        TRAINING_STATUS["stage"] = None
        TRAINING_STATUS["stage_started_at"] = None
    return jsonify({"success": True})


@app.route("/train", methods=["POST"])
def train():
    if not start_training():
        return jsonify({"success": False, "error": "Training already in progress"}), 409
    return jsonify({"success": True, "status_url": "/api/training_status"})


@app.route("/api/stats")
def api_stats():
    """Get match statistics."""
    try:
        conn = _history_connection()
        cursor = conn.cursor()

        # Total matches
        total = cursor.execute("SELECT COUNT(*) FROM match_history").fetchone()[0]

        # Matches in last hour
        last_hour = cursor.execute(
            "SELECT COUNT(*) FROM match_history WHERE created_at >= datetime('now', '-1 hour')"
        ).fetchone()[0]

        # Matches in last 24 hours
        last_24h = cursor.execute(
            "SELECT COUNT(*) FROM match_history WHERE created_at >= datetime('now', '-24 hours')"
        ).fetchone()[0]

        # Average per hour (last 24h)
        avg_per_hour = round(last_24h / 24, 1) if last_24h > 0 else 0

        # Average per minute (last hour)
        avg_per_minute = round(last_hour / 60, 1) if last_hour > 0 else 0

        conn.close()

        return jsonify({
            "total_matches": total,
            "last_hour": last_hour,
            "last_24h": last_24h,
            "avg_per_hour": avg_per_hour,
            "avg_per_minute": avg_per_minute,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/health")
def api_health():
    current = get_matcher()
    return jsonify(
        {
            "status": "ok",
            "loaded": True,
            "num_codes": len(current.codes),
            "num_mappings": len(current.exact_mappings),
            "version": "3.0",
        }
    )


if __name__ == "__main__":
    # Suppress Flask's default access logs
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)

    host = os.environ.get("RADMATCHER_HOST", "0.0.0.0")
    try:
        port = int(os.environ.get("RADMATCHER_PORT", "5000"))
    except ValueError:
        port = 5000

    print("\n" + "=" * 80)
    print(f"RadMatcher v{VERSION} - Running on http://localhost:{port}")
    print("=" * 80)
    print("\nLoading matcher (this can take a minute on first start)...")
    start_time = time.time()
    get_matcher()
    elapsed = int(time.time() - start_time)
    print(f"Matcher ready in {elapsed}s")
    print("\nMatch Logging Enabled\n")
    app.run(host=host, port=port, debug=False)
