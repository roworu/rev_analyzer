# domain_model.py
import logging
import re
from typing import List, Tuple

import numpy as np
from transformers import pipeline
import joblib
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

_log = logging.getLogger(__name__)

# ----- config -----
MODEL_NAMES = [
    "cardiffnlp/twitter-roberta-base-sentiment",   # often 3-class
    "distilbert-base-uncased-finetuned-sst-2-english",  # 2-class
]

DEFAULT_TEMPERATURE = 1.5
CALIBRATOR_PATH = "calibrator.joblib"

# ----- globals -----
_pipelines: List = []
_calibrator = None


def init_models():
    """
    Initialize sentiment pipelines and optional calibrator.
    Any exception during init is caught and fully logged (stacktrace).
    Call this from your app startup (and check logs).
    """
    global _pipelines, _calibrator
    _pipelines = []
    _calibrator = None

    try:
        for mn in MODEL_NAMES:
            try:
                p = pipeline("sentiment-analysis", model=mn)
                _pipelines.append(p)
                # id2label may exist or not; keep it readable in logs
                try:
                    id2label = p.model.config.id2label
                except Exception:
                    id2label = None
                _log.info(f"[domain_model] loaded {mn}, id2label={id2label}")
            except Exception:
                _log.exception(f"[domain_model] failed to load model {mn} — continuing with remaining models")

        # try to load calibrator, but don't crash if missing/bad
        try:
            _calibrator = joblib.load(CALIBRATOR_PATH)
            _log.info(f"[domain_model] loaded calibrator from {CALIBRATOR_PATH}")
        except FileNotFoundError:
            _log.info(f"[domain_model] no calibrator file found at {CALIBRATOR_PATH} (ok)")
            _calibrator = None
        except Exception:
            _log.exception(f"[domain_model] failed to load calibrator {CALIBRATOR_PATH} (ignored)")
            _calibrator = None

    except Exception:
        # Very defensive: catch absolutely everything and log full traceback
        _log.exception("[domain_model] unexpected error during init_models — falling back to empty pipeline")
        _pipelines = []
        _calibrator = None


def _result_to_pos_prob(res: dict) -> float:
    """
    Convert pipeline result to probability of positive sentiment in [0,1].
    Supports 2-class and 3-class models, common label formats.
    """
    label = str(res.get("label", "")).lower()
    score = float(res.get("score", 0.0))

    # Common 2-class: POSITIVE/NEGATIVE or LABEL_1/LABEL_0
    if "positive" in label or label.endswith("_1"):
        return score
    if "negative" in label or label.endswith("_0"):
        return 1.0 - score

    # 3-class: LABEL_2 or 'neutral' -> treat as neutral (0.5)
    if label.endswith("_2") or "neutral" in label:
        return 0.5

    # Try to extract numeric stars if present
    m = re.search(r"([1-5])", label)
    if m:
        stars = int(m.group(1))
        return (stars - 1) / 4.0

    # fallback: use raw score
    return score


def _ensemble_prob(text: str) -> float:
    """
    Run all loaded pipelines and return mean positive-prob.
    Safe: if no pipelines are loaded, return neutral 0.5.
    """
    probs = []
    for p in _pipelines:
        try:
            r = p(text, truncation=True)[0]
            probs.append(_result_to_pos_prob(r))
        except Exception:
            _log.exception("[domain_model] pipeline inference failed for one model (skipping)")
    if not probs:
        _log.warning("[domain_model] no pipelines available; returning neutral probability 0.5")
        return 0.5
    return float(np.mean(probs))


def _apply_calibrator(prob: float) -> float:
    global _calibrator
    if _calibrator is None:
        return prob
    try:
        if hasattr(_calibrator, "predict_proba"):
            return float(_calibrator.predict_proba(np.array([[prob]]))[:, 1][0])
        return float(_calibrator.predict(np.array([prob]))[0])
    except Exception:
        _log.exception("[domain_model] calibrator failed during application; ignoring calibrator")
        return prob


def _temp_scale(prob: float, T: float) -> float:
    eps = 1e-6
    p = float(np.clip(prob, eps, 1 - eps))
    logit = np.log(p / (1.0 - p))
    return float(1.0 / (1.0 + np.exp(-logit / float(T))))


def _prob_to_grade(prob: float) -> int:
    # linear mapping 0..1 -> 1..10
    return int(round(1 + 9 * float(np.clip(prob, 0.0, 1.0))))


def predict(text: str, temperature: float = DEFAULT_TEMPERATURE) -> Tuple[int, float]:
    """
    Public predict method: returns (grade:int 1..10, confidence:float 0..1).
    Never raises due to missing models; logs warnings and returns neutral fallback if needed.
    """
    raw_prob = _ensemble_prob(text)
    prob = _apply_calibrator(raw_prob)
    if temperature and temperature != 1.0:
        prob = _temp_scale(prob, temperature)
    grade = _prob_to_grade(prob)
    _log.info(f"[domain_model] prediction made: grade={grade}, prob={float(prob)}")
    return grade, float(prob)



def fit_calibrator(raw_probs: List[float], y_true: List[int], method: str = "isotonic"):
    """
    Fit and persist a calibrator from a labeled set. Saves to CALIBRATOR_PATH.
    """
    X = np.array(raw_probs).reshape(-1, 1)
    y = np.array(y_true).astype(int)
    if method == "isotonic":
        cal = IsotonicRegression(out_of_bounds="clip").fit(raw_probs, y)
    else:
        cal = LogisticRegression().fit(X, y)
    joblib.dump(cal, CALIBRATOR_PATH)
    _log.info(f"[domain_model] saved calibrator to {CALIBRATOR_PATH}")
    return cal
