
import numpy as np
import logging

# Setup logger for the detector
logger = logging.getLogger("harmonic-pro.detector")

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    logger.error("Librosa is not installed or missing system dependencies (libsndfile).")
    LIBROSA_AVAILABLE = False

# Profile wurden verfeinert, um charakteristische Intervalle stärker zu betonen.
# 1.0 = Grundton, 7 = Quinte, 3/4 = Terz
# Dorian zeichnet sich durch die GROSSE Sexte (Stufe 9) aus.
# Aeolian durch die KLEINE Sexte (Stufe 8).
MODE_PROFILES = {
    "Ionian (Major)":     [1.0, 0.0, 0.4, 0.0, 0.8, 0.6, 0.0, 1.0, 0.0, 0.4, 0.0, 0.8],
    "Dorian":            [1.0, 0.0, 0.3, 0.8, 0.0, 0.4, 0.0, 1.0, -0.5, 1.2, 0.8, 0.0],
    "Phrygian":          [1.0, 1.2, 0.0, 0.8, 0.0, 0.4, 0.0, 1.0, 0.4, 0.0, 0.8, 0.0],
    "Lydian":            [1.0, 0.0, 0.4, 0.0, 0.8, 0.0, 1.2, 1.0, 0.0, 0.4, 0.0, 0.8],
    "Mixolydian":        [1.0, 0.0, 0.4, 0.0, 0.8, 0.6, 0.0, 1.0, 0.0, 0.4, 1.2, 0.0],
    "Aeolian (Minor)":   [1.0, 0.0, 0.3, 0.8, 0.0, 0.4, 0.0, 1.0, 1.0, 0.0, 0.8, 0.0],
    "Locrian":           [1.0, 0.8, 0.0, 0.8, 0.0, 0.4, 1.2, 0.0, 0.4, 0.0, 0.8, 0.0]
}

PITCH_CLASSES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

class ModeDetector:
    def __init__(self, sample_rate=22050):
        self.sr = sample_rate

    def analyze(self, file_path):
        if not LIBROSA_AVAILABLE:
            return {
                "success": False,
                "error": "Backend Error: 'librosa' fehlt auf diesem System."
            }

        try:
            # Audio laden
            y, sr = librosa.load(file_path, sr=self.sr, duration=45) # 45 Sek. reichen meist
            
            # Harmonische Komponenten extrahieren
            y_harmonic = librosa.effects.harmonic(y)
            
            # Chroma CQT (Konstante Q-Transformation) ist musikalischer als STFT
            chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr, bins_per_octave=12)
            
            # Aggregation über die Zeit (Median ist robuster gegen Ausreißer)
            mean_chroma = np.median(chroma, axis=1)
            
            # Normalisierung
            if np.max(mean_chroma) > 0:
                mean_chroma = mean_chroma / np.max(mean_chroma)
            
            best_root_idx = 0
            best_root_score = -1.0
            
            # 1. Schritt: Grundton finden (Root Detection)
            # Wir suchen nach der stärksten Energieverteilung im Quintenzirkel-Verbund
            for root_idx in range(12):
                shifted = np.roll(mean_chroma, -root_idx)
                # Einfacher Root-Check: Grundton (0) und Quinte (7)
                score = shifted[0] + shifted[7] * 0.8
                if score > best_root_score:
                    best_root_score = score
                    best_root_idx = root_idx
            
            # 2. Schritt: Modus-Vergleich
            root_chroma = np.roll(mean_chroma, -best_root_idx)
            raw_scores = []
            
            for mode_name, profile in MODE_PROFILES.items():
                # Dot-Product zwischen Audio-Chroma und theoretischem Profil
                score = np.dot(root_chroma, profile)
                raw_scores.append({"mode": mode_name, "raw": float(score)})
            
            # In Prozent umrechnen
            min_raw = min(s["raw"] for s in raw_scores)
            adj_scores = [{"mode": s["mode"], "val": max(0, s["raw"] - min_raw)} for s in raw_scores]
            total_adj = sum(s["val"] for s in adj_scores)
            
            final_scores = []
            for s in adj_scores:
                final_scores.append({
                    "mode": s["mode"],
                    "score": round((s["val"] / total_adj) * 100, 1) if total_adj > 0 else 14.2
                })
            
            final_scores.sort(key=lambda x: x["score"], reverse=True)
            
            return {
                "root_note": PITCH_CLASSES[best_root_idx],
                "mode": final_scores[0]["mode"],
                "confidence": final_scores[0]["score"],
                "scores": final_scores,
                "success": True
            }
            
        except Exception as e:
            logger.exception("Fehler im Backend Algorithmus")
            return {"success": False, "error": str(e)}
