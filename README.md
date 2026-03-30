# 🎬 Bullet Script Analyzer

A production-ready AI tool designed to analyze short-form scripted content (microdramas, web series, OTT shorts). It generates structured storytelling insights including emotional arcs, engagement scoring, actionable improvement suggestions, and cliffhanger detection.

Built as a **100% local, zero-cost, privacy-first** application using Ollama and Qwen3. No API keys required.
Demo Video - https://drive.google.com/file/d/1ADsTunK-u0n8xYZRZdnAWqExQVPwBFsf/view?usp=drive_link
---

## 🚀 Key Features

- **Pydantic-Enforced Structured Output:** Uses Ollama's native JSON Schema constraints at the inference level for guaranteed, parse-safe `ScriptAnalysis` object generation.
- **Dual-Metric Engagement Scoring:** Provides transparency into AI evaluation by juxtaposing the model's *Holistic Score* against a computed *Factor Average*.
- **Side-by-Side Comparison:** Built-in "Compare" tab to analyze revisions or benchmark two different scripts against each other on all dimensions.
- **Model Auto-Detection:** Automatically scales to the user's hardware by detecting and selecting the best available Qwen3 variant (32b → 14b → 8b).
- **Multi-Genre Samples:** Pre-loaded with Drama, Thriller, and Romance scripts for immediate, zero-friction demonstrations.
- **Export Ready:** Download evaluations as raw JSON for data pipelines or formatted text for writers' rooms.

---

## ⚙️ Tech Stack & Architecture

- **LLM:** Qwen3 (via Ollama) — Apache 2.0 licensed, optimized for creative reasoning.
- **UI:** Streamlit — Customized with premium CSS (gradient headers, metric cards, status pills).
- **Validation:** Pydantic v2.

### The "Single-Call" Design Paradigm
The analyzer intentionally uses a **single LLM call** to populate the entire 6-dimension schema.
* **Why?** Cross-field structural coherence. If an emotional arc peaks at intensity 9, the single-call context ensures the engagement factors organically recognize and score that tension build. Multi-call architectures (e.g., summarizing, then scoring separately) often result in contradictory insights.

### Native JSON Enforcement
We do **not** use prompt-engineering tricks (like "respond only in JSON") to get structured data.
Instead, we pass `ScriptAnalysis.model_json_schema()` directly to Ollama's `format=` parameter. This constrains the LLM token generation at the lowest inference level—making it physically impossible for the model to output markdown fences, preambles, or schema violations.

---

## 🧠 Prompt Engineering Strategy

- **Persona Priming:** The system prompt establishes the model as an *expert in short-form content* (microdramas, OTT shorts), anchoring the analysis to the pacing and hook-heavy expectations of the format.
- **Speed Optimization (`/no_think`):** Qwen3's chain-of-thought is explicitly disabled via the `/no_think` prefix. For creative script analysis—unlike deep math reasoning—the quality remains high while inference time is cut by ~40%.
- **Grounded Rubrics:** The user prompt forces the model to justify every score and suggestion by citing *exact moments or dialogue* from the script, preventing lazy, generic AI advice.
- **Determinism (`temperature=0`):** Maximizes schema adherence and prevents hallucinated scores by forcing the highest-probability tokens.

---

## 📊 Deep Dive: The Engagement Scoring System

A common flaw in AI-driven evaluation is the "black box" score. To solve this for AI Product leaders and Content Directors, this tool uses a **Dual-Metric Transparency System**.

### 1. The LLM Holistic Score (e.g., 8.7/10)
This score is generated directly by the LLM in the main analysis pass. It represents the model's synthesized judgment of the script's overall gripping power.
**Real-world meaning:** It captures *emergent* qualities of storytelling—like genre subversion, cultural timing, or that indescribable "it factor"—that strict rubrics often miss.

### 2. The Factor Average (e.g., 8.0/10)
This is a mathematically computed arithmetic mean of the five individual `engagement_factors` (Hook, Conflict, Dialogue, Tension, Cliffhanger) graded by the model.
**Real-world meaning:** It represents the script's mechanical, structural soundness.

### Interpreting the Delta between the two:
By displaying both side-by-side, the UI provides immediate diagnostic value:
* **Score > Average (e.g., 8.7 vs 8.0):** The script has deep emergent appeal. It might have flawed mechanics (like clunky dialogue), but the core concept or hook is so strong it elevates the piece.
* **Score < Average (e.g., 6.5 vs 8.0):** The script is structurally textbook, but lacks a soul or defining hook. It "ticks the boxes" but fails to truly engage.
* **Score ≈ Average:** The script's holistic quality perfectly scales with its mechanical execution.

This transparency proves to users that the score isn't just a random number generation—it's a grounded, debatable metric.

---

## 🛠️ Local Setup & Run

### 1. Install Ollama
Download from [https://ollama.com](https://ollama.com) and install.

### 2. Pull the Qwen3 Model
```bash
ollama pull qwen3:8b        # ~5GB download, runs smoothly on 8GB RAM
# For better reasoning depth (requires 16GB RAM):
ollama pull qwen3:14b
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch Desktop App
```bash
# Terminal 1: Start inference server
ollama serve

# Terminal 2: Launch UI
cd bullet-script-analyzer
streamlit run app.py
```
Navigate to the provided `localhost:8501` link. The app will auto-detect the best Qwen3 model you pulled.

---

## 🛤️ Roadmap & Possible Improvements

- **Diff-Highlighted Revisions:** Enhancing the Compare tab to show exact textual diffs alongside the changing emotional beats between Draft 1 and Draft 2.
- **Real-World Calibration:** Fine-tuning a Qwen3 LoRA against Bullet's actual OTT viewership retention data to align the Engagement Score away from LLM heuristics and toward real-world performance metrics.
- **File Parsing:** Adding unstructured ingestion via PyPDF2 or `python-docx` for drag-and-drop script loading.
- **Genre-Dynamic UI:** Extracting the `SAMPLE_SCRIPTS` genre list into an inference parameter, allowing the LLM to dynamically swap its rubric (e.g., prioritizing "Joke Density" for Comedy instead of "Tension" for Thriller).
