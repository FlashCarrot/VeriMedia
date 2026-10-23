# VeriMedia

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![UNICC Initiative](https://img.shields.io/badge/Initiative-UNICC%20Ethical%20AI-009EDB.svg)](https://www.unicc.org/)
[![UN Languages](https://img.shields.io/badge/UN%20Languages-AR%20%7C%20ZH%20%7C%20EN%20%7C%20FR%20%7C%20RU%20%7C%20ES-10b981.svg)](#key-features)

**VeriMedia** is an open-source, AI-powered multimodal media ethics toolkit developed as part of a **United Nations International Computing Centre (UNICC)** initiative. It empowers journalists, humanitarian organizations, researchers, and content creators to detect xenophobic framing, dehumanizing language, and misinformation across **text, audio, and video** in all **6 official UN languages**, providing actionable editorial guidance to foster accurate and empathetic public discourse on refugees, migrants, and marginalized communities.

---

## Why VeriMedia Matters to the Open-Source Ecosystem

- **Humanitarian & Public-Interest AI Safety**: Purpose-built aligned with UN and international ethical journalism guidelines to counter xenophobia and bias against forcibly displaced and indigenous populations.
- **Open 6-Language UN Toxicity Benchmark & Fine-Tuning Pipeline**: Ships with curated, labeled multilingual training and validation datasets (`data/fine_tuning/training.jsonl`, `validation.jsonl`) covering **all 6 Official Languages of the United Nations — Arabic (العربية), Chinese (中文), English, French (Français), Russian (Русский), and Spanish (Español)** — along with reproducible fine-tuning scripts (`prepare_fine_tuning.py`, `finetune.py`, `run_fine_tuning.py`).
- **End-to-End Multimodal Architecture**: Combines document parsing (`.txt`, `.pdf`, `.docx`), FFmpeg/MoviePy audio extraction & compression, Whisper speech-to-text transcription, Structured JSON LLM evaluation, CJK-ready word cloud visualization, and Unicode PDF report generation.
- **Zero-Barrier Interactive Demo Mode**: Includes an offline multilingual heuristic engine and one-click sample presets so researchers and educators can evaluate and demonstrate the platform immediately without paid API keys.

---

## Key Features

- **Text & Document Analysis**: Upload `.txt`, `.pdf`, or `.docx` files — or paste text directly in the browser — to evaluate toxicity severity (`None`, `Mild`, `High`, `Max`) and ethical framing.
- **Audio & Podcast Analysis**: Transcribe `.mp3`, `.wav`, `.ogg`, `.m4a`, and `.flac` recordings via Whisper and analyze spoken discourse for harmful stereotypes.
- **Video Broadcast Analysis**: Automatically extract, normalize, and compress audio tracks from `.mp4`, `.mov`, `.webm`, `.mkv`, and `.avi` videos for transcript and ethical evaluation.
- **Multilingual Word Cloud Visualization**: Automatically highlights problematic terms and generates downloadable word clouds with native CJK (Chinese/Japanese/Korean) font discovery.
- **Unicode PDF Report Export**: Generate publication-ready PDF ethics reports with full CJK character support (`STSong-Light`), embedded word clouds, and severity-tailored editorial recommendations.
- **Flexible AI Engine Configuration**: Use server `.env` keys, enter a temporary session-scoped OpenAI API key & custom `OPENAI_BASE_URL` directly in the UI, or run in offline **Interactive Demo Mode**.

---

## Project Architecture

```text
VeriMedia/
├── app.py                      # Flask web application & route controllers
├── verimedia/                  # Core modular Python package
│   ├── config.py               # Environment, limits, and model configuration
│   ├── analyzer.py             # Structured JSON LLM & Fine-tuned model engine + Demo Mode
│   ├── media_processor.py      # PDF/DOCX extraction & FFmpeg/MoviePy audio-video pipeline
│   ├── visualizer.py           # Multilingual/CJK WordCloud generator
│   ├── pdf_generator.py        # ReportLab PDF builder with CJK & XML safety
│   └── result_store.py         # Server-side result cache
├── data/
│   ├── fine_tuning/            # Open multilingual JSONL datasets (EN/ZH/FR/ES/AR)
│   └── training/               # Raw categorized training corpora
├── prepare_fine_tuning.py      # Dataset preparation CLI
├── finetune.py                 # Fine-tuning job management CLI
├── run_fine_tuning.py          # Automated fine-tuning runner & monitor
└── tests/
    └── test_verimedia.py       # Automated unit & integration test suite
```

---

## Quick Start

### Prerequisites

- Python 3.9+
- FFmpeg (bundled automatically via `imageio-ffmpeg`, or installed system-wide)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/FlashCarrot/VeriMedia.git
   cd VeriMedia
   ```

2. **Run the automated setup script** (creates `venv`, installs dependencies, and initializes `.env`):
   ```bash
   ./setup.sh
   ```

3. **Start the application**:
   ```bash
   ./venv/bin/python app.py
   ```

4. **Open in your browser**:
   Navigate to `http://127.0.0.1:5004/`. You can immediately test with **Interactive Demo Mode** and the built-in multilingual sample presets, or click **Configure API Key** in the top navigation bar to use live GPT-4o & Whisper models.

---

## Running Tests

```bash
./venv/bin/python -m unittest discover -s tests -v
```

---

## Team & Acknowledgments

Developed as part of the **UNICC (United Nations International Computing Centre)** project by:
- **Fengyu Yang** — Text Content & AI Pipeline Specialist
- **Qianwen Zhu** — Audio Content Specialist
- **Qi Sheng** — Video Content Specialist
- **Somya Panchal** — Integration and System Specialist

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
