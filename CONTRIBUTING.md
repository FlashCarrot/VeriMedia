# Contributing to VeriMedia

Thank you for your interest in contributing to **VeriMedia** — an open-source, multimodal AI media ethics and anti-xenophobia analysis toolkit originally developed in collaboration with the **United Nations International Computing Centre (UNICC)**.

## Ways to Contribute

1. **Multilingual Datasets (`data/fine_tuning/`)**: Expand our open-source labeled examples for ethical reporting across additional languages and regional contexts.
2. **Model & Pipeline Integrations (`verimedia/analyzer.py`)**: Improve structured output schemas, fine-tuning scripts, and support for additional open-weights or API models.
3. **Media Processing & Accessibility (`verimedia/media_processor.py`, `verimedia/pdf_generator.py`)**: Enhance audio/video transcription pipelines and accessible report exports.

## Development Workflow

1. Fork the repository and create a feature branch:
   ```bash
   git checkout -b feat/your-feature-name
   ```
2. Set up the local virtual environment and install dependencies:
   ```bash
   ./setup.sh
   ```
3. Run the automated test suite before submitting a pull request:
   ```bash
   ./venv/bin/python -m unittest discover -s tests -v
   ```
4. Open a Pull Request with a clear summary of your changes.
