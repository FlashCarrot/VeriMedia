import os
import re
from flask import (
    Flask,
    flash,
    jsonify,
    make_response,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from werkzeug.utils import secure_filename

from verimedia.analyzer import (
    SAMPLE_PRESETS,
    analyze_audio_file,
    analyze_text_file,
    analyze_text_input,
    analyze_video_file,
    resolve_api_config,
)
from verimedia.config import (
    ALLOWED_EXTENSIONS,
    MAX_AUDIO_SIZE,
    MAX_CONTENT_LENGTH,
    MAX_VIDEO_SIZE,
    UPLOAD_FOLDER,
    allowed_file,
    get_env_api_key,
    get_fine_tuned_model_id,
)
from verimedia.media_processor import convert_video_to_audio
from verimedia.pdf_generator import build_report_pdf
from verimedia.result_store import (
    cleanup_temp_files,
    load_analysis_result,
    save_analysis_result,
)
from verimedia.visualizer import get_chinese_font

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key-for-development')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
app.config['MAX_VIDEO_SIZE'] = MAX_VIDEO_SIZE
app.config['MAX_AUDIO_SIZE'] = MAX_AUDIO_SIZE
app.config['ALLOWED_EXTENSIONS'] = ALLOWED_EXTENSIONS

FINE_TUNED_MODEL_ID = get_fine_tuned_model_id()
if FINE_TUNED_MODEL_ID:
    print(f"Configured fine-tuned model ID: {FINE_TUNED_MODEL_ID}")


def _get_request_api_settings():
    """Retrieve API key / base URL / demo mode settings from current session or form."""
    custom_key = (
        request.form.get('custom_api_key')
        or session.get('custom_api_key')
        or None
    )
    custom_base_url = (
        request.form.get('custom_base_url')
        or session.get('custom_base_url')
        or None
    )
    force_demo = bool(session.get('force_demo', False))
    return custom_key, custom_base_url, force_demo


@app.context_processor
def inject_app_context():
    """Provide API / Demo mode status and sample presets to all templates."""
    custom_key = session.get('custom_api_key')
    custom_base_url = session.get('custom_base_url')
    force_demo = bool(session.get('force_demo', False))
    resolved = resolve_api_config(
        custom_api_key=custom_key,
        custom_base_url=custom_base_url,
        force_demo=force_demo,
    )
    return {
        'api_status': {
            'is_demo': resolved['is_demo'],
            'source': resolved['source'],
            'has_env_key': bool(get_env_api_key()),
            'has_session_key': bool(custom_key),
            'force_demo': force_demo,
            'base_url': custom_base_url or os.environ.get('OPENAI_BASE_URL', ''),
        },
        'sample_presets': SAMPLE_PRESETS,
    }


@app.before_request
def before_request():
    cleanup_temp_files()


@app.errorhandler(413)
def request_entity_too_large(error):
    flash('The file is too large. Maximum allowed size is 500MB.', 'error')
    return redirect(url_for('index')), 413


@app.route('/')
def index():
    """Render the home page."""
    return render_template('index.html')


@app.route('/about')
def about():
    """Render the about page."""
    return render_template('about.html')


@app.route('/settings/api-key', methods=['POST'])
def update_api_settings():
    """Configure or clear session-scoped OpenAI API key, Base URL, or Demo Mode."""
    action = request.form.get('action', 'save')
    if action == 'clear':
        session.pop('custom_api_key', None)
        session.pop('custom_base_url', None)
        session['force_demo'] = False
        flash('Cleared custom API settings.', 'success')
    elif action == 'demo':
        session['force_demo'] = True
        flash('Switched to Interactive Demo Mode (no OpenAI API key required).', 'success')
    else:
        api_key = request.form.get('api_key', '').strip()
        base_url = request.form.get('base_url', '').strip()
        force_demo = request.form.get('force_demo') == 'on'

        if api_key:
            session['custom_api_key'] = api_key
            session['force_demo'] = False
        else:
            session['force_demo'] = force_demo

        if base_url:
            session['custom_base_url'] = base_url
        else:
            session.pop('custom_base_url', None)

        if session.get('force_demo'):
            flash('Demo Mode enabled.', 'success')
        elif api_key or get_env_api_key():
            flash('OpenAI API configuration updated for this session.', 'success')
        else:
            flash('No API key provided; running in Demo Mode automatically.', 'warning')

    return redirect(request.referrer or url_for('index'))


def _store_and_render_result(result):
    """
    Cache the complete result on disk (avoiding the 4KB cookie overflow) and render results.html.
    """
    if result.get('toxicity_level') == 'Error':
        flash(result.get('report', 'An error occurred during analysis.'), 'error')
        return redirect(url_for('index'))

    # Split report and transcription into clean paragraphs for safe template rendering (no |safe XSS)
    report_text = str(result.get('report', '') or '')
    result['report_paragraphs'] = [
        p.strip() for p in re.split(r'\n\s*\n', report_text) if p.strip()
    ]
    transcription_text = str(result.get('transcription', '') or '')
    result['transcription_paragraphs'] = (
        [p.strip() for p in re.split(r'\n\s*\n', transcription_text) if p.strip()]
        if transcription_text
        else []
    )

    result_id = save_analysis_result(result)
    result['result_id'] = result_id

    # Store only the lightweight result_id in Flask session cookie (<100 bytes)
    session['last_result_id'] = result_id
    return render_template('results.html', result=result)


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle media file upload or direct text input and return ethical analysis results."""
    file_path = None
    custom_key, custom_base_url, force_demo = _get_request_api_settings()

    try:
        file_type = request.form.get('file_type', 'text')
        input_mode = request.form.get('input_mode', 'file')
        direct_text = request.form.get('text_content', '').strip()

        # Support direct text input / preset samples when in text mode
        if file_type == 'text' and (input_mode == 'paste' or direct_text):
            if not direct_text:
                flash('Please enter or select sample text to analyze.', 'error')
                return redirect(url_for('index'))
            result = analyze_text_input(
                direct_text,
                api_key=custom_key,
                base_url=custom_base_url,
                force_demo=force_demo,
            )
            return _store_and_render_result(result)

        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(url_for('index'))

        file = request.files['file']
        if not file or file.filename == '':
            flash('No file selected', 'error')
            return redirect(url_for('index'))

        # Check file size before saving
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        if file_type == 'video' and file_size > app.config['MAX_VIDEO_SIZE']:
            flash(
                f'The video file is too large. Maximum allowed size is {app.config["MAX_VIDEO_SIZE"] / (1024 * 1024):.0f}MB.',
                'error',
            )
            return redirect(url_for('index'))
        if file_type == 'audio' and file_size > app.config['MAX_AUDIO_SIZE']:
            flash(
                f'The audio file is too large. Maximum allowed size is {app.config["MAX_AUDIO_SIZE"] / (1024 * 1024):.0f}MB.',
                'error',
            )
            return redirect(url_for('index'))

        if not allowed_file(file.filename, file_type):
            flash('Invalid file type for the selected category.', 'error')
            return redirect(url_for('index'))

        filename = secure_filename(file.filename)
        if not filename:
            filename = f"upload.{file.filename.rsplit('.', 1)[-1].lower()}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        try:
            result = process_file(
                file_path,
                file_type,
                api_key=custom_key,
                base_url=custom_base_url,
                force_demo=force_demo,
            )
            return _store_and_render_result(result)
        except Exception as e:
            error_msg = str(e)
            if "Could not extract audio" in error_msg:
                flash(
                    "Could not extract audio from the video. Please ensure the video contains an audio track.",
                    'error',
                )
            elif "too large" in error_msg.lower():
                flash(
                    "The extracted audio is too large for processing. Please use a shorter video.",
                    'error',
                )
            else:
                flash(f"Error processing file: {error_msg}", 'error')
            return redirect(url_for('index'))
        finally:
            if file_path and os.path.exists(file_path):
                try:
                    os.unlink(file_path)
                except OSError as cleanup_error:
                    print(f"Error cleaning up file {file_path}: {cleanup_error}")

    except Exception as e:
        if file_path and os.path.exists(file_path):
            try:
                os.unlink(file_path)
            except OSError:
                pass
        flash(f"An unexpected error occurred: {str(e)}", 'error')
        return redirect(url_for('index'))


def process_file(file_path, file_type, api_key=None, base_url=None, force_demo=False):
    """Route an uploaded file to the appropriate media analyzer."""
    if file_type == 'text':
        return analyze_text(file_path, api_key=api_key, base_url=base_url, force_demo=force_demo)
    if file_type == 'audio':
        return analyze_audio(file_path, api_key=api_key, base_url=base_url, force_demo=force_demo)
    if file_type == 'video':
        return analyze_video(file_path, api_key=api_key, base_url=base_url, force_demo=force_demo)

    return {
        'toxicity_level': 'Unknown',
        'suggestions': ['Unable to process this file type'],
        'report': 'No analysis available for this file type',
    }


def analyze_text(file_path, api_key=None, base_url=None, force_demo=False):
    """Analyze a text document (.txt, .pdf, .docx)."""
    return analyze_text_file(file_path, api_key=api_key, base_url=base_url, force_demo=force_demo)


def analyze_audio(file_path, api_key=None, base_url=None, force_demo=False):
    """Transcribe and analyze an audio recording."""
    return analyze_audio_file(file_path, content_type="audio", api_key=api_key, base_url=base_url, force_demo=force_demo)


def analyze_video(file_path, api_key=None, base_url=None, force_demo=False):
    """Extract audio from video, transcribe, and analyze."""
    return analyze_video_file(file_path, api_key=api_key, base_url=base_url, force_demo=force_demo)


@app.route('/download_report_pdf', methods=['GET', 'POST'])
def download_report_pdf():
    """Generate and download a PDF report using server-cached analysis results."""
    try:
        result_id = (
            request.form.get('result_id')
            or request.args.get('result_id')
            or session.get('last_result_id')
        )
        cached = load_analysis_result(result_id) if result_id else None

        if cached:
            toxicity_level = cached.get('toxicity_level', 'Unknown')
            report_content = cached.get('report', 'No report available')
            suggestions = cached.get('suggestions', [])
            xenophobic_words = cached.get('xenophobic_words', [])
            wordcloud_image = cached.get('wordcloud_image')
            transcription = cached.get('transcription')
        elif request.method == 'POST':
            toxicity_level = request.form.get('toxicity_level', 'Unknown')
            report_content = request.form.get('report_content', '')
            suggestions = session.get('suggestions', [])
            xenophobic_words = session.get('xenophobic_words', [])
            wordcloud_image = session.get('wordcloud_image')
            transcription = session.get('transcription')
        else:
            toxicity_level = request.args.get('toxicity_level', 'Unknown')
            report_content = session.get('report_content', 'No report available')
            suggestions = session.get('suggestions', [])
            xenophobic_words = session.get('xenophobic_words', [])
            wordcloud_image = session.get('wordcloud_image')
            transcription = session.get('transcription')

        pdf_data = build_report_pdf(
            toxicity_level=toxicity_level,
            report_content=report_content,
            suggestions=suggestions,
            xenophobic_words=xenophobic_words,
            wordcloud_image=wordcloud_image,
            transcription=transcription,
        )

        response = make_response(pdf_data)
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = 'attachment; filename=VeriMedia_Analysis_Report.pdf'
        return response

    except Exception as e:
        flash(f"Error generating PDF: {str(e)}", "error")
        return redirect(url_for('index'))


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5004))
    app.run(debug=True, port=port)
