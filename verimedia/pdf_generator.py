import base64
import datetime
import re
from io import BytesIO
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import (
    Image,
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)

CJK_REGEX = re.compile(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]')


def _ensure_cjk_font():
    """Register ReportLab built-in CJK CID font STSong-Light and return its font name."""
    font_name = 'STSong-Light'
    try:
        pdfmetrics.getFont(font_name)
    except KeyError:
        try:
            pdfmetrics.registerFont(UnicodeCIDFont(font_name))
        except Exception as e:
            print(f"Warning: Could not register CJK font {font_name}: {e}")
            return 'Helvetica'
    return font_name


def _contains_cjk(text):
    """Return True if text contains Chinese/Japanese/Korean characters."""
    if not text:
        return False
    return bool(CJK_REGEX.search(str(text)))


def _safe_paragraph_text(text):
    """Escape XML special characters (&, <, >) and convert single newlines to <br/>."""
    if not text:
        return ""
    escaped = escape(str(text).strip())
    return escaped.replace('\n', '<br/>')


def get_default_suggestions_for_level(toxicity_level):
    """Fallback suggestions tailored to each toxicity level for PDF export."""
    level = (toxicity_level or '').lower()
    if level == 'none':
        return [
            "Continue using inclusive and balanced language",
            "Maintain your ethical reporting approach",
            "Keep providing accurate and fair content",
            "Your content demonstrates responsible reporting practices",
            "The content is free from xenophobic or problematic language",
        ]
    if level == 'mild':
        return [
            "Consider alternative phrasing for potentially problematic terms",
            "Add more context to sensitive topic discussions",
            "Review specific highlighted sections for more balanced perspectives",
            "Ensure diverse viewpoints are represented",
            "Be more specific when referring to groups to avoid generalizations",
        ]
    if level == 'high':
        return [
            "Substantially revise sections containing problematic language",
            "Replace generalizations with specific, fact-based statements",
            "Add context, nuance, and multiple perspectives to balance the narrative",
            "Consider consulting cultural sensitivity experts for revisions",
            "Implement fact-checking for claims made in the content",
        ]
    if level == 'max':
        return [
            "Consider a comprehensive rewrite following ethical guidelines",
            "Consult with subject matter experts on ethical representation",
            "Implement a thorough editorial review process",
            "Address all xenophobic elements identified in the analysis",
            "Ensure all claims are supported by credible sources",
        ]
    return ["Review content for ethical and inclusive reporting standards."]


def build_report_pdf(
    toxicity_level='Unknown',
    report_content='',
    suggestions=None,
    xenophobic_words=None,
    wordcloud_image=None,
    transcription=None,
):
    """
    Build a complete PDF analysis report and return raw PDF bytes.
    Supports CJK characters, XML escaping, multi-paragraph layout, and embedded WordCloud.
    """
    suggestions = list(suggestions or [])
    xenophobic_words = list(xenophobic_words or [])
    level_lower = (toxicity_level or 'unknown').lower()

    if level_lower == 'none' and not suggestions:
        suggestions = get_default_suggestions_for_level('none')
    elif not suggestions or len(suggestions) < 3:
        suggestions = get_default_suggestions_for_level(level_lower)

    if not report_content or report_content == 'No report available' or report_content.startswith('Error:'):
        report_content = (
            "The analysis could not generate a detailed report for this content. "
            "Please review the provided suggestions and consider running the analysis again."
        )

    # Detect whether any content requires CJK font rendering
    all_text = " ".join([
        str(toxicity_level),
        str(report_content),
        " ".join(str(s) for s in suggestions),
        " ".join(str(w) for w in xenophobic_words),
        str(transcription or ""),
    ])
    use_cjk = _contains_cjk(all_text)
    base_font = _ensure_cjk_font() if use_cjk else 'Helvetica'
    bold_font = base_font if use_cjk else 'Helvetica-Bold'

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        name='VeriMediaTitle',
        parent=styles['Heading1'],
        fontName=bold_font,
        fontSize=16,
        alignment=TA_CENTER,
        spaceAfter=20,
    )
    subtitle_style = ParagraphStyle(
        name='VeriMediaSubtitle',
        parent=styles['Heading2'],
        fontName=bold_font,
        fontSize=13,
        spaceBefore=8,
        spaceAfter=10,
    )
    body_style = ParagraphStyle(
        name='VeriMediaBody',
        parent=styles['Normal'],
        fontName=base_font,
        fontSize=10.5,
        leading=15,
        alignment=TA_JUSTIFY,
        spaceAfter=8,
    )
    normal_style = ParagraphStyle(
        name='VeriMediaNormal',
        parent=styles['Normal'],
        fontName=base_font,
        fontSize=10.5,
        leading=14,
    )
    warning_title_style = ParagraphStyle(
        name='VeriMediaWarning',
        parent=normal_style,
        textColor=colors.red,
        fontName=bold_font,
        spaceAfter=4,
    )

    content = []

    # Header & Toxicity Level
    content.append(Paragraph("VERIMEDIA ANALYSIS REPORT", title_style))
    content.append(Spacer(1, 8))
    content.append(Paragraph(f"TOXICITY LEVEL: {_safe_paragraph_text(toxicity_level)}", subtitle_style))
    content.append(Spacer(1, 8))

    # Section 1: Suggestions
    section_titles = {
        'none': "POSITIVE CONTENT ATTRIBUTES:",
        'mild': "RECOMMENDED IMPROVEMENTS:",
        'high': "NECESSARY REVISIONS:",
        'max': "CRITICAL CHANGES REQUIRED:",
    }
    content.append(Paragraph(section_titles.get(level_lower, "IMPROVEMENT SUGGESTIONS:"), subtitle_style))

    if suggestions:
        suggestion_items = [
            ListItem(Paragraph(_safe_paragraph_text(s), normal_style))
            for s in suggestions if str(s).strip()
        ]
        content.append(
            ListFlowable(
                suggestion_items,
                bulletType='bullet',
                leftIndent=20,
                bulletFontSize=8,
                bulletOffsetY=2,
            )
        )
    else:
        content.append(Paragraph("No suggestions available.", normal_style))

    content.append(Spacer(1, 12))

    # Section 2: Word Cloud & Identified Words
    if wordcloud_image and level_lower != 'none':
        try:
            img_data = base64.b64decode(wordcloud_image)
            img_bio = BytesIO(img_data)
            content.append(Paragraph("XENOPHOBIC WORDS VISUALIZATION:", subtitle_style))
            content.append(Image(img_bio, width=450, height=225))
            if xenophobic_words:
                content.append(Spacer(1, 10))
                content.append(Paragraph("Identified Problematic / Xenophobic Terms:", normal_style))
                words_text = ", ".join(_safe_paragraph_text(w) for w in xenophobic_words)
                content.append(Paragraph(words_text, body_style))
            content.append(Spacer(1, 12))
        except Exception as img_err:
            print(f"Warning: Could not embed word cloud in PDF: {img_err}")

    # Section 3: Comprehensive Report (rendered paragraph by paragraph)
    report_titles = {
        'none': "CONTENT ASSESSMENT:",
        'mild': "ANALYSIS REPORT:",
        'high': "TOXICITY ASSESSMENT REPORT:",
        'max': "CRITICAL ETHICS REPORT:",
    }
    content.append(Paragraph(report_titles.get(level_lower, "COMPREHENSIVE REPORT:"), subtitle_style))

    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', str(report_content)) if p.strip()]
    for para in paragraphs:
        content.append(Paragraph(_safe_paragraph_text(para), body_style))

    # Section 4: Transcription (if audio/video)
    if transcription and str(transcription).strip():
        content.append(Spacer(1, 10))
        content.append(Paragraph("FULL TRANSCRIPTION:", subtitle_style))
        for para in [p.strip() for p in re.split(r'\n\s*\n', str(transcription)) if p.strip()]:
            content.append(Paragraph(_safe_paragraph_text(para), body_style))

    # Severity guidance footer
    if level_lower == 'high':
        content.append(Spacer(1, 10))
        content.append(Paragraph("IMPORTANT NOTE:", warning_title_style))
        content.append(
            Paragraph(
                "Content with high toxicity levels may negatively impact your audience and reputation. "
                "We strongly recommend implementing the revisions outlined in this report before publication or distribution.",
                body_style,
            )
        )
    elif level_lower == 'max':
        content.append(Spacer(1, 10))
        content.append(Paragraph("URGENT ACTION REQUIRED:", warning_title_style))
        content.append(
            Paragraph(
                "Content with maximum toxicity levels is likely to cause harm if published or distributed in its current form. "
                "We strongly advise against using this content without comprehensive revision following the guidelines provided in this report.",
                body_style,
            )
        )

    content.append(Spacer(1, 18))
    current_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    content.append(Paragraph("Generated by VeriMedia - UNICC", normal_style))
    content.append(Paragraph(f"Date: {current_date}", normal_style))

    doc.build(content)
    pdf_data = buffer.getvalue()
    buffer.close()
    return pdf_data
