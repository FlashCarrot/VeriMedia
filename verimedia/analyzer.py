import json
import os
import re
import httpx
from openai import OpenAI

from verimedia.config import (
    MAX_AUDIO_SIZE,
    MAX_TEXT_CHARS,
    VALID_TOXICITY_LEVELS,
    get_env_api_key,
    get_fine_tuned_model_id,
)
from verimedia.media_processor import (
    convert_audio_to_wav,
    convert_video_to_audio,
    extract_text_from_file,
)
from verimedia.visualizer import generate_wordcloud_base64

# Curated sample texts for instant Demo / 1-click testing
SAMPLE_PRESETS = {
    "ethical_en": {
        "title": "Balanced Refugee Reporting (English - None)",
        "content": (
            "Local communities and municipal agencies welcomed 240 displaced families arriving from conflict zones "
            "this week. According to UNHCR and city housing coordinators, transitional shelters and language classes "
            "have been established in partnership with neighborhood volunteers. Community leaders emphasized that "
            "accurate data, transparent resource allocation, and mutual respect are essential for long-term integration."
        ),
    },
    "mild_en": {
        "title": "Sensationalized Border Coverage (English - Mild)",
        "content": (
            "A massive flood of migrants continues to pour into border towns, overwhelming local schools and clinics. "
            "Many residents worry that these outsiders have no intention of adapting to local customs, even though "
            "aid groups argue that most arrivals are families seeking asylum through legal channels."
        ),
    },
    "high_zh": {
        "title": "偏见与刻板印象报道示例 (中文 - High)",
        "content": (
            "随着大量外来难民涌入城市，这些落后的群体正在破坏社区的文明秩序。支持同化政策的人士认为，"
            "这些外来者的原始习俗助长了懒惰和迷信，只有彻底抹去他们原有的文化身份并强制灌输主流价值观，"
            "才能解决所谓的外来人口问题，防止他们成为社会的负担。"
        ),
    },
    "max_en": {
        "title": "Dehumanizing Xenophobic Editorial (English - Max)",
        "content": (
            "Our nation is facing a full-scale invasion by savage illegals and parasites who bring crime, disease, "
            "and backwardness into civilized neighborhoods. These dangerous outsiders are subverting our culture "
            "and must be eradicated from our cities immediately before they destroy our society completely."
        ),
    },
}


def resolve_api_config(custom_api_key=None, custom_base_url=None, force_demo=False):
    """
    Determine the effective OpenAI API key, base URL, and whether Demo Mode should be used.
    Returns dict: {"api_key": str|None, "base_url": str|None, "is_demo": bool, "source": str}
    """
    if force_demo:
        return {"api_key": None, "base_url": None, "is_demo": True, "source": "demo"}

    key = (custom_api_key or "").strip()
    source = "session"
    if not key:
        key = get_env_api_key()
        source = "env" if key else "demo"

    base_url = (custom_base_url or os.environ.get("OPENAI_BASE_URL", "")).strip() or None
    return {
        "api_key": key,
        "base_url": base_url,
        "is_demo": not bool(key),
        "source": source,
    }


def create_openai_client(api_key, base_url=None):
    """Instantiate an OpenAI client safely."""
    kwargs = {
        "api_key": api_key,
        "http_client": httpx.Client(timeout=90.0),
    }
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def normalize_toxicity_level(raw_level):
    """Normalize any LLM-returned toxicity string to one of ('None', 'Mild', 'High', 'Max')."""
    if not raw_level:
        return "Mild"
    cleaned = str(raw_level).strip().lower()
    mapping = {
        "none": "None",
        "low": "None",
        "safe": "None",
        "mild": "Mild",
        "medium": "Mild",
        "moderate": "Mild",
        "high": "High",
        "severe": "High",
        "max": "Max",
        "maximum": "Max",
        "extreme": "Max",
        "critical": "Max",
    }
    for token, canonical in mapping.items():
        if re.search(rf"\b{token}\b", cleaned) or cleaned == token:
            return canonical
    return "Mild"


def determine_toxicity_level_with_ft(client, text_content, content_type="text"):
    """
    Attempt to classify toxicity using the fine-tuned model if configured.
    Gracefully returns None if no fine-tuned model exists or if the API key does not have access to it.
    """
    fine_tuned_model_id = get_fine_tuned_model_id()
    if not fine_tuned_model_id:
        return None

    system_messages = {
        "text": "You are an expert analyzing text content for ethical standards. Classify the toxicity level.",
        "audio": "You are an expert analyzing audio transcripts for ethical standards. Classify the toxicity level.",
        "video": "You are an expert analyzing video transcripts for ethical standards. Classify the toxicity level.",
    }
    system_message = system_messages.get(content_type, system_messages["text"])
    user_message = (
        f"Analyze the following {content_type} content for harmful language, misinformation, "
        f"and problematic content. Content: {text_content}"
    )

    try:
        response = client.chat.completions.create(
            model=fine_tuned_model_id,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            temperature=0.0,
            max_tokens=15,
        )
        raw_output = response.choices[0].message.content.strip()
        for valid in VALID_TOXICITY_LEVELS:
            if valid.lower() in raw_output.lower():
                return valid
        return None
    except Exception as e:
        print(f"Fine-tuned model skipped (falling back to primary GPT model): {e}")
        return None


def _apply_level_context(result_dict, content_type="text"):
    """
    Enrich the analysis result with consistent severity context, default suggestions if needed,
    and generate the WordCloud image when problematic words are present.
    """
    toxicity_level = normalize_toxicity_level(result_dict.get("toxicity_level"))
    result_dict["toxicity_level"] = toxicity_level
    level_lower = toxicity_level.lower()

    suggestions = [str(s).strip() for s in result_dict.get("suggestions", []) if str(s).strip()]
    report = str(result_dict.get("report", "")).strip()
    summary = str(result_dict.get("summary", "")).strip()
    xenophobic_words = [
        str(w).strip() for w in result_dict.get("xenophobic_words", []) if str(w).strip()
    ]

    media_label = {
        "text": "content",
        "audio": "audio content",
        "video": "video content",
    }.get(content_type, "content")

    if level_lower == "none":
        positive_report = (
            f"Your {media_label} has been analyzed and found to contain no problematic or xenophobic language. "
            f"The {media_label} appears to be ethical, balanced, and respectful. It presents information in a fair "
            f"and accurate manner that follows best practices for responsible reporting."
        )
        if summary:
            report = f"Content Summary: {summary}\n\n{positive_report}"
        elif report:
            report = f"{report}\n\n{positive_report}"
        else:
            report = positive_report

        if content_type == "video":
            report += (
                "\n\nNote: This analysis is based primarily on the audio transcript of the video. "
                "Visual elements should also be reviewed for ethical representation."
            )

        suggestions = [
            f"Continue using inclusive and balanced language in {media_label}",
            "Maintain your ethical reporting approach",
            f"Keep providing accurate and fair {media_label}",
        ]
        xenophobic_words = []

    elif level_lower == "mild":
        report += (
            f"\n\nMild Toxicity Context: {media_label.capitalize()} with mild toxicity typically contains subtle "
            "issues that can be improved with minor adjustments. While generally acceptable, addressing these "
            "minor concerns will enhance the ethical quality of your reporting."
        )
        if content_type == "video":
            report += (
                "\n\nNote: This analysis is based primarily on the audio content of the video. "
                "Visual elements should also be reviewed for ethical representation."
            )
        if len(suggestions) < 3:
            suggestions = [
                "Consider replacing generalized terms with more inclusive alternatives",
                "Add more context when discussing potentially sensitive topics",
                "Review specific sections that may benefit from more balanced perspectives",
                "Use more precise language when referring to groups of people",
            ]

    elif level_lower == "high":
        report += (
            f"\n\nHigh Toxicity Context: {media_label.capitalize()} with high toxicity contains significant "
            "issues that require substantial revision. The problematic elements identified may perpetuate "
            "harmful stereotypes or contain misinformation that should be addressed promptly."
        )
        if content_type == "video":
            report += (
                "\n\nNote: This analysis is based primarily on the audio content of the video. "
                "Visual elements should be carefully reviewed as they may contain additional problematic content."
            )
        if len(suggestions) < 3:
            suggestions = [
                "Thoroughly revise sections containing problematic language or stereotypes",
                "Replace generalizations about groups with specific, fact-based statements",
                "Consider consulting cultural sensitivity experts for content revision",
                "Add substantial context and multiple perspectives to balance the narrative",
                "Remove or completely rewrite passages containing xenophobic language",
            ]

    elif level_lower == "max":
        report += (
            f"\n\nMaximum Toxicity Alert: {media_label.capitalize()} with maximum toxicity requires immediate "
            "and comprehensive revision. The analysis identifies serious ethical concerns that may cause harm "
            "if published or distributed in its current form."
        )
        if content_type == "video":
            report += (
                "\n\nNote: This analysis focuses on the audio transcript. Visual elements may contain additional "
                "problematic content requiring separate evaluation and revision."
            )
        if len(suggestions) < 3:
            suggestions = [
                "Consider completely rewriting or re-producing the content with ethical reporting guidelines",
                "Consult with subject matter experts on cultural sensitivity and accurate representation",
                "Implement a comprehensive editorial review process before publication",
                "Ensure claims are supported by credible, well-researched sources",
                "Address the xenophobic elements identified in the analysis as a priority",
            ]

    wordcloud_image = generate_wordcloud_base64(xenophobic_words) if xenophobic_words else None

    result_dict["suggestions"] = suggestions
    result_dict["report"] = report
    result_dict["xenophobic_words"] = xenophobic_words
    result_dict["wordcloud_image"] = wordcloud_image
    return result_dict


def analyze_content_with_mock(content, content_type="text", content_truncated=False):
    """
    Intelligent multilingual Mock / Demo analyzer for when no OpenAI API key is configured
    or when the user selects Demo Mode. Inspects actual text for severe, high, and mild
    xenophobic/biased patterns across English, Chinese, and common international terms.
    """
    text_lower = content.lower()

    severe_terms = [
        "invasion", "invaders", "parasites", "savage", "savages", "vermin", "subhuman",
        "eradicate", "infestation", "illegals", "criminal aliens", "disease carriers",
        "野蛮人", "寄生虫", "入侵者", "劣等", "毒瘤", "根除", "驱逐所有",
    ]
    high_terms = [
        "primitive", "backward", "lazy", "superstition", "erase their identity",
        "wipe out", "inferior culture", "burden on society", "refuse to civilize",
        "barbaric", "indian problem", "kill the indian", "threat to our culture",
        "落后", "原始习俗", "懒惰", "迷信", "抹去", "野蛮", "社会负担", "同化", "印度问题",
    ]
    mild_terms = [
        "flood of migrants", "swarm", "wave of illegals", "outsiders", "pouring in",
        "overwhelming", "drain on resources", "illegal immigrants", "tide of refugees",
        "涌入", "外来者", "泛滥", "失控", "冲击社区",
    ]

    matched_severe = [t for t in severe_terms if t.lower() in text_lower]
    matched_high = [t for t in high_terms if t.lower() in text_lower]
    matched_mild = [t for t in mild_terms if t.lower() in text_lower]

    all_matched = []
    for term in matched_severe + matched_high + matched_mild:
        if term not in all_matched:
            all_matched.append(term)

    if len(matched_severe) >= 2 or (matched_severe and matched_high):
        toxicity_level = "Max"
    elif matched_severe or len(matched_high) >= 2:
        toxicity_level = "High"
    elif matched_high or matched_mild:
        toxicity_level = "Mild"
    else:
        toxicity_level = "None"

    has_cjk = bool(re.search(r'[\u4e00-\u9fff]', content))
    preview = content.strip()[:180].replace("\n", " ")
    if len(content.strip()) > 180:
        preview += "..."

    if toxicity_level == "None":
        if has_cjk:
            summary = f"该内容围绕相关议题进行了客观、平衡的阐述（摘要：{preview}）。"
            report = (
                f"内容摘要：{summary}\n\n"
                "伦理评估结论：经 VeriMedia 演示引擎检测，当前内容未发现排外主义、非人化修辞或明显群体偏见。"
                "文本采用了中立、尊重事实的叙述方式，符合联合国及相关新闻机构关于难民、移民与弱势群体报道的伦理准则。"
            )
            suggestions = [
                "继续保持客观、包容与尊重事实的报道语言",
                "在引用数据时持续注明权威来源（如 UNHCR 或官方统计）",
                "保持多方视角的平衡呈现，增强受众理解",
            ]
        else:
            summary = f"The submitted {content_type} discusses its subject using neutral, fact-oriented language."
            report = (
                f"Content Summary: {summary}\n\n"
                "Ethical Assessment: No dehumanizing metaphors, xenophobic slurs, or sweeping generalizations "
                "were detected in the submitted material. The framing respects human dignity and aligns with "
                "ethical journalism standards for reporting on displaced and minority populations."
            )
            suggestions = [
                "Continue using people-first, inclusive terminology",
                "Maintain balanced sourcing and verifiable statistical context",
                "Keep centering human dignity and factual accuracy in reporting",
            ]
    else:
        flagged_display = ", ".join(f'"{w}"' for w in all_matched[:8])
        if has_cjk:
            report = (
                f"内容摘要：提交的文本内容包含需要重点审视的群体定性与偏见表述（检测到关键词：{flagged_display}）。\n\n"
                f"详细分析：该内容被评估为 {toxicity_level} 级别风险。文中出现了将特定群体贴上刻板标签或将文化差异描述为“落后/威胁”的修辞。"
                "此类表述容易强化公众对难民、移民或原住民群体的负面偏见，忽视了历史、经济与人道主义背景。\n\n"
                "改进方向：建议将带有贬义或非人化色彩的词汇替换为中性、准确的描述性语言，补充受影响群体的真实声音与多维度背景资料，"
                "避免将复杂的社会议题简化为单一群体的责任。"
            )
            suggestions = [
                f"移除或重写带有偏见色彩的词汇（如：{flagged_display}），改用客观中立的表述",
                "避免对特定文化或群体作出“落后”、“负担”等笼统定性，补充具体历史与社会背景",
                "引入受影响群体的直接引语与多元视角，确保报道平衡公正",
                "核查文中的因果论断是否有权威研究或事实数据支撑",
            ]
        else:
            report = (
                f"Content Summary: The analyzed {content_type} contains framing and terminology that raise "
                f"ethical reporting concerns, specifically involving terms such as {flagged_display}.\n\n"
                f"Detailed Ethical Evaluation: The content has been classified at the '{toxicity_level}' toxicity level. "
                "The presence of catastrophe/threat metaphors or cultural generalizations risks dehumanizing displaced "
                "populations and reinforcing xenophobic stereotypes. Ethical journalism standards (including UNICC and "
                "UNHCR media guidelines) recommend avoiding language that portrays human movement as a natural disaster "
                "or security invasion.\n\n"
                "Editorial Guidance: Revise the flagged phrases to use precise, people-first descriptions, provide "
                "verifiable policy and humanitarian context, and ensure affected communities are represented fairly."
            )
            suggestions = [
                f"Replace flagged terms ({flagged_display}) with neutral, people-first language",
                "Avoid natural-disaster or military metaphors ('flood', 'invasion', 'swarm') when describing human migration",
                "Include perspectives from affected individuals, humanitarian experts, and verifiable data sources",
                "Distinguish clearly between factual reporting and editorial opinion",
            ]

    if content_truncated:
        report += (
            "\n\nNote: The analyzed content was truncated due to length limitations. "
            "The analysis is based on the first portion of the document."
        )

    result = {
        "toxicity_level": toxicity_level,
        "summary": summary if toxicity_level == "None" else "",
        "suggestions": suggestions,
        "report": report,
        "xenophobic_words": all_matched,
        "is_demo": True,
    }
    return _apply_level_context(result, content_type=content_type)


def analyze_content_with_llm(
    content,
    content_type="text",
    content_truncated=False,
    api_key=None,
    base_url=None,
    force_demo=False,
):
    """
    Core analysis pipeline using OpenAI Structured JSON output (`response_format={"type": "json_object"}`),
    with automatic fallback to Mock/Demo mode when no API key is configured.
    """
    config = resolve_api_config(custom_api_key=api_key, custom_base_url=base_url, force_demo=force_demo)
    if config["is_demo"]:
        return analyze_content_with_mock(content, content_type=content_type, content_truncated=content_truncated)

    client = create_openai_client(config["api_key"], config["base_url"])
    model_name = os.environ.get("OPENAI_MODEL", "gpt-4o").strip() or "gpt-4o"

    # Optional fine-tuned model pre-classification
    ft_toxicity_level = determine_toxicity_level_with_ft(client, content, content_type=content_type)

    system_prompt = (
        "You are an expert in analyzing media content for ethical reporting on topics related to refugees, "
        "migrants, indigenous peoples, and other forcibly displaced or marginalized populations. "
        "Your task is to analyze the provided content for xenophobic language, misinformation, stereotypes, "
        "and harmful framing.\n\n"
        "Respond strictly with a valid JSON object matching this exact schema:\n"
        "{\n"
        '  "toxicity_level": "None" | "Mild" | "High" | "Max",\n'
        '  "summary": "A brief 1-2 sentence neutral summary of the content",\n'
        '  "suggestions": ["Actionable suggestion 1", "Actionable suggestion 2", "Actionable suggestion 3"],\n'
        '  "report": "A comprehensive, multi-paragraph plain-text ethical analysis report (no markdown bold/headers)",\n'
        '  "xenophobic_words": ["problematic word or phrase 1", "problematic word or phrase 2"]\n'
        "}\n"
        "Write the summary, suggestions, and report in the same language as the input content (default to English). "
        "If toxicity_level is 'None', set xenophobic_words to an empty list []."
    )

    user_prompt = (
        f"Analyze the following {content_type} content for xenophobic language, misinformation, and harmful content.\n"
        + (f"Pre-classified toxicity level hint: {ft_toxicity_level}\n" if ft_toxicity_level else "")
        + f"\nContent:\n{content}"
    )

    response = client.chat.completions.create(
        model=model_name,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=2000,
    )

    raw_json = response.choices[0].message.content
    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError:
        # Fallback if a custom OpenAI-compatible endpoint ignored json_object mode
        match = re.search(r'\{[\s\S]*\}', raw_json)
        parsed = json.loads(match.group(0)) if match else {"report": raw_json}

    toxicity_level = ft_toxicity_level or normalize_toxicity_level(parsed.get("toxicity_level"))
    report = str(parsed.get("report", "")).strip()
    if content_truncated:
        report += (
            "\n\nNote: The analyzed content was truncated due to length limitations. "
            "The analysis is based on the first portion of the document."
        )

    result = {
        "toxicity_level": toxicity_level,
        "summary": parsed.get("summary", ""),
        "suggestions": parsed.get("suggestions", []),
        "report": report,
        "xenophobic_words": parsed.get("xenophobic_words", []),
        "is_demo": False,
    }
    return _apply_level_context(result, content_type=content_type)


def analyze_text_input(text_content, api_key=None, base_url=None, force_demo=False):
    """Analyze raw text directly (used by both file extraction and direct text paste)."""
    if not text_content or not text_content.strip():
        return {
            "toxicity_level": "Error",
            "suggestions": ["No text content provided"],
            "report": "Error: Please provide non-empty text content to analyze.",
        }

    if len(text_content) > MAX_TEXT_CHARS:
        analyzed_content = text_content[:MAX_TEXT_CHARS]
        content_truncated = True
    else:
        analyzed_content = text_content
        content_truncated = False

    return analyze_content_with_llm(
        analyzed_content,
        content_type="text",
        content_truncated=content_truncated,
        api_key=api_key,
        base_url=base_url,
        force_demo=force_demo,
    )


def analyze_text_file(file_path, api_key=None, base_url=None, force_demo=False):
    """Extract text from a file (.txt, .pdf, .docx) and analyze it."""
    try:
        content, error_msg = extract_text_from_file(file_path)
        if error_msg:
            return {
                "toxicity_level": "Error",
                "suggestions": ["Text extraction failed"],
                "report": error_msg,
            }
        return analyze_text_input(
            content,
            api_key=api_key,
            base_url=base_url,
            force_demo=force_demo,
        )
    except Exception as e:
        return {
            "toxicity_level": "Error",
            "suggestions": ["An error occurred during text analysis"],
            "report": f"Error: {str(e)}",
        }


def transcribe_audio_file(file_path, api_key=None, base_url=None, force_demo=False):
    """
    Transcribe an audio file using OpenAI Whisper API, or return a realistic demo transcript
    when running in Demo Mode.
    """
    config = resolve_api_config(custom_api_key=api_key, custom_base_url=base_url, force_demo=force_demo)
    if config["is_demo"]:
        filename = os.path.basename(file_path)
        return (
            f"[Demo Mode Transcript for {filename}] Good evening. Tonight our broadcast examines the arrival "
            "of displaced families at the regional reception center. While some commentators have described the "
            "situation as an overwhelming flood of outsiders straining municipal budgets, humanitarian coordinators "
            "emphasize that structured resettlement and community dialogue are addressing local needs effectively.",
            None,
        )

    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if os.path.getsize(file_path) > MAX_AUDIO_SIZE:
        return (
            None,
            f"Error: The audio file is {file_size_mb:.1f}MB, which exceeds the 25MB limit. "
            "Please compress or shorten your audio file.",
        )

    client = create_openai_client(config["api_key"], config["base_url"])
    file_extension = file_path.rsplit(".", 1)[1].lower() if "." in file_path else ""

    temp_converted_path = None
    try:
        try:
            with open(file_path, "rb") as audio_file:
                transcript_text = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text",
                )
        except Exception as direct_error:
            if file_extension in ("m4a", "ogg", "oga", "flac"):
                temp_converted_path = convert_audio_to_wav(file_path)
                with open(temp_converted_path, "rb") as audio_file:
                    transcript_text = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        response_format="text",
                    )
            else:
                return None, f"Error: Could not transcribe the audio file. {str(direct_error)}"

        if not transcript_text or not str(transcript_text).strip():
            return None, "Error: The audio file could not be transcribed. Please ensure it contains clear speech."

        return str(transcript_text).strip(), None
    finally:
        if temp_converted_path and os.path.exists(temp_converted_path):
            try:
                os.unlink(temp_converted_path)
            except OSError:
                pass


def analyze_audio_file(file_path, content_type="audio", api_key=None, base_url=None, force_demo=False):
    """Transcribe and analyze an audio file."""
    try:
        transcribed_text, error_msg = transcribe_audio_file(
            file_path, api_key=api_key, base_url=base_url, force_demo=force_demo
        )
        if error_msg:
            return {
                "toxicity_level": "Error",
                "suggestions": ["Audio transcription failed"],
                "report": error_msg,
            }

        result = analyze_content_with_llm(
            transcribed_text,
            content_type=content_type,
            content_truncated=False,
            api_key=api_key,
            base_url=base_url,
            force_demo=force_demo,
        )
        result["transcription"] = transcribed_text
        return result
    except Exception as e:
        return {
            "toxicity_level": "Error",
            "suggestions": ["An error occurred during audio analysis"],
            "report": f"Error: {str(e)}",
        }


def analyze_video_file(file_path, api_key=None, base_url=None, force_demo=False):
    """Extract audio from a video file, transcribe it, and run video-aware ethical analysis."""
    config = resolve_api_config(custom_api_key=api_key, custom_base_url=base_url, force_demo=force_demo)
    audio_path = None
    try:
        # Even in Demo Mode, if the video has an audio track we try extracting it; if it's a dummy/silent video in demo mode, we gracefully fallback
        try:
            audio_path = convert_video_to_audio(file_path)
        except Exception as extract_err:
            if config["is_demo"]:
                audio_path = file_path
            else:
                raise extract_err

        return analyze_audio_file(
            audio_path,
            content_type="video",
            api_key=api_key,
            base_url=base_url,
            force_demo=force_demo,
        )
    finally:
        if audio_path and audio_path != file_path and os.path.exists(audio_path):
            try:
                os.unlink(audio_path)
            except OSError:
                pass
