import base64
import os
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm
from wordcloud import WordCloud


def get_chinese_font():
    """
    Find a system font that supports Chinese / CJK characters.
    Checks explicit OS font paths first (.ttc and .ttf) and uses
    fallback_to_default=False in matplotlib font_manager so it never
    falsely returns DejaVuSans.ttf.
    """
    candidate_paths = [
        # macOS system CJK fonts
        '/System/Library/Fonts/PingFang.ttc',
        '/System/Library/Fonts/STHeiti Medium.ttc',
        '/System/Library/Fonts/STHeiti Light.ttc',
        '/System/Library/Fonts/Hiragino Sans GB.ttc',
        '/System/Library/Fonts/Supplemental/Songti.ttc',
        '/System/Library/Fonts/Supplemental/Arial Unicode.ttf',
        '/Library/Fonts/Arial Unicode.ttf',
        # Linux CJK fonts
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
        '/usr/share/fonts/truetype/arphic/uming.ttc',
        # Windows CJK fonts
        'C:/Windows/Fonts/msyh.ttc',
        'C:/Windows/Fonts/simhei.ttf',
        'C:/Windows/Fonts/simsun.ttc',
    ]

    for path in candidate_paths:
        if os.path.exists(path):
            return path

    chinese_font_families = [
        'PingFang SC',
        'Heiti SC',
        'STHeiti',
        'Hiragino Sans GB',
        'Arial Unicode MS',
        'Microsoft YaHei',
        'WenQuanYi Micro Hei',
        'Noto Sans CJK SC',
        'Noto Sans CJK TC',
        'Noto Sans CJK JP',
        'SimHei',
        'SimSun',
        'Songti SC',
    ]

    for family in chinese_font_families:
        try:
            font_path = fm.findfont(
                fm.FontProperties(family=family),
                fallback_to_default=False
            )
            if font_path and os.path.exists(font_path) and 'dejavu' not in font_path.lower():
                return font_path
        except Exception:
            continue

    return None


def generate_wordcloud_base64(words):
    """
    Generate a PNG WordCloud encoded as a base64 string from a list of problematic words/phrases.
    Supports multilingual / CJK terms.
    """
    if not words:
        return None

    cleaned_words = [str(w).strip() for w in words if str(w).strip()]
    if not cleaned_words:
        return None

    try:
        # Weight phrases by frequency so WordCloud renders multi-word or CJK tokens cleanly
        frequencies = {}
        total = len(cleaned_words)
        for idx, word in enumerate(cleaned_words):
            weight = max(1, total - idx)
            frequencies[word] = frequencies.get(word, 0) + weight

        font_path = get_chinese_font()
        wordcloud_params = {
            'width': 800,
            'height': 400,
            'background_color': 'white',
            'colormap': 'viridis',
            'contour_width': 1,
            'contour_color': 'steelblue',
            'prefer_horizontal': 0.9,
        }
        if font_path:
            wordcloud_params['font_path'] = font_path

        wc = WordCloud(**wordcloud_params).generate_from_frequencies(frequencies)

        img_buffer = BytesIO()
        wc.to_image().save(img_buffer, format='PNG')
        img_buffer.seek(0)
        return base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Error generating word cloud: {e}")
        return None
