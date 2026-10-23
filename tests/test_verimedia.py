import io
import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from app import app
from verimedia.analyzer import SAMPLE_PRESETS, analyze_content_with_llm, normalize_toxicity_level
from verimedia.pdf_generator import build_report_pdf
from verimedia.result_store import load_analysis_result, save_analysis_result
from verimedia.visualizer import generate_wordcloud_base64, get_chinese_font


class VeriMediaTestCase(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()

    def test_index_and_about_routes(self):
        resp = self.client.get('/')
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b'VeriMedia', resp.data)
        self.assertIn(b'Paste Text / Try Sample', resp.data)

        about_resp = self.client.get('/about')
        self.assertEqual(about_resp.status_code, 200)
        self.assertIn(b'About VeriMedia', about_resp.data)

    def test_chinese_font_discovery_not_dejavu(self):
        font_path = get_chinese_font()
        if font_path is not None:
            self.assertTrue(os.path.exists(font_path))
            self.assertNotIn('dejavu', font_path.lower())

        wc_b64 = generate_wordcloud_base64(['原始习俗', '刻板印象', '社会负担', '外来者'])
        self.assertIsNotNone(wc_b64)
        self.assertTrue(len(wc_b64) > 100)

    def test_direct_text_paste_and_cookie_size_limit(self):
        sample_zh = SAMPLE_PRESETS['high_zh']['content']
        resp = self.client.post(
            '/upload',
            data={
                'file_type': 'text',
                'input_mode': 'paste',
                'text_content': sample_zh,
            },
            follow_redirects=True,
        )
        self.assertEqual(resp.status_code, 200)
        html = resp.data.decode('utf-8')
        self.assertIn('Media Analysis Results', html)
        self.assertIn('High', html)
        self.assertIn('data:image/png;base64,', html)

        # Ensure Set-Cookie header is well below the 4093-byte browser limit
        cookie_headers = resp.headers.getlist('Set-Cookie')
        for cookie in cookie_headers:
            self.assertLess(
                len(cookie.encode('utf-8')),
                1024,
                f"Session cookie exceeded safe size: {len(cookie)} bytes",
            )

    def test_xss_escaping_in_results_template(self):
        malicious_text = (
            "This flood of migrants and outsiders <script>alert('XSS')</script> "
            "is overwhelming local clinics."
        )
        resp = self.client.post(
            '/upload',
            data={
                'file_type': 'text',
                'input_mode': 'paste',
                'text_content': malicious_text,
            },
            follow_redirects=True,
        )
        self.assertEqual(resp.status_code, 200)
        html = resp.data.decode('utf-8')
        self.assertNotIn("<script>alert('XSS')</script>", html)

    def test_pdf_generation_with_cjk_and_xml_special_chars(self):
        wc_b64 = generate_wordcloud_base64(['落后', '原始习俗', 'flood & <invasion>'])
        cached_result = {
            'toxicity_level': 'High',
            'report': '第一段：包含特殊字符 A < B & C > D 的中文伦理分析报告。\n\n第二段：补充说明。',
            'suggestions': [
                '建议一：避免使用 <标签> 与 & 符号导致的崩溃',
                '建议二：替换带有偏见的词汇',
                '建议三：补充多方信源与上下文',
            ],
            'xenophobic_words': ['落后', '原始习俗', 'A & B <C>'],
            'wordcloud_image': wc_b64,
            'transcription': '转录内容：测试音视频转录文本 & <特殊符号>。',
        }
        result_id = save_analysis_result(cached_result)
        loaded = load_analysis_result(result_id)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded['toxicity_level'], 'High')

        pdf_resp = self.client.post(
            '/download_report_pdf',
            data={'result_id': result_id},
        )
        self.assertEqual(pdf_resp.status_code, 200)
        self.assertEqual(pdf_resp.headers['Content-Type'], 'application/pdf')
        self.assertTrue(pdf_resp.data.startswith(b'%PDF-'))
        self.assertGreater(len(pdf_resp.data), 2000)

    def test_structured_json_llm_pipeline_mocked(self):
        fake_json_response = json.dumps({
            "toxicity_level": "Mild",
            "summary": "An article discussing border arrivals with dramatic metaphors.",
            "suggestions": [
                "Replace 'wave' metaphor with factual numbers",
                "Include humanitarian perspectives",
                "Clarify legal asylum procedures",
            ],
            "report": "Paragraph 1 of analysis.\n\nParagraph 2 of analysis.",
            "xenophobic_words": ["wave", "outsiders"],
        })

        with patch('verimedia.analyzer.create_openai_client') as mock_create_client, \
             patch('verimedia.analyzer.determine_toxicity_level_with_ft', return_value=None):
            mock_client = MagicMock()
            mock_choice = MagicMock()
            mock_choice.message.content = fake_json_response
            mock_client.chat.completions.create.return_value.choices = [mock_choice]
            mock_create_client.return_value = mock_client

            res = analyze_content_with_llm(
                "Sample border coverage text",
                content_type="text",
                api_key="sk-test-valid-key",
                force_demo=False,
            )
            self.assertFalse(res['is_demo'])
            self.assertEqual(res['toxicity_level'], 'Mild')
            self.assertEqual(res['xenophobic_words'], ['wave', 'outsiders'])
            self.assertIsNotNone(res['wordcloud_image'])

            # Verify response_format={"type": "json_object"} was used
            call_kwargs = mock_client.chat.completions.create.call_args.kwargs
            self.assertEqual(call_kwargs.get('response_format'), {"type": "json_object"})

    def test_api_settings_endpoint(self):
        resp = self.client.post(
            '/settings/api-key',
            data={
                'action': 'save',
                'api_key': 'sk-session-key-123',
                'base_url': 'https://api.openai.com/v1',
            },
            follow_redirects=True,
        )
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b'Live AI (Session Key)', resp.data)

        clear_resp = self.client.post(
            '/settings/api-key',
            data={'action': 'clear'},
            follow_redirects=True,
        )
        self.assertEqual(clear_resp.status_code, 200)
        self.assertIn(b'Demo Mode', clear_resp.data)


if __name__ == '__main__':
    unittest.main()
