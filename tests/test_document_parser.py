import pytest
from src.document_parser import parse_html_document


def test_parse_html_extracts_text():
    html = '''<html><head><title>Test Doc</title></head>
    <body><div role="main"><p>Hello world</p></div></body></html>'''
    result = parse_html_document(html, "test.html")
    assert result["title"] == "Test Doc"
    assert "Hello world" in result["text"]
    assert result["source"] == "test.html"
