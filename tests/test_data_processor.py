#!/usr/bin/env python
"""
Tests for the data processor module
"""

import os
import tempfile
import pytest
from readinglens.data_processor import MarkdownProcessor


@pytest.fixture
def sample_markdown_file():
    """Create a sample markdown file for testing"""
    content = """# Article Title: Test Article
    
This is some regular content that wasn't highlighted.

## First Highlight
This is a section that I found interesting and highlighted.

More regular content here...

## Second Highlight
Another section that I found valuable.
"""

    with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False) as f:
        f.write(content)
        file_path = f.name

    yield file_path

    # Cleanup
    os.unlink(file_path)


def test_parse_markdown_file(sample_markdown_file):
    """Test parsing a markdown file"""
    processor = MarkdownProcessor()
    article = processor.parse_markdown_file(sample_markdown_file)

    # Check basic fields
    assert article["title"] == "Article Title: Test Article"
    assert "file_path" in article

    # Check highlights
    assert len(article["highlights"]) == 2
    assert article["highlights"][0]["heading"] == "First Highlight"
    assert "This is a section that I found interesting" in article["highlights"][0]["content"]

    assert article["highlights"][1]["heading"] == "Second Highlight"
    assert "Another section that I found valuable" in article["highlights"][1]["content"]

    # Check full content
    assert "regular content" in article["full_content"]


def test_chunk_text():
    """Test chunking long text"""
    processor = MarkdownProcessor(max_length=32)  # Small max_length for testing

    # Generate a long text
    long_text = "This is a very long text " * 10

    # Chunk the text
    chunks = processor._chunk_text(long_text)

    # Should create multiple chunks
    assert len(chunks) > 1

    # Empty text should return empty list
    assert processor._chunk_text("") == []
