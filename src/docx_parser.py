"""
Parser for DOCX files to extract text, titles, and paragraphs.
Converts document structure to a simple JSON format.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from docx import Document
from docx.shared import Pt
from docx.enum.style import WD_STYLE_TYPE


def parse_docx(docx_path: str) -> List[Dict[str, Any]]:
    """
    Parse a DOCX file and extract structured content.
    
    Args:
        docx_path: Path to the DOCX file
        
    Returns:
        List of dictionaries with keys: text, type, title_level, metadata
    """
    doc = Document(docx_path)
    elements = []
    
    for paragraph in doc.paragraphs:
        # Skip empty paragraphs
        if not paragraph.text.strip():
            continue
        
        # Detect title levels based on paragraph style
        title_level = None
        element_type = "paragraph"
        
        # Check if paragraph is a heading
        if paragraph.style.name.startswith("Heading"):
            element_type = "title"
            # Extract heading level (Heading 1, Heading 2, Heading 3)
            try:
                level_str = paragraph.style.name.split()[-1]
                title_level = int(level_str)
                if title_level > 3:
                    title_level = None
                    element_type = "paragraph"
            except (ValueError, IndexError):
                # If we can't parse the level, check style hierarchy
                if "Heading 1" in paragraph.style.name or paragraph.style.name == "Title":
                    title_level = 1
                elif "Heading 2" in paragraph.style.name:
                    title_level = 2
                elif "Heading 3" in paragraph.style.name:
                    title_level = 3
                else:
                    element_type = "paragraph"
        
        # Check paragraph formatting for title detection (fallback)
        if element_type == "paragraph":
            # Check if text is bold and short (likely a title)
            runs = paragraph.runs
            if runs and all(run.bold for run in runs if run.text.strip()):
                word_count = len(paragraph.text.split())
                if word_count <= 10:  # Short bold text might be a title
                    element_type = "title"
                    title_level = 1  # Default to level 1 if uncertain
        
        element = {
            "text": paragraph.text.strip(),
            "type": element_type,
            "title_level": title_level,
            "metadata": {
                "style": paragraph.style.name,
                "word_count": len(paragraph.text.split())
            }
        }
        
        elements.append(element)
    
    return elements


def save_to_json(elements: List[Dict[str, Any]], output_path: str) -> None:
    """
    Save parsed elements to a JSON file.
    
    Args:
        elements: List of parsed elements
        output_path: Path to save the JSON file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(elements, f, ensure_ascii=False, indent=2)


def load_from_json(json_path: str) -> List[Dict[str, Any]]:
    """
    Load parsed elements from a JSON file.
    
    Args:
        json_path: Path to the JSON file
        
    Returns:
        List of parsed elements
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

