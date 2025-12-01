"""
Parser for DOCX files to extract text, titles, and paragraphs.
Converts document structure to a simple JSON format.
"""

import json
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
from docx import Document
from docx.shared import Pt
from docx.enum.style import WD_STYLE_TYPE


def parse_docx(docx_path: str) -> List[Dict[str, Any]]:
    """
    Parse a DOCX file and extract structured content.
    Generates UUIDs for titles and tracks hierarchical structure.
    
    Args:
        docx_path: Path to the DOCX file
        
    Returns:
        List of dictionaries with keys: text, type, title_level, title_id, metadata
    """
    doc = Document(docx_path)
    elements = []
    
    # Track current hierarchical IDs
    current_h1_id = None
    current_h2_id = None
    current_h3_id = None
    
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
        
        # Generate UUID for titles and update hierarchical structure
        title_id = None
        h1_id_for_element = current_h1_id
        h2_id_for_element = current_h2_id
        h3_id_for_element = current_h3_id
        
        if element_type == "title" and title_level is not None:
            title_id = str(uuid.uuid4())
            
            # Update hierarchical IDs based on title level
            # The title itself gets its ID in the appropriate hierarchical field
            if title_level == 1:
                current_h1_id = title_id
                current_h2_id = None  # Reset lower levels
                current_h3_id = None
                h1_id_for_element = title_id  # H1 has its own ID in h1_id
                h2_id_for_element = None
                h3_id_for_element = None
            elif title_level == 2:
                current_h2_id = title_id
                current_h3_id = None  # Reset lower levels
                # H2 keeps parent H1 and has its own ID in h2_id
                h2_id_for_element = title_id
                h3_id_for_element = None
            elif title_level == 3:
                current_h3_id = title_id
                # H3 keeps parent H1 and H2, and has its own ID in h3_id
                h3_id_for_element = title_id
        
        element = {
            "text": paragraph.text.strip(),
            "type": element_type,
            "title_level": title_level,
            "title_id": title_id,  # UUID for this title (None for paragraphs)
            "h1_id": h1_id_for_element,  # H1 ID in hierarchy (can be None)
            "h2_id": h2_id_for_element,  # H2 ID in hierarchy (can be None)
            "h3_id": h3_id_for_element,   # H3 ID in hierarchy (can be None)
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

