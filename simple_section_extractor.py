import fitz  # PyMuPDF
import re
from typing import Dict, List

class SimpleSectionExtractor:
    """Simple extractor for standard academic paper sections"""
    
    def __init__(self):
        # Standard sections in academic papers (in typical order)
        self.standard_sections = [
            'Abstract',
            'Introduction', 
            'Background',
            'Related Work',
            'Literature Review',
            'Methodology',
            'Methods',
            'Approach',
            'Materials and Methods',
            'Experimental Setup',
            'Experiments',
            'Results',
            'Results and Discussion',
            'Discussion',
            'Analysis',
            'Evaluation',
            'Conclusion',
            'Conclusions',
            'Future Work',
            'Acknowledgments',
            'Acknowledgements',
            'References'
        ]
    
    def extract_from_pdf(self, pdf_file) -> str:
        """Extract plain text from PDF"""
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    
    def extract_sections(self, pdf_file) -> Dict[str, str]:
        """
        Extract standard academic sections from PDF
        Returns: Dict with section names as keys and content as values
        """
        # Get full text
        text = self.extract_from_pdf(pdf_file)
        
        # Start with full paper
        sections = {"Full Paper": text}
        
        # Find all section positions
        section_positions = []
        text_lower = text.lower()
        
        for section_name in self.standard_sections:
            # Search for this section in various formats
            patterns = [
                # "1. Introduction" or "1 Introduction"
                rf'\n\s*\d+\.?\s+{re.escape(section_name)}\s*\n',
                # "Introduction" at start of line
                rf'\n\s*{re.escape(section_name)}\s*\n',
                # "INTRODUCTION" (all caps)
                rf'\n\s*{re.escape(section_name.upper())}\s*\n',
            ]
            
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    section_positions.append({
                        'name': section_name,
                        'start': match.start(),
                        'match_text': match.group()
                    })
                    break  # Found one, move to next section
        
        # Sort by position
        section_positions.sort(key=lambda x: x['start'])
        
        # Extract content between sections
        for i, section in enumerate(section_positions):
            start = section['start']
            
            # Find end position (next section or end of text)
            if i + 1 < len(section_positions):
                end = section_positions[i + 1]['start']
            else:
                end = len(text)
            
            # Extract content
            content = text[start:end].strip()
            
            # Remove the section header line from content
            content = re.sub(r'^\s*\d*\.?\s*' + re.escape(section['name']) + r'\s*\n', '', content, flags=re.IGNORECASE)
            content = re.sub(r'^\s*' + re.escape(section['name'].upper()) + r'\s*\n', '', content, flags=re.IGNORECASE)
            
            # Only add if has substantial content
            if len(content.strip()) > 100:
                sections[section['name']] = content.strip()
        
        return sections
    
    def get_available_sections(self, sections: Dict[str, str]) -> List[str]:
        """Get list of sections that were found (excluding Full Paper)"""
        return [s for s in sections.keys() if s != "Full Paper"]