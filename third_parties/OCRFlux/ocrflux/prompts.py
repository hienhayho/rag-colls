import re
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class PageResponse:
    primary_language: Optional[str]
    is_rotation_valid: bool
    rotation_correction: int
    is_table: bool
    is_diagram: bool
    natural_text: Optional[str]

    def __post_init__(self):
        # Validate rotation_correction is one of the allowed values
        if self.rotation_correction not in {0, 90, 180, 270}:
            raise ValueError("rotation_correction must be one of [0, 90, 180, 270].")

        # Type checks
        if not isinstance(self.primary_language, (str, type(None))):
            raise TypeError("primary_language must be of type Optional[str].")
        if not isinstance(self.is_rotation_valid, bool):
            raise TypeError("is_rotation_valid must be of type bool.")
        if not isinstance(self.rotation_correction, int):
            raise TypeError("rotation_correction must be of type int.")
        if not isinstance(self.is_table, bool):
            raise TypeError("is_table must be of type bool.")
        if not isinstance(self.is_diagram, bool):
            raise TypeError("is_diagram must be of type bool.")
        if not isinstance(self.natural_text, (str, type(None))):
            raise TypeError("natural_text must be of type Optional[str].")

def build_element_merge_detect_prompt(text_list_1,text_list_2) -> str:
    task = '''Below are two consecutive pages in Markdown format, where each element of them is numbered. Identify pairs of elements which should be merged across the two pages, such as text paragraphs or tables that span across the two pages. Return pairs as [(element_index_of_page1, element_index_of_page2), ...] or [] if no elements should be merged.\n'''
    task += "Previous page:\n"
    for i,text in  enumerate(text_list_1):
        task += f"{i}. {text}\n\n"
    task += "Next page:\n"
    for i,text in  enumerate(text_list_2):
        task += f"{i}. {text}\n\n"
    return task

def build_html_table_merge_prompt(table1,table2) -> str:
    return (
        f"Below are two tables in HTML format, merge them into one table in HTML format.\n"
        f"TABLE 1:\n"
        f"{table1}\n"
        f"TABLE 2:\n"
        f"{table2}\n"
    )

def build_page_to_markdown_prompt() -> str:
    return (
        f"Below is the image of one page of a document. "
        f"Just return the plain text representation of this document as if you were reading it naturally.\n"
        f"ALL tables should be presented in HTML format.\n"
        f"If there are images or figures in the page, present them as \"<Image>(left,top),(right,bottom)</Image>\", (left,top,right,bottom) are the coordinates of the top-left and bottom-right corners of the image or figure.\n"
        f"Present all titles and headings as H1 headings.\n"
        f"Do not hallucinate.\n"
    )
