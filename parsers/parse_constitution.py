import argparse
import os
import re
from typing import List, Dict
import json

START_MARKER = """GOD BLESS KENYA"""

END_MARKER = """
                      FIRST SCHEDULE
                          COUNTIES
"""
ODD_HEADER = r"\[Rev\. 2022\] *Constitution of Kenya *[0-9]*"
EVEN_HEADER = r"[0-9]* *Constitution of Kenya *\[Rev\. 2022\]"
CHAPTER_MARKER = re.compile(r'^\s*CHAPTER\s+[A-Z]+\s*\n(?P<heading>[A-Z ]+)$',re.MULTILINE)

def remove_start_end_boilerplate(text: str):
    text = text.split(START_MARKER)[1]
    text = text.split(END_MARKER)[0]
    text = re.sub(ODD_HEADER, '', text)
    text = re.sub(EVEN_HEADER, '', text)
    return text

def split_into_chapters(text) -> List[Dict[str,str]]:
    chapters = []
    matches = list(CHAPTER_MARKER.finditer(text))
    for i, match in enumerate(matches):
        heading = match.group('heading').strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chapter_text = f"""{text[start:end].strip()}"""

        chapters.append({
            "heading": heading,
            "text": chapter_text
        })

    return chapters

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="parse_constitution",
        description="converts txt version of kenyan constitution to list of dicts in a json file",
        epilog="Please ensure you extract the pdf using a tool like pdftotext"
    )
    parser.add_argument("--textfile")
    parser.add_argument("--jsonfile")
    args = parser.parse_args()

    if not os.path.exists(args.textfile):
        print(f"Textfile {args.textfile} not found")
        raise

    with open(args.textfile, "r", encoding="utf-8") as f:
        text = f.read()

    text = remove_start_end_boilerplate(text=text)
    chapters = split_into_chapters(text=text)
    with open(args.jsonfile, "w") as f:
        f.write(json.dumps(chapters))