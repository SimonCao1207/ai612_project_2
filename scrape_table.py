import json

import requests
from bs4 import BeautifulSoup


def parse_omr_table(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    content = soup.find("main")

    table_info = {"description": "", "table columns": {}, "links to": {}}

    # Extract table description
    description_paragraphs = content.find_all("p")
    if description_paragraphs:
        table_info["description"] = " ".join(
            p.get_text(strip=True) for p in description_paragraphs
        )

    # Extract table columns and their detailed descriptions
    headings = content.find_all(["h3", "h2"])
    for heading in headings:
        col_name = heading.get_text(strip=True).strip("`")
        desc = ""
        next_sibling = heading.find_next_sibling()
        while next_sibling and next_sibling.name == "p":
            desc += next_sibling.get_text().strip()
            next_sibling = next_sibling.find_next_sibling()
        if col_name:
            table_info["table columns"][col_name] = desc.strip()

    return {"omr": table_info}


def main():
    url = "https://mimic.mit.edu/docs/iv/modules/hosp/omr/"
    omr_info = parse_omr_table(url)

    # Save to JSON file
    with open("omr_table_schema.json", "w") as f:
        json.dump(omr_info, f, indent=4)

    print("OMR table schema has been saved to 'omr_table_schema.json'.")


if __name__ == "__main__":
    main()
