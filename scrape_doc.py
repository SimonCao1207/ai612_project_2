import json

import requests
from bs4 import BeautifulSoup
from bs4.element import Tag

BASE_URL = "https://mimic.mit.edu"
MODULES_URL = f"{BASE_URL}/docs/iv/modules/"

list_of_tables = [
    "patients",
    "admissions",
    "d_icd_diagnoses",
    "d_icd_procedures",
    "d_labitems",
    "d_items",
    "diagnoses_icd",
    "procedures_icd",
    "labevents",
    "prescriptions",
    "cost",
    "chartevents",
    "inputevents",
    "outputevents",
    "microbiologyevents",
    "icustays",
    "transfers",
]

ignored_modules = ["ED", "CXR", "ECG", "Note"]


def preprocess_text(text):
    cleaned_text = (
        text.replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
    )

    return cleaned_text


def get_module_links(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    modules_info = []

    # Find all divs with class 'entry'
    entries = soup.find_all("div", class_="entry")
    for entry in entries:
        module = {}
        # Extract table name and link
        h5 = entry.find("h5")
        if h5 and h5.a:
            module_name = h5.a.get_text(strip=True)
            link = h5.a["href"]
            module["module_name"] = module_name
            module["link"] = f"https://mimic.mit.edu{link}"
        # Extract description
        p = entry.find("p")
        if p:
            module["description"] = p.get_text().strip()
        modules_info.append(module)
    return modules_info


def parse_table_page(url, table_name):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    content = soup.find("main")

    table_info = {"description": "", "columns": {}, "links to": {}}

    # Extract table linked to
    links_to_section = content.find(
        ["h2", "h3", "p"], string=lambda text: text and "links to" in text.lower()
    )

    ul = None
    if links_to_section:
        next_sibling = links_to_section.find_next_sibling()
        while next_sibling and next_sibling.name != "ul":
            next_sibling = next_sibling.find_next_sibling()
        ul = next_sibling
    if ul:
        for li in ul.find_all("li"):
            em = li.find("em")
            codes = li.find_all("code")
            if em and codes:
                other_table = em.get_text(strip=True)
                # Initialize the list if not present
                if other_table not in table_info["links to"]:
                    table_info["links to"][other_table] = []
                for code in codes:
                    table_info["links to"][other_table].append(
                        code.get_text(strip=True)
                    )
    elif table_name in list_of_tables:
        print("Links to section not found.")

    # Extract table description
    brief_summary_heading = soup.find(
        "h1", id=lambda x: x and "brief-summary" in x.lower()
    )
    if not brief_summary_heading:
        brief_summary_heading = soup.find(
            "h1", string=lambda text: text and "brief summary" in text.lower()
        )
    table_description = None
    if brief_summary_heading:
        table_description = brief_summary_heading.find_next_sibling("p")
    if table_description:
        table_info["description"] = preprocess_text(
            table_description.get_text().strip()
        )
    else:
        table_heading = soup.find("h2", id=lambda x: x and table_name in x)
        table_description = (
            table_heading.find_next_sibling("p") if table_heading else None
        )
        table_info["description"] = preprocess_text(
            table_description.get_text().strip() if table_description else ""
        )

        if not table_description:
            more_table_heading = soup.find(
                "h1", string=lambda text: text and table_name.lower() in text.lower()
            )
            more_table_description = (
                more_table_heading.find_next_sibling("p")
                if more_table_heading
                else None
            )
            table_info["description"] = (
                preprocess_text(more_table_description.get_text().strip())
                if more_table_description
                else ""
            )
        if not table_info["description"]:
            print("Table description not found.")

    # Extract table columns and their detailed descriptions
    columns_detail_section = None
    for tag in ["h2", "h1"]:
        for keyword in ["detailed description", "table columns"]:
            columns_detail_section = soup.find(
                tag, string=lambda text: text and keyword in text.lower()
            )
            if columns_detail_section:
                break
        if columns_detail_section:
            break

    if not columns_detail_section and table_name in list_of_tables:
        print("Details section not found.")
    else:
        for sibling in columns_detail_section.find_next_siblings():
            if isinstance(sibling, Tag) and sibling.name == "Feedback":
                break
            if isinstance(sibling, Tag) and sibling.name in ["h3", "h2"]:
                field_name = sibling.get_text().strip()
                field_content = []
                p = sibling.find_next_sibling()
                while isinstance(p, Tag) and p.name == "p":
                    field_content.append(p.get_text().strip())
                    p = p.find_next_sibling()
                if field_content:
                    table_info["columns"][field_name] = preprocess_text(
                        ". ".join(field_content)
                    )

    return table_info


def get_table_links(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    tables_info = []

    # Find all divs with class 'entry'
    entries = soup.find_all("div", class_="entry")
    for entry in entries:
        table = {}
        # Extract table name and link
        h5 = entry.find("h5")
        if h5 and h5.a:
            table_name = h5.a.get_text(strip=True)
            link = h5.a["href"]
            table["table_name"] = table_name
            table["link"] = f"https://mimic.mit.edu{link}"
        # Extract description
        p = entry.find("p")
        if p:
            table["description"] = p.get_text().strip()
        tables_info.append(table)

    return tables_info


def main():
    module_links = get_module_links(MODULES_URL)
    mimic_schema = {}
    num_success = 0

    for module_info in module_links:
        module_link = module_info["link"]
        module_name = module_info["module_name"]
        if module_name in ignored_modules:
            print(f"Skipping module: {module_name}")
            continue
        module_description = module_info["description"]
        mimic_schema[module_name] = {
            "description": module_description,
            "tables": {},
        }
        print(f"Processing module: {module_name}")
        table_scheme = {}
        tables_info = get_table_links(module_link)
        for table_info in tables_info:
            lst = table_info["table_name"].split(" ")
            table_name = (
                "".join(lst[:-1]).lower()
                if lst[-1] == "table"
                else table_info["table_name"].replace(" ", "").lower()
            )

            if table_name in list_of_tables:
                print(f"---Processing table: {table_name}")
                table_link = table_info["link"]
                table_scheme[table_name] = {
                    "description": preprocess_text(table_info["description"]),
                    "link": table_link,
                }
                info = parse_table_page(table_link, table_name)
                table_scheme[table_name].update(info)
                num_success += 1
        mimic_schema[module_name]["tables"].update(table_scheme)

    print(f"Number of successful processed tables: {num_success}")

    # Save to JSON file
    with open("mimic_iv_schema.json", "w") as f:
        json.dump(mimic_schema, f, indent=4)


if __name__ == "__main__":
    main()
