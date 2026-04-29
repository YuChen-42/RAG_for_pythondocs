from bs4 import BeautifulSoup


def parse_html_document(html_content: str, source_name: str) -> dict:
    soup = BeautifulSoup(html_content, "lxml")
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else source_name

    main = soup.find("div", {"role": "main"})
    if not main:
        main = soup.find("div", {"class": "body"})
    if not main:
        main = soup.body

    for tag in main.find_all(["script", "style", "nav", "header", "footer"]):
        tag.decompose()

    text = main.get_text(separator="\n", strip=True)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    text = "\n".join(lines)

    return {"source": source_name, "title": title, "text": text}
