import trafilatura


def scrape_page(url):
    try:
        downloaded = trafilatura.fetch_url(url)

        if not downloaded:
            return ""

        content = trafilatura.extract(downloaded)

        return content or ""

    except Exception:
        return ""