"""
Private Firm Data Collection — Web scraping for operational images and text.

Extension module for applying the multimodal framework to VC-backed private firms.
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class PrivateFirmScraper:
    """
    Scrape operational images and text descriptions from company websites.

    Targets product pages, "How it works", "Our technology", and press/media
    sections for high-quality operational imagery.
    """

    # Pages likely to contain operational images
    OPERATIONAL_PATHS = [
        "/product", "/products", "/platform", "/technology",
        "/how-it-works", "/solutions", "/features", "/about",
        "/our-technology", "/what-we-do",
    ]

    def __init__(
        self,
        output_dir: str | Path = "data/processed/private",
        rate_limit: float = 1.0,
        max_images_per_firm: int = 20,
    ):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
        })
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit = rate_limit
        self.max_images = max_images_per_firm

    def _throttle(self):
        time.sleep(self.rate_limit)

    def scrape_firm(self, firm_id: str, website_url: str) -> dict:
        """
        Scrape a single firm's website for operational images and text.

        Returns:
            dict with keys: firm_id, images (list of paths), text (str)
        """
        firm_dir = self.output_dir / firm_id
        firm_dir.mkdir(parents=True, exist_ok=True)

        # Normalize base URL
        parsed = urlparse(website_url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"

        all_images = []
        all_text = []

        # Scrape main page + operational sub-pages
        pages_to_scrape = [website_url]
        for path in self.OPERATIONAL_PATHS:
            pages_to_scrape.append(urljoin(base_url, path))

        visited = set()
        for page_url in pages_to_scrape:
            if page_url in visited:
                continue
            visited.add(page_url)

            self._throttle()
            try:
                resp = self.session.get(page_url, timeout=15)
                if resp.status_code != 200:
                    continue
            except requests.RequestException:
                continue

            soup = BeautifulSoup(resp.text, "lxml")

            # Extract text
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            page_text = soup.get_text(separator=" ", strip=True)
            # Keep only meaningful paragraphs
            paragraphs = [p for p in page_text.split("\n") if len(p.strip()) > 50]
            all_text.extend(paragraphs[:20])  # Cap per page

            # Extract images
            for img_tag in soup.find_all("img"):
                if len(all_images) >= self.max_images:
                    break

                src = img_tag.get("src", "") or img_tag.get("data-src", "")
                if not src:
                    continue

                # Resolve relative URLs
                img_url = urljoin(page_url, src)

                # Skip tiny tracking pixels, SVGs, and icons
                if any(skip in img_url.lower() for skip in [
                    ".svg", "favicon", "icon", "logo", "pixel", "tracking",
                    "1x1", "spacer", "blank",
                ]):
                    continue

                # Download image
                img_path = self._download_image(img_url, firm_dir, len(all_images))
                if img_path is not None:
                    all_images.append(str(img_path))

        # Save text
        text_path = firm_dir / "description.txt"
        text_path.write_text("\n".join(all_text[:50]), encoding="utf-8")

        logger.info(
            "Firm %s: scraped %d images, %d text paragraphs from %s",
            firm_id, len(all_images), len(all_text), website_url,
        )

        return {
            "firm_id": firm_id,
            "images": all_images,
            "text": "\n".join(all_text[:50]),
        }

    def _download_image(
        self, url: str, output_dir: Path, idx: int
    ) -> Path | None:
        """Download a single image, filtering by size."""
        from PIL import Image
        import io

        try:
            resp = self.session.get(url, timeout=10, stream=True)
            if resp.status_code != 200:
                return None

            content_type = resp.headers.get("Content-Type", "")
            if "image" not in content_type:
                return None

            img_data = resp.content
            img = Image.open(io.BytesIO(img_data)).convert("RGB")
            w, h = img.size

            # Skip small images
            if w < 200 or h < 200:
                return None

            # Skip extreme aspect ratios (logos)
            ar = w / h
            if ar > 5.0 or ar < 0.2:
                return None

            ext = url.split(".")[-1].split("?")[0].lower()
            if ext not in ("jpg", "jpeg", "png", "webp"):
                ext = "png"

            out_path = output_dir / f"img_{idx:04d}.{ext}"
            img.save(out_path)
            return out_path

        except Exception as e:
            logger.debug("Image download failed %s: %s", url, e)
            return None

    def scrape_batch(self, firms: list[dict]) -> list[dict]:
        """
        Scrape a batch of firms.

        Args:
            firms: List of dicts with keys: firm_id, website_url,
                   and optionally: valuation, revenue, sector.
        """
        results = []
        for i, firm in enumerate(firms):
            logger.info("[%d/%d] Scraping %s", i + 1, len(firms), firm["firm_id"])
            result = self.scrape_firm(firm["firm_id"], firm["website_url"])
            # Attach valuation labels if available
            for key in ["valuation", "revenue", "sector"]:
                if key in firm:
                    result[key] = firm[key]
            results.append(result)
        return results


def main():
    """CLI for private firm scraping."""
    import argparse
    import json

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="Scrape private firm websites")
    parser.add_argument("--firms-json", required=True,
                        help="JSON file with list of {firm_id, website_url, ...}")
    parser.add_argument("--output-dir", default="data/processed/private")
    parser.add_argument("--max-images", type=int, default=20)
    args = parser.parse_args()

    firms = json.loads(Path(args.firms_json).read_text())
    scraper = PrivateFirmScraper(
        output_dir=args.output_dir,
        max_images_per_firm=args.max_images,
    )
    scraper.scrape_batch(firms)


if __name__ == "__main__":
    main()
