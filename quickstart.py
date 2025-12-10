#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import math
import base64
from datetime import datetime
from urllib.parse import urlparse
from typing import Dict, Any, Tuple, List, Optional

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from openai import OpenAI

# -------------- Config / Setup --------------

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_CSE_KEY = os.getenv("GOOGLE_CSE_API_KEY")
GOOGLE_CSE_CX = os.getenv("GOOGLE_CSE_CX")
DEDUP_STRATEGY = os.getenv("DEDUP_STRATEGY", "coarse").lower()  # coarse | full | none
MAX_EMAILS = int(os.getenv("MAX_EMAILS", "2000"))  # total messages to fetch across pages; <=0 means no cap
# Default to loose subject filtering; set STRICT_ORDER_SUBJECT=1 to re-enable tight subject filters.
STRICT_ORDER_SUBJECT = os.getenv("STRICT_ORDER_SUBJECT", "0") not in ("0", "false", "False")

# Heuristic to strip trailing color suffixes from product names (e.g., " - White / Grey")
COLOR_SUFFIX_RE = re.compile(r"\s*[-–—:]\s*([a-z0-9 ,./]+)$", re.IGNORECASE)

# Kith color code helpers (from quickstart copy)
KITH_COLOR_CODES = {
    "101": "Black",
    "102": "White",
    "103": "Red",
    "104": "Veil",
    "105": "Sandrift",
    "106": "Navy",
    "107": "Slate",
    "108": "Woodrose",
}
KITH_COLOR_TO_CODE = {v: k for k, v in KITH_COLOR_CODES.items()}
KITH_CODE_RE = re.compile(r"\bkh[a-z]\d{6}\b", re.I)
HANDLE_FROM_URL_RE = re.compile(r"/products/([a-z0-9\-]+)", re.I)

def normalize_product_name(name: str) -> str:
    n = re.sub(r"\s+", " ", (name or "")).strip().lower()
    if not n:
        return ""
    m = COLOR_SUFFIX_RE.search(n)
    if m:
        suffix = m.group(1)
        # If suffix looks like a color block (short, includes slashes/numbers/common color tokens), drop it
        if len(suffix.split()) <= 4 or "/" in suffix or any(tok in suffix for tok in ["white", "black", "grey", "gray", "navy", "red", "blue", "green", "beige", "tan", "brown"]):
            n = n[: m.start()].strip()
    return n


def build_image_query(item: Dict[str, Any], site_domain: Optional[str] = None) -> str:
    parts = [
        item.get("brand") or "",
        item.get("product_name") or "",
        item.get("color") or "",
    ]
    q = " ".join(p for p in parts if isinstance(p, str) and p.strip()).strip()
    q = re.sub(r"\s+", " ", q)
    if site_domain:
        q = f"{q} site:{site_domain}"
    return q


def build_abercrombie_image_query(item: Dict[str, Any]) -> str:
    """
    Abercrombie-specific image query: prefer a clean, broad search without site filter.
    """
    brand = "abercrombie"
    product = item.get("product_name") or ""
    color = item.get("color") or ""
    size = item.get("size") or ""
    parts = [brand, product, color, size]
    q = " ".join(p for p in parts if isinstance(p, str) and p.strip())
    # Strip punctuation like '/' or '&' that can hurt recall; avoid style_code to keep query clean
    q = re.sub(r"[^A-Za-z0-9\s]", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q


def build_suitsupply_image_query(item: Dict[str, Any]) -> str:
    """
    SuitSupply-specific query: brand + product + color/size for better recall.
    """
    parts = [
        "suitsupply",
        item.get("product_name") or "",
        item.get("color") or "",
        item.get("size") or "",
    ]
    q = " ".join(p for p in parts if isinstance(p, str) and p.strip())
    q = re.sub(r"\s+", " ", q).strip()
    return q


def build_abercrombie_image_query(item: Dict[str, Any]) -> str:
    """
    Specific image query for Abercrombie: include brand, product, color, and style_code,
    and bias toward the abercrombie.com domain.
    """
    return build_image_query(item, site_domain="abercrombie.com")

if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY")
if not GOOGLE_CSE_KEY or not GOOGLE_CSE_CX:
    print("⚠️  Missing Google CSE key or CX – image search will not work.")

client = OpenAI(api_key=OPENAI_API_KEY)

# If modifying these scopes, delete token.json.
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

# -------------- Brand Config --------------

"""
BRAND_CONFIG is the single source of truth for:
- Which senders we support
- Whether they are Shopify-powered vs non-Shopify
- How to search their site for PDP URLs
"""

BRAND_CONFIG: Dict[str, Dict[str, Any]] = {
    # Kith (online orders + in-store receipts)
    "kith": {
        "senders": ["no-reply@kithnyc.com"],
        "type": "shopify",  # online is shopify-powered
        "search_domain": "kith.com",
    },
    # Aime Leon Dore
    "aime_leon_dore": {
        "senders": ["orders@aimeleondore.com"],
        "type": "shopify",
        "search_domain": "aimeleondore.com",
    },
    # Aplasticplant
    "aplasticplant": {
        "senders": ["info@aplasticplant.com"],
        "type": "shopify",
        "search_domain": "aplasticplant.com",
    },
    # Shopbop – acts as a department store
    "shopbop": {
        "senders": ["service@em.shopbop.com"],
        "type": "non_shopify",  # PDPs are not first-party Shopify even though backend may be
        "search_domain": "shopbop.com",
    },
    # SuitSupply
    "suitsupply": {
        "senders": ["message@service.suitsupply.com"],
        "type": "non_shopify",
        "search_domain": "suitsupply.com",
    },
    # Banana Republic / Factory
    "banana_republic": {
        "senders": [
            "orders@email.bananarepublic.com",
            "orders@email.bananarepublicfactory.com",
        ],
        "type": "non_shopify",
        "search_domain": "bananarepublic.gap.com",
    },
    "banana_republic_factory": {
        "senders": [
            "orders@email.bananarepublicfactory.com",
        ],
        "type": "non_shopify",
        "search_domain": "bananarepublicfactory.gapfactory.com",
    },
    # Abercrombie
    "abercrombie": {
        "senders": ["abercrombie@tm.abercrombie.com"],
        "type": "non_shopify",
        "search_domain": "abercrombie.com",
    },
    # Net-A-Porter
    "netaporter": {
        "senders": ["customercare@emails.net-a-porter.com"],
        "type": "non_shopify",
        "search_domain": "net-a-porter.com",
    },
    # WeWoreWhat (Shopify-powered)
    "weworewhat": {
        # Include common Narvar + direct store sender variations
        "senders": [
            "weworewhat@weworewhat.narvar",
            "weworewhat@weworewhat.narvar.com",
            "orders@weworewhat.com",
            "store+2560491566@t.shopifyemail.com",
            "no-reply@weworewhat.com",
        ],
        "type": "shopify",
        "search_domain": "weworewhat.com",
    },
    # Everlane (Shopify-powered)
    "everlane": {
        "senders": ["support@everlane.com"],
        "type": "shopify",
        "search_domain": "everlane.com",
    },
}

# Flatten sender → brand_key mapping for Gmail query
SUPPORTED_SENDERS: Dict[str, str] = {}
for brand_key, cfg in BRAND_CONFIG.items():
    for sender in cfg["senders"]:
        SUPPORTED_SENDERS[sender] = brand_key

# Default headers for Shopify MCP calls
DEFAULT_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "User-Agent": "ClosetAI/1.0",
}

# ---------- Small utils borrowed from quickstart copy ----------

def _simplify_query(q: str) -> str:
    """Remove slashes/extra fluff; cap to ~8 words; collapse spaces."""
    q = q.replace("/", " ")
    q = re.sub(r"\s+", " ", q).strip()
    parts = q.split()
    if len(parts) > 8:
        q = " ".join(parts[:8])
    return q


def _coerce_json(val):
    """Turn a stringified JSON blob into dict/list when possible."""
    if isinstance(val, (dict, list)):
        return val
    if isinstance(val, str):
        try:
            return json.loads(val)
        except Exception:
            try:
                return json.loads(bytes(val, "utf-8").decode("unicode_escape"))
            except Exception:
                return None
    return None


def _norm(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def parse_money(val):
    """Return (amount_float, currency) from strings or dicts with amount/currency."""
    if isinstance(val, dict):
        amt = val.get("amount") or val.get("price") or val.get("value")
        cur = val.get("currency") or val.get("currencyCode") or val.get("currency_code")
        try:
            return (float(amt), cur)
        except Exception:
            return (None, cur)
    if isinstance(val, (int, float)):
        return (float(val), None)
    if isinstance(val, str):
        cur = None
        m = re.search(r"([A-Za-z]{3})", val)
        if m:
            cur = m.group(1).upper()
        s = re.sub(r"[^\d\.\-]", "", val)
        try:
            return (float(s), cur)
        except Exception:
            return (None, cur)
    return (None, None)


def approx_equal(a, b, tol=1.5):
    if a is None or b is None:
        return False
    return math.fabs(a - b) <= tol


# -------------- Gmail Helpers --------------

def gmail_authenticate() -> Any:
    """Authenticate to Gmail and return a service client."""
    creds: Optional[Credentials] = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    # If there are no valid credentials, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists("credentials.json"):
                raise RuntimeError(
                    "credentials.json not found. Download it from Gmail API console."
                )
            flow = InstalledAppFlow.from_client_secrets_file(
                "credentials.json", SCOPES
            )
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open("token.json", "w") as token:
            token.write(creds.to_json())
    service = build("gmail", "v1", credentials=creds)
    return service


def get_message_body(message: Dict[str, Any]) -> str:
    """
    Extract the text/plain or text/html body from a Gmail message.
    Walks multipart trees to avoid missing HTML nested inside other parts.
    """
    payload = message.get("payload", {})

    def decode_data(data: Optional[str]) -> str:
        if not data:
            return ""
        try:
            return base64.urlsafe_b64decode(data.encode("UTF-8")).decode("utf-8", errors="ignore")
        except Exception:
            return ""

    def walk(part: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
        """
        Returns (text_plain, text_html) if found in this part subtree.
        """
        mime = part.get("mimeType", "")
        body = part.get("body", {})
        data = body.get("data")

        # If this is a leaf text part, decode and return.
        if mime == "text/plain":
            return decode_data(data), None
        if mime == "text/html":
            html = decode_data(data)
            return None, BeautifulSoup(html, "html.parser").get_text(separator="\n")

        # If multipart, walk children.
        for child in part.get("parts", []) or []:
            txt, html_txt = walk(child)
            if txt or html_txt:
                # Prefer plain text but keep html as backup.
                if txt:
                    return txt, html_txt
                return None, html_txt
        return None, None

    txt, html_txt = walk(payload)
    if txt:
        return txt
    if html_txt:
        return html_txt

    # Fallback to body at top-level if nothing found
    return decode_data(payload.get("body", {}).get("data"))


def extract_image_urls_from_text(text: str) -> List[str]:
    """Extract plausible image URLs from email text."""
    if not text:
        return []
    urls = re.findall(r"https?://[^\s)\"']+", text)
    out = []
    for u in urls:
        parsed = urlparse(u)
        if not parsed.scheme.startswith("http"):
            continue
        lower = u.lower()
        if any(lower.endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".webp", ".gif"]):
            out.append(u)
        elif any(tok in lower for tok in ["cdn.", "/media/", "/files/"]):
            out.append(u)
    return out[:5]


def extract_kith_style_code(text: str):
    if not text:
        return None
    m = KITH_CODE_RE.search(text)
    return m.group(0).lower() if m else None


def extract_handle_from_url(url: str):
    m = HANDLE_FROM_URL_RE.search(url or "")
    return m.group(1).lower() if m else None


def generate_product_description(item: Dict[str, Any]) -> Optional[str]:
    """
    Use OpenAI to create a concise product description using brand, color, editor_note, and other metadata.
    """
    try:
        brand = item.get("brand") or ""
        name = item.get("product_name") or ""
        color = item.get("color") or ""
        editor_note = item.get("editor_note") or ""
        fabric = item.get("fabric") or ""
        fit = item.get("fit") or ""

        prompt = f"""
Write a concise, informative product description (2-3 sentences) for a clothing item.
Use the details below; prefer concrete facts over marketing fluff.

Brand: {brand}
Product: {name}
Color: {color}
Editor note/details: {editor_note}
Fabric: {fabric}
Fit: {fit}

Return just the description text (no JSON, no bullets).
""".strip()

        resp = client.responses.create(
            model="gpt-4o-mini",
            input=[
                {"role": "system", "content": "You write concise, factual product descriptions."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_output_tokens=200,
        )
        for out in resp.output:
            if out.type == "message":
                for c in getattr(out, "content", []) or []:
                    if c.type == "output_text":
                        text = c.text.strip()
                        return text
    except Exception as e:
        print(f"⚠️ product description generation failed: {e}")
    return None


def expand_webview_body(email_body: str, brand_key: str = "") -> str:
    """
    For brands that send barebones emails with a 'view in browser' link (e.g., Banana Republic),
    fetch the linked webview HTML and return its text if present.
    """
    if not email_body:
        return email_body

    # Only attempt for known brands that use view.email.* webviews
    webview_domains = [
        "view.email.bananarepublic.com",
        "view.email.bananarepublicfactory.com",
    ]
    urls = re.findall(r"https?://[^\s]+", email_body)
    for url in urls:
        if any(domain in url for domain in webview_domains):
            try:
                resp = requests.get(
                    url,
                    timeout=15,
                    headers={"User-Agent": DEFAULT_HEADERS["User-Agent"]},
                )
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, "html.parser")
                text = soup.get_text(separator="\n")
                if text.strip():
                    print(f"🌐 Fetched webview for {brand_key or 'email'}: {url}")
                    return text
            except Exception as e:
                print(f"⚠️ Failed to fetch webview {url}: {e}")
    return email_body


# -------------- OpenAI Helpers --------------

def call_openai_for_extraction(prompt: str) -> Any:
    """
    Calls the OpenAI API with a system + user message to extract product metadata.
    """
    resp = client.responses.create(
        model="gpt-4o",
        input=[
            {
                "role": "system",
                "content": "You are a structured data extraction assistant. "
                           "You extract clothing product line items from order confirmation emails. "
                           "Return ONLY strict JSON – no prose, no markdown code fences. "
                           "Prefer capturing every purchased line item rather than returning an empty list.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.0,
        max_output_tokens=2000,
    )

    # The new Responses API: parse JSON from the first output_text
    for out in resp.output:
        if out.type == "message":
            # New Responses API attaches message content directly to the output
            for c in getattr(out, "content", []) or []:
                if c.type == "output_text":
                    text = c.text
                    # Try to parse JSON
                    try:
                        return json.loads(text)
                    except Exception:
                        # Last resort: extract JSON substring via simple heuristic
                        start = text.find("{")
                        end = text.rfind("}")
                        if start != -1 and end != -1 and end > start:
                            return json.loads(text[start : end + 1])
    raise RuntimeError("No JSON output from OpenAI")


def extract_product_metadata(email_text: str) -> List[Dict[str, Any]]:
    """
    Use OpenAI to extract product line items from the email body text.
    """
    strict_prompt = f"""
Extract ONLY purchased clothing or accessory line items from the email below. Require evidence of an order/receipt/shipment (e.g., order number, quantity, price, subtotal, delivery status). Ignore marketing content, recommendations, lookbooks, sale banners, or unrelated links.

Return ONLY valid JSON (no code fences, no prose) shaped exactly as:
{{
  "items": [
    {{
      "product_name": "...",          // required string
      "brand": "...",                 // string or null
      "size": "...",                  // string or null (e.g., "M", "32x34", "10")
      "color": "...",                 // string or null (e.g., "Navy", "104")
      "price": "...",                 // string or null, keep currency symbol if present (e.g., "$120")
      "quantity": 1,                  // integer or null
      "style_code": "..."             // SKU/style if present else null
    }}
  ]
}}

If no products are present, return {{"items":[]}}.

Email text:
\"\"\"{email_text}\"\"\""""

    relaxed_prompt = f"""
Extract all purchased clothing or accessory line items from the email below. Focus on items actually ordered or shipped; ignore pure marketing listings unrelated to an order.

Return ONLY valid JSON (no code fences, no prose) shaped exactly as:
{{
  "items": [
    {{
      "product_name": "...",
      "brand": "...",
      "size": "...",
      "color": "...",
      "price": "...",
      "quantity": 1,
      "style_code": "..."
    }}
  ]
}}

If no products are present, return {{"items":[]}}.

Email text:
\"\"\"{email_text}\"\"\""""

    # Try strict first, then relax if empty
    prompts = [strict_prompt, relaxed_prompt]
    for p in prompts:
        data = call_openai_for_extraction(p)
        items = data.get("items", [])
        if isinstance(items, list) and items:
            return items
    return []


def log_failed_extraction(entry: Dict[str, Any], path: str = "skipped_messages.jsonl") -> None:
    """
    Append a skipped extraction record for later debugging.
    """
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"⚠️ Failed to log skipped message: {e}")


def log_exception_record(entry: Dict[str, Any], path: str = "exceptions.jsonl") -> None:
    """
    Append an exception record for post-mortem debugging.
    """
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"⚠️ Failed to log exception record: {e}")


# -------------- Google Search Helpers --------------

def google_web_search(query: str, num_results: int = 3) -> List[str]:
    """
    Generic Google Custom Search to return web page links.
    """
    if not GOOGLE_CSE_KEY or not GOOGLE_CSE_CX:
        return []
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_CSE_KEY,
        "cx": GOOGLE_CSE_CX,
        "q": query,
        "num": num_results,
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        items = data.get("items", [])
        return [item["link"] for item in items if "link" in item]
    except Exception as e:
        print(f"⚠️ Google web search error: {e}")
        return []


def google_image_search(query: str, site_domain: Optional[str] = None) -> Optional[str]:
    """
    Use Google CSE to get an image URL. If site_domain is provided, prefer images on that domain.
    """
    if not GOOGLE_CSE_KEY or not GOOGLE_CSE_CX:
        return None
    if not query or not query.strip():
        return None
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_CSE_KEY,
        "cx": GOOGLE_CSE_CX,
        "q": query,
        "searchType": "image",
        "num": 5,
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        print(f"🧭 CSE image URL: {resp.url}")
        resp.raise_for_status()
        data = resp.json()
        items = data.get("items", [])
        if not items:
            print(
                f"⚠️ Google image search returned 0 items for query='{query}' (site={site_domain}) "
                f"info={data.get('searchInformation')} error={data.get('error')}"
            )
            # Retry without site bias if we had one
            if site_domain:
                try:
                    params_no_site = dict(params)
                    params_no_site["q"] = query.replace(f" site:{site_domain}", "")
                    resp2 = requests.get(url, params=params_no_site, timeout=15)
                    print(f"🧭 CSE retry URL: {resp2.url}")
                    resp2.raise_for_status()
                    data2 = resp2.json()
                    items2 = data2.get("items", [])
                    if not items2:
                        print(
                            f"⚠️ Retry without site filter also returned 0 items "
                            f"info={data2.get('searchInformation')} error={data2.get('error')}"
                        )
                        return None
                    return items2[0].get("link")
                except Exception as e2:
                    print(f"⚠️ Google image retry error: {e2}")
                    return None
            return None
        return items[0].get("link")
    except Exception as e:
        print(f"⚠️ Google image search error: {e}")
    return None


# -------------- PDP Scraping Helpers --------------

def scrape_og_image(url: str) -> Optional[str]:
    """Scrape OG image or the best large image from a product page."""
    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": DEFAULT_HEADERS["User-Agent"]})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
    except Exception as e:
        print(f"⚠️ scrape_og_image error for {url}: {e}")
        return None

    candidates: List[str] = []

    def norm(u: str) -> Optional[str]:
        if not u:
            return None
        u = u.strip(" '\"")
        if u.startswith("//"):
            u = "https:" + u
        if u.startswith("/"):
            try:
                from urllib.parse import urljoin
                u = urljoin(url, u)
            except Exception:
                pass
        return u

    # 1) og:image
    og = soup.find("meta", property="og:image")
    if og and og.get("content"):
        candidates.append(norm(og.get("content")))

    # 2) twitter:image
    tw = soup.find("meta", attrs={"name": "twitter:image"})
    if tw and tw.get("content"):
        candidates.append(norm(tw.get("content")))

    # 3) link rel image_src
    link_img = soup.find("link", rel="image_src")
    if link_img and link_img.get("href"):
        candidates.append(norm(link_img.get("href")))

    # 4) JSON-LD product/image blocks (helps sites like SuitSupply/Abercrombie)
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
            if isinstance(data, list):
                items = data
            else:
                items = [data]
            for obj in items:
                img_field = obj.get("image")
                if isinstance(img_field, str):
                    candidates.append(norm(img_field))
                elif isinstance(img_field, list):
                    for i in img_field:
                        if isinstance(i, str):
                            candidates.append(norm(i))
        except Exception:
            continue

    # 5) Largest srcset entry
    for img in soup.find_all("img"):
        srcset = img.get("srcset")
        if srcset:
            parts = [p.strip().split(" ") for p in srcset.split(",")]
            if parts:
                best = parts[-1][0]  # last is usually largest
                candidates.append(norm(best))
        src = img.get("src") or img.get("data-src")
        if src:
            lower = src.lower()
            if any(skip in lower for skip in ["sprite", "icon", "logo"]):
                continue
            candidates.append(norm(src))

    for c in candidates:
        if c and c.startswith("http"):
            return c
    return None


def scrape_product_details(url: str, brand_key: str) -> Dict[str, str]:
    """
    Scrape PDP HTML for brand-specific details like fabric and fit.
    We will NOT use these to overwrite the main description to avoid messy blobs.
    """
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
    except Exception as e:
        print(f"⚠️ scrape_product_details error for {url}: {e}")
        return {}

    text = soup.get_text(separator="\n", strip=True)
    text = re.sub(r"\n+", "\n", text)

    details: Dict[str, str] = {}

    # Simple heuristics for fabric and fit
    fabric_lines = []
    fit_lines = []

    for line in text.split("\n"):
        lower_line = line.lower()
        if any(keyword in lower_line for keyword in ["fabric", "cotton", "wool", "polyester", "linen", "cashmere"]):
            fabric_lines.append(line.strip())
        if any(keyword in lower_line for keyword in ["fit", "relaxed", "slim", "regular", "tailored"]):
            fit_lines.append(line.strip())

    if fabric_lines:
        details["fabric"] = " ".join(fabric_lines[:5])
    if fit_lines:
        details["fit"] = " ".join(fit_lines[:5])

    return details


# -------------- Shopify MCP Helpers --------------


def _extract_variants(product_raw: dict):
    for key in ["variants", "merchandises", "merchandise", "skus", "items"]:
        v = product_raw.get(key)
        if isinstance(v, list) and v:
            return v
    for key in ["product", "node"]:
        if isinstance(product_raw.get(key), dict):
            return _extract_variants(product_raw[key])
    return []


def _variant_options(v):
    opts = {}
    so = v.get("selectedOptions") or v.get("selected_options") or v.get("options")
    if isinstance(so, list):
        for o in so:
            name = _norm(o.get("name"))
            val = (o.get("value") or "").strip()
            if name:
                opts[name] = val
    vt = v.get("title") or ""
    if "/" in vt:
        parts = [p.strip() for p in vt.split("/")]
        for p in parts:
            if p.upper() in ["XS", "S", "M", "L", "XL", "XXL", "XXXL"]:
                opts.setdefault("size", p)
            else:
                if "color" not in opts:
                    opts.setdefault("color", p)
    return opts


def pick_best_variant(product_raw: dict, desired_color: str, desired_size: str, desired_price, style_code: Optional[str] = None):
    d_color = _norm(desired_color)
    d_size = _norm(desired_size)
    d_price, _ = parse_money(desired_price)
    best, best_score = None, -1
    variants = _extract_variants(product_raw)
    for v in variants:
        opts = _variant_options(v)
        v_color = _norm(opts.get("color") or opts.get("colour") or "")
        v_size = _norm(opts.get("size") or "")
        v_price_raw = v.get("price") or v.get("priceV2") or v.get("presentmentPrices")
        if isinstance(v_price_raw, list) and v_price_raw:
            v_price_raw = v_price_raw[0]
        v_price, _ = parse_money(v_price_raw)
        score = 0
        if d_color and (d_color == v_color or d_color in _norm(v.get("title", "")) or d_color in _norm(v.get("sku", "") or "")):
            score += 2
        if d_size and d_size == v_size:
            score += 2
        if d_price is not None and v_price is not None and approx_equal(d_price, v_price, tol=2.0):
            score += 1
        if v.get("available") is True:
            score += 0.1
        if style_code and style_code.lower() in (_norm(v.get("sku") or v.get("id") or v.get("title") or "")):
            score += 3
        if score > best_score:
            best, best_score = v, score
    return best, best_score


def mcp_search(
    search_term: str,
    store_domain: str,
    context: str = "Order email enrichment: return product details including variants",
):
    """
    Calls Storefront MCP search_shop_catalog. Returns (normalized_items, raw_items).
    Handles result.content[].text stringified JSON and simplified retry.
    """
    endpoint = f"https://{store_domain}/api/mcp"

    def _normalize(p: dict):
        title = p.get("title") or p.get("name") or ""
        handle = p.get("handle") or p.get("product_handle") or ""
        url = p.get("url") or p.get("product_url") or (f"https://{store_domain}/products/{handle}" if handle else "")
        image = p.get("image") or p.get("image_url") or ""
        description = p.get("description") or p.get("body_html") or p.get("excerpt") or ""
        price = p.get("price") or p.get("amount")
        currency = p.get("currency") or p.get("price_currency")
        variant_id = p.get("variant_id") or p.get("merchandise_id")
        return {
            "title": title,
            "handle": handle,
            "url": url,
            "image": image,
            "description": description,
            "price": price,
            "currency": currency,
            "variant_id": variant_id,
        }

    def _call(q):
        payload = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "id": 1,
            "params": {"name": "search_shop_catalog", "arguments": {"query": q, "context": context}},
        }
        r = requests.post(endpoint, headers=DEFAULT_HEADERS, json=payload, timeout=20)
        print(f"🛠️ MCP Debug: Searching '{q}' @ {store_domain} (ctx: {context})")
        print(f"🛠️ MCP Response [{r.status_code}]: {r.text[:300]}")
        r.raise_for_status()
        data = r.json()
        if "error" in data:
            raise RuntimeError(data["error"])

        res = data.get("result", {})
        raw_items = []

        if isinstance(res, dict) and isinstance(res.get("items"), list):
            raw_items = res["items"]
        elif isinstance(res, dict) and isinstance(res.get("content"), list):
            for entry in res["content"]:
                if entry.get("type") == "output":
                    out = entry.get("output")
                    if isinstance(out, list):
                        raw_items.extend(out)
                    elif isinstance(out, dict):
                        raw_items.append(out)
                elif entry.get("type") == "text":
                    parsed = _coerce_json(entry.get("text", ""))
                    if isinstance(parsed, dict):
                        if isinstance(parsed.get("products"), list):
                            raw_items.extend(parsed["products"])
                        elif isinstance(parsed.get("items"), list):
                            raw_items.extend(parsed["items"])
                    elif isinstance(parsed, list):
                        raw_items.extend(parsed)
        elif isinstance(res, list):
            raw_items = res

        norm = [_normalize(p) for p in raw_items]
        return norm, raw_items

    # full query then simplified
    norm, raw = _call(search_term)
    if not norm:
        simple = _simplify_query(search_term)
        if simple and simple.lower() != search_term.lower():
            print(f"🔁 Retrying MCP with simplified query: {simple}")
            norm, raw = _call(simple)

    if norm:
        print("✅ MCP first match:", json.dumps(norm[0], indent=2)[:800])

    return norm, raw


def shopify_search_suggest(store_domain: str, query: str, limit: int = 8):
    """
    Uses Shopify's public search suggest to fetch likely product handles.
    We'll then re-query MCP by handle for structured data.
    """
    try:
        endpoint = f"https://{store_domain}/search/suggest.json"
        params = {
            "q": query,
            "resources[type]": "product",
            "resources[limit]": str(limit),
        }
        r = requests.get(
            endpoint,
            params=params,
            timeout=12,
            headers={"User-Agent": DEFAULT_HEADERS["User-Agent"]},
        )
        print(f"🧭 suggest.json → {r.url}")
        r.raise_for_status()
        data = r.json()
        handles = []

        res = (data.get("resources") or {}).get("results") or {}
        products = res.get("products") or res.get("products_unavailable") or []
        for p in products:
            h = p.get("handle")
            if not h:
                url = p.get("url") or p.get("path") or ""
                if "/products/" in url:
                    h = url.rsplit("/products/", 1)[-1].strip("/ ")
            if h:
                handles.append(h)

        seen = set()
        uniq = []
        for h in handles:
            if h not in seen:
                uniq.append(h)
                seen.add(h)
        return uniq

    except Exception as e:
        print(f"❌ search/suggest failed: {e}")
        return []


def _pick_best_image_url(variant: Optional[Dict[str, Any]], product: Optional[Dict[str, Any]]) -> Optional[str]:
    """
    Prefer variant image, then product image(s).
    """
    candidates: List[str] = []

    def push(val):
        if isinstance(val, str) and val.strip():
            candidates.append(val.strip())
        if isinstance(val, list):
            for v in val:
                if isinstance(v, dict):
                    push(v.get("url") or v.get("src"))
                elif isinstance(v, str):
                    push(v)

    if variant:
        vimg = variant.get("image")
        if isinstance(vimg, dict):
            push(vimg.get("url") or vimg.get("src"))
        else:
            push(vimg)
        push(variant.get("image_urls") or variant.get("images"))
    if product:
        push(product.get("image"))
        push(product.get("image_url"))
        push(product.get("image_urls") or product.get("images"))

    for c in candidates:
        return c
    return None


def enrich_with_shopify_mcp(
    brand_key: str,
    cfg: Dict[str, Any],
    item: Dict[str, Any],
    email_body: str,
) -> Dict[str, Any]:
    """
    Use Shopify MCP (hosted at the brand domain) to enrich an item with product_url, image_url, description, etc.
    Mirrors the flow used in quickstart copy.py so we do not rely on localhost MCP.
    """
    store_domain = cfg.get("search_domain") or ""
    if not store_domain:
        return item

    base_query = (item.get("product_name") or "").strip()
    base_title_query = _simplify_query(base_query)
    style_code = extract_kith_style_code(email_body) or extract_kith_style_code(base_query)
    color_name = (item.get("color") or "").strip()
    color_code = KITH_COLOR_TO_CODE.get(color_name) if color_name else None
    kith_handle = None

    mcp_norm, mcp_raw = [], []

    # Kith: try style-code/handle first (from quickstart copy)
    if brand_key == "kith" and style_code:
        # If color was a numeric code (e.g., "104"), keep it for style_code search
        if not color_code and (item.get("color") or "").strip().isdigit():
            if (item.get("color") or "").strip() in KITH_COLOR_CODES:
                color_code = (item.get("color") or "").strip()
                # Also populate human-readable color for later display
                item["color"] = KITH_COLOR_CODES[color_code]
        if style_code and color_code:
            n, r = mcp_search(f"{style_code}-{color_code}", store_domain, context="Enrich by handle")
            mcp_norm, mcp_raw = n or [], r or []
        if not mcp_norm:
            n, r = mcp_search(style_code, store_domain, context="Enrich by style code")
            mcp_norm, mcp_raw = n or [], r or []
        kith_handle = style_code
        if color_code:
            kith_handle = f"{style_code}-{color_code}"
        # Try handle-based MCP if we have one
        if kith_handle and not mcp_norm:
            n, r = mcp_search(kith_handle, store_domain, context="Enrich by handle from style code")
            mcp_norm, mcp_raw = n or [], r or []

    # If still nothing, title-based search
    if not mcp_norm:
        mcp_norm, mcp_raw = mcp_search(
            base_title_query or base_query,
            store_domain,
            context="Title-based enrichment including variants",
        )

    # If still nothing, try public search/suggest to get handles → MCP by handle
    if not mcp_norm:
        handles = shopify_search_suggest(store_domain, base_title_query or base_query, limit=8)
        print(f"🧭 suggest handles for '{base_title_query or base_query}': {handles}")
        for h in handles:
            n, r = mcp_search(h, store_domain, context="Enrich by handle from search/suggest")
            if n:
                mcp_norm, mcp_raw = n, r
                break

    enriched = dict(item)
    enriched.setdefault("brand_key", brand_key)
    mcp_image: Optional[str] = None

    picked = False
    if mcp_raw:
        best_product = None
        best_variant = None
        best_score = -1
        norm_title_query = normalize_product_name(base_title_query or base_query)

        for product in mcp_raw:
            product_score = 0
            handle_val = (product.get("handle") or "").lower()
            url_val = (product.get("url") or product.get("product_url") or "").lower()
            title_val = (product.get("title") or product.get("name") or "").lower()
            norm_title_val = normalize_product_name(title_val)

            # Style code match is a strong signal
            if style_code and (style_code in handle_val or style_code in url_val or style_code in title_val):
                product_score += 5

            # Name similarity boost
            if norm_title_query and norm_title_query in norm_title_val:
                product_score += 2

            v, v_score = pick_best_variant(
                product_raw=product,
                desired_color=item.get("color", ""),
                desired_size=item.get("size", ""),
                desired_price=item.get("price") or item.get("Price") or None,
                style_code=style_code,
            )
            total_score = product_score + (v_score or 0)
            if total_score > best_score:
                best_score = total_score
                best_variant = v
                best_product = product

        if best_variant and best_product:
            handle = best_product.get("handle") or best_product.get("product_handle") or ""
            base_url = best_product.get("url") or best_product.get("product_url")
            if not base_url and handle:
                base_url = f"https://{store_domain}/products/{handle}"
            enriched["product_url"] = base_url

            var_id = best_variant.get("id") or best_variant.get("variant_id") or best_variant.get("merchandiseId") or best_variant.get("merchandise_id")
            if base_url and var_id:
                enriched["product_url"] = f"{base_url}?variant={var_id}"

            img_choice = _pick_best_image_url(best_variant, best_product)
            if img_choice:
                mcp_image = img_choice

            enriched["editor_note"] = best_product.get("description") or best_product.get("body_html")
            vprice = best_variant.get("price") or best_variant.get("priceV2")
            p_amt, p_cur = parse_money(vprice)
            if p_amt is None and "price" in best_product:
                p_amt, p_cur = parse_money(best_product.get("price"))
            enriched["price"] = p_amt
            enriched["currency"] = p_cur
            enriched["variant_id"] = var_id
            picked = True

    # If MCP didn't give us a variant, fall back to normalized entry
    if not picked:
        if mcp_norm:
            best = mcp_norm[0]
            enriched["product_url"] = best.get("url")
            mcp_image = best.get("image")
            enriched["editor_note"] = best.get("description")
            enriched["price"] = best.get("price")
            enriched["currency"] = best.get("currency")
            enriched["variant_id"] = best.get("variant_id")
        else:
            print(f"⚠️ No Shopify MCP products found for: {brand_key} / {base_title_query}")

    # Shopify MCP image (variant/product)
    if not enriched.get("image_url") and mcp_image:
        enriched["image_url"] = mcp_image

    # If MCP didn't give us an image, try brand-specific PDP search + OG image (copy.py behavior)
    if not enriched.get("image_url"):
        if brand_key == "kith":
            page = google_web_search(f"{base_title_query or base_query} site:kith.com/products")
            if page and "kith.com/products" in page:
                enriched["product_url"] = enriched.get("product_url") or page
                scraped_img = scrape_og_image(page)
                if scraped_img:
                    enriched["image_url"] = scraped_img
                handle = extract_handle_from_url(enriched.get("product_url"))
                if handle:
                    n, r = mcp_search(handle, store_domain, context="Enrich by handle from URL")
                    if n:
                        best = n[0]
                        enriched["product_url"] = best.get("url") or enriched.get("product_url")
                        enriched["image_url"] = enriched.get("image_url") or best.get("image")
                        enriched["editor_note"] = best.get("description") or enriched.get("editor_note")
                        enriched["price"] = best.get("price") or enriched.get("price")
                        enriched["currency"] = best.get("currency") or enriched.get("currency")
                        enriched["variant_id"] = best.get("variant_id")

        elif brand_key == "aime_leon_dore":
            page = google_web_search(f"{base_title_query or base_query} site:aimeleondore.com/products")
            if page and "aimeleondore.com/products" in page:
                enriched["product_url"] = enriched.get("product_url") or page
                scraped_img = scrape_og_image(page)
                if scraped_img:
                    enriched["image_url"] = scraped_img

        elif brand_key == "shopbop":
            page = google_web_search(f"{base_title_query or base_query} site:shopbop.com")
            if page and "shopbop.com" in page:
                enriched["product_url"] = enriched.get("product_url") or page
                scraped_img = scrape_og_image(page)
                if scraped_img:
                    enriched["image_url"] = scraped_img

    # Last-resort Google image search (broad)
    if not enriched.get("image_url"):
        enriched["image_url"] = google_image_search(
            " ".join([enriched.get("brand", ""), base_title_query or enriched.get("product_name", ""), enriched.get("color", "")]).strip()
        )

    return enriched


def enrich_non_shopify(
    brand_key: str,
    cfg: Dict[str, Any],
    item: Dict[str, Any],
    email_images: List[str],
) -> Dict[str, Any]:
    """
    Enrich non-Shopify items by:
    1) Searching for a specific PDP URL on the brand's domain
    2) Scraping OG image and basic details
    """
    enriched = dict(item)
    search_domain = cfg.get("search_domain")
    base_title_query = item.get("product_name") or ""
    base_title_query = base_title_query.strip()

    if not search_domain or not base_title_query:
        return enriched

    # Google web search for PDP with brand + color hints to improve matches (helps Abercrombie)
    query = " ".join([base_title_query, item.get("brand", ""), item.get("color", ""), f"site:{search_domain}"]).strip()
    links = google_web_search(query, num_results=3)
    if not links:
        print(f"⚠️ No PDP links found for {brand_key} (query='{query}')")

    for page in links:
        enriched["product_url"] = enriched.get("product_url") or page

        # Try OG image first
        page_image = scrape_og_image(page)
        if page_image:
            enriched["image_url"] = enriched.get("image_url") or page_image

        # Scrape fabric/fit; leave description alone to keep it readable
        extra = scrape_product_details(page, brand_key)
        for key in ["fabric", "fit"]:
            if extra.get(key) and not enriched.get(key):
                enriched[key] = extra[key]

        # If we found at least an image or some extra detail, we can stop
        if enriched.get("image_url") or extra:
            break

    # Image resolution order:
    # 1) Scrape product URL (if found)
    if enriched.get("product_url") and not enriched.get("image_url"):
        print(f"🕵️ Scraping product URL for image: {enriched['product_url']}")
        og_img = scrape_og_image(enriched["product_url"])
        if og_img:
            enriched["image_url"] = og_img
        else:
            print(f"⚠️ No image scraped from product URL {enriched['product_url']}")

    # 2) Abercrombie: use the clean, broad query first
    if brand_key == "abercrombie" and not enriched.get("image_url"):
        g_img = google_image_search(build_abercrombie_image_query(enriched))
        if g_img:
            enriched["image_url"] = g_img
        else:
            print(f"⚠️ Abercrombie broad image search returned none for {enriched.get('product_name')}")

    # 2b) SuitSupply: brand-specific broad query (only when brand is suitsupply)
    if brand_key == "suitsupply" and not enriched.get("image_url"):
        g_img = google_image_search(build_suitsupply_image_query(enriched))
        if g_img:
            enriched["image_url"] = g_img
        else:
            print(f"⚠️ SuitSupply brand-specific image search returned none for {enriched.get('product_name')}")

    # 3) Email-embedded images
    if not enriched.get("image_url"):
        for img in email_images:
            enriched["image_url"] = img
            break

    # 4) Site-scoped image search on brand/CDN hosts
    if not enriched.get("image_url"):
        preferred_hosts = []
        if brand_key == "abercrombie":
            preferred_hosts = ["img.abercrombie.com", search_domain]
        elif brand_key == "suitsupply":
            preferred_hosts = ["cdn.suitsupply.com", search_domain]
        else:
            preferred_hosts = [search_domain]

        scoped_query = build_image_query(enriched, site_domain=search_domain)
        got = False
        for host in preferred_hosts:
            if not host:
                continue
            g_img = google_image_search(scoped_query, site_domain=host)
            if g_img:
                enriched["image_url"] = g_img
                got = True
                break
        if not got:
            print(f"⚠️ Site-scoped Google image search returned none for {brand_key} / {enriched.get('product_name')} (hosts={preferred_hosts})")

    # 5) Broad image search without site constraint
    if not enriched.get("image_url"):
        g_img = google_image_search(build_image_query(enriched))
        if g_img:
            enriched["image_url"] = g_img
        else:
            print(f"⚠️ Broad Google image search returned none for {brand_key} / {enriched.get('product_name')}")

    return enriched


# -------------- Main Pipeline --------------

def main():
    service = gmail_authenticate()

    # Build a Gmail query to find relevant order emails across all supported senders
    sender_query = " OR ".join(SUPPORTED_SENDERS.keys())
    subject_filter = ""
    if STRICT_ORDER_SUBJECT:
        subject_filter = " subject:(order OR receipt OR confirmation OR shipped OR delivered)"
    gmail_query = f"from:({sender_query}){subject_filter}"

    
    print("📨 Searching for order confirmation emails...")
    messages: List[Dict[str, Any]] = []
    page_token: Optional[str] = None
    while True:
        resp = (
            service.users()
            .messages()
            .list(userId="me", q=gmail_query, maxResults=200, pageToken=page_token)
            .execute()
        )
        batch = resp.get("messages", [])
        messages.extend(batch)
        page_token = resp.get("nextPageToken")
        if (MAX_EMAILS > 0 and len(messages) >= MAX_EMAILS) or not page_token or not batch:
            break
    if MAX_EMAILS > 0 and len(messages) > MAX_EMAILS:
        messages = messages[:MAX_EMAILS]

    if not messages:
        print("No messages found.")
        return

    print(f"Found {len(messages)} messages (capped at {MAX_EMAILS}).")
    all_products: List[Dict[str, Any]] = []
    seen_ids: set = set()
    skip_stats = {
        "unsupported_sender": 0,
        "empty_body": 0,
        "no_extraction": 0,
        "duplicates": 0,
        "exceptions": 0,
    }
    brand_counts: Dict[str, int] = {}
    duplicate_examples: List[str] = []
    earliest_ts: Optional[int] = None

    for msg_meta in messages:
        try:
            msg_id = msg_meta["id"]
            msg = (
                service.users()
                .messages()
                .get(userId="me", id=msg_id, format="full")
                .execute()
            )
            try:
                ts = int(msg.get("internalDate"))
                earliest_ts = ts if earliest_ts is None else min(earliest_ts, ts)
            except Exception:
                pass

            headers = msg.get("payload", {}).get("headers", [])
            header_dict = {h["name"]: h["value"] for h in headers}
            from_header = header_dict.get("From", "")
            subject_header = header_dict.get("Subject", "")

            print(f"\n-------------------------------")
            print(f"📧 Processing email: {subject_header}")
            print(f"From: {from_header}")

            # Identify which supported sender / brand this email corresponds to
            matched_sender = None
            brand_key = None
            for sender, bk in SUPPORTED_SENDERS.items():
                if sender.lower() in from_header.lower():
                    matched_sender = sender
                    brand_key = bk
                    break

            if not matched_sender or not brand_key:
                print("⚠️  Skipping email – sender not in SUPPORTED_SENDERS.")
                skip_stats["unsupported_sender"] += 1
                log_failed_extraction(
                    {
                        "reason": "unsupported_sender",
                        "subject": subject_header,
                        "from": from_header,
                        "body_text": get_message_body(msg)[:4000],
                    }
                )
                continue

            cfg = BRAND_CONFIG[brand_key]
            brand_type = cfg["type"]

            body_text = get_message_body(msg)
            body_text = expand_webview_body(body_text, brand_key=brand_key)
            email_images = extract_image_urls_from_text(body_text)
            if not body_text.strip():
                print("⚠️ Could not extract email body.")
                skip_stats["empty_body"] += 1
                continue

            # Extract product metadata using OpenAI
            extracted = extract_product_metadata(body_text)
            if not extracted:
                print("⚠️ No products extracted from this email.")
                skip_stats["no_extraction"] += 1
                log_failed_extraction(
                    {
                        "reason": "no_extraction",
                        "brand_key": brand_key,
                        "subject": subject_header,
                        "from": from_header,
                        "body_text": body_text[:4000],  # cap to keep file reasonable
                    }
                )
                continue

            print(f"✅ Extracted {len(extracted)} candidate items from email for brand_key={brand_key}")

            for item in extracted:
                brand_counts[brand_key] = brand_counts.get(brand_key, 0) + 1
                # Brand normalization based on sender / brand_key
                if brand_key == "kith" and "receipt" in subject_header.lower():
                    item["brand"] = "Kith"
                elif brand_key in ("banana_republic", "banana_republic_factory"):
                    # Always normalize to Banana Republic for UI, regardless of what LLM returned
                    item["brand"] = "Banana Republic"
                elif brand_key == "abercrombie":
                    item["brand"] = "Abercrombie & Fitch"
                elif brand_key == "netaporter":
                    # Treat this as retailer; brand usually lives in product_name
                    item["retailer"] = "NET-A-PORTER"
                elif brand_key == "weworewhat":
                    item["brand"] = "WeWoreWhat"
                elif brand_key == "everlane":
                    item["brand"] = "Everlane"

                # Kith color codes → color names
                if brand_key == "kith":
                    color = item.get("color", "")
                    # If color looks like a color code, keep it as-is or map if needed later
                    # (placeholder for more advanced mapping)

                # De-dupe with configurable strictness so we can keep all variants when desired.
                # Dedup keys: use raw name for full; normalized (color-stripped) for coarse.
                raw_name_key = re.sub(r"\s+", " ", (item.get("product_name") or "").strip().lower())
                coarse_name_key = normalize_product_name(item.get("product_name") or "")
                color_key = re.sub(r"\s+", " ", (item.get("color") or "").strip().lower())
                size_key = re.sub(r"\s+", " ", (item.get("size") or "").strip().lower())

                full_key = f"{brand_key}|{raw_name_key}|{color_key}|{size_key}"
                coarse_key = f"{brand_key}|{coarse_name_key or raw_name_key}"

                should_skip = False
                if DEDUP_STRATEGY == "coarse":
                    should_skip = full_key in seen_ids or coarse_key in seen_ids
                elif DEDUP_STRATEGY == "full":
                    should_skip = full_key in seen_ids
                elif DEDUP_STRATEGY == "none":
                    should_skip = False
                else:
                    should_skip = full_key in seen_ids or coarse_key in seen_ids  # default to coarse

                if should_skip:
                    print(f"🛑 Skipping duplicate item: {full_key} (strategy={DEDUP_STRATEGY})")
                    skip_stats["duplicates"] += 1
                    if len(duplicate_examples) < 5:
                        duplicate_examples.append(full_key)
                    continue

                seen_ids.add(full_key)
                seen_ids.add(coarse_key)

                base_title_query = item.get("product_name", "")

                enriched = dict(item)
                enriched.setdefault("brand_key", brand_key)
                # Quick initial image guess to avoid blanks (broad Google image)
                enriched["image_url"] = google_image_search(build_image_query(item))

                # ---------- Enrichment branch ----------
                if brand_type == "shopify":
                    enriched = enrich_with_shopify_mcp(
                        brand_key=brand_key,
                        cfg=cfg,
                        item=item,
                        email_body=body_text,
                    )
                else:
                    enriched = enrich_non_shopify(
                        brand_key=brand_key,
                        cfg=cfg,
                        item=item,
                        email_images=email_images,
                    )

                # Last-resort image if still missing
                if not enriched.get("image_url"):
                    enriched["image_url"] = google_image_search(
                        build_image_query(enriched)
                    )

                # Generate product description if missing
                if not enriched.get("product_description"):
                    desc = generate_product_description(enriched)
                    if desc:
                        enriched["product_description"] = desc

                print(
                    f"🧺 Final enriched item: "
                    f"{enriched.get('brand')} - {enriched.get('product_name')} "
                    f"(color={enriched.get('color')}, size={enriched.get('size')})"
                )

                all_products.append(enriched)

        except Exception as e:
            print(f"⚠️ Error processing message: {e}")
            skip_stats["exceptions"] += 1
            try:
                log_exception_record(
                    {
                        "error": str(e),
                        "brand_key": locals().get("brand_key"),
                        "subject": locals().get("subject_header"),
                        "from": locals().get("from_header"),
                    }
                )
            except Exception:
                pass

    print(f"🧾 Total products extracted: {len(all_products)}")
    print(
        "🧮 Skip summary -> "
        f"unsupported_sender={skip_stats['unsupported_sender']}, "
        f"empty_body={skip_stats['empty_body']}, "
        f"no_extraction={skip_stats['no_extraction']}, "
        f"duplicates={skip_stats['duplicates']}, "
        f"exceptions={skip_stats['exceptions']}"
    )
    if brand_counts:
        print(f"📊 Items extracted per brand: {brand_counts}")
    if earliest_ts:
        print(f"⏳ Oldest message processed: {datetime.fromtimestamp(earliest_ts/1000).isoformat()}")
    if duplicate_examples:
        print(f"🧬 Sample duplicate keys: {duplicate_examples}")
    if all_products:
        with open("products.json", "w") as f:
            json.dump(all_products, f, indent=2)
        print("✅ Saved product metadata to products.json")
    else:
        print("⚠️ No products to save.")


if __name__ == "__main__":
    main()
