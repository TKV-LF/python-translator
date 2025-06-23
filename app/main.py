from urllib.parse import urljoin, urlparse
import re
from fastapi import FastAPI, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse
import httpx
from bs4 import BeautifulSoup, NavigableString
import os
from dotenv import load_dotenv
import asyncio
from typing import Dict, List, Literal
import html
import json
import logging
from starlette.exceptions import HTTPException as StarletteHTTPException
from datetime import datetime
from fastapi.responses import FileResponse


load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize API keys (currently not used for dichtienghoa/vietphrase)
API_KEYS = []

app = FastAPI()

# Only mount static files if directory exists
static_dir = "app/static"
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

templates = Jinja2Templates(directory="app/templates")


current_api_key_index = 0
DICHTIENGHOA_URL = "https://dichtienghoa.com/transtext"
VIETPHRASE_URL = "http://vietphrase.info/VietPhrase/Browser"


# Translation settings
MAX_TOKENS_PER_REQUEST = 3000
CONCURRENT_TRANSLATIONS = 3  # Number of parallel translations

translation_cache: Dict[str, str] = {}

# Translation method type
TranslationMethod = Literal["openrouter", "base", "vietphrase"]

def get_next_api_key() -> str:
    """Rotate to the next available API key."""
    global current_api_key_index
    valid_keys = [key for key in API_KEYS if key]  # Filter out None or empty keys
    if not valid_keys:
        logger.error("No valid API keys configured")
        return ""
        
    current_api_key_index = (current_api_key_index + 1) % len(valid_keys)
    return valid_keys[current_api_key_index]

def get_current_api_key() -> str:
    """Get the current API key."""
    return API_KEYS[current_api_key_index]

def clean_content(content_tag: BeautifulSoup) -> BeautifulSoup:
    """Clean up content by removing duplicates and unwanted elements."""
    # Remove unwanted elements
    for tag in content_tag.find_all(['script', 'style', 'iframe', 'ins', 'button', 'input', 'form']):
        tag.decompose()
    
    # Remove duplicate paragraphs
    seen_texts = set()
    for tag in content_tag.find_all(['p', 'div']):
        text = tag.get_text().strip()
        if text in seen_texts:
            tag.decompose()
        else:
            seen_texts.add(text)
    
    return content_tag

def split_html_content(content: str) -> List[str]:
    """Split HTML content into smaller chunks while preserving HTML structure."""
    soup = BeautifulSoup(content, 'html.parser')
    chunks = []
    current_chunk = []
    current_size = 0
    seen_texts = set()  # Track unique text content
    
    for element in soup.find_all(['p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
        # Skip empty elements
        
        if not element.get_text(strip=True):
            continue
            
        # Skip duplicate content
        text = element.get_text().strip()
        if text in seen_texts:
            continue
        seen_texts.add(text)
        
        element_html = str(element)
        element_size = len(element_html)
        
        if current_size + element_size > MAX_TOKENS_PER_REQUEST and current_chunk:
            chunks.append(''.join(current_chunk))
            current_chunk = []
            current_size = 0
        
        current_chunk.append(element_html)
        current_size += element_size
    
    if current_chunk:
        chunks.append(''.join(current_chunk))
    
    return chunks

async def translate_with_dichtienghoa(text: str) -> str:
    """Translate text using dichtienghoa.com API."""
    if not text.strip():
        return ""
    
    # Check cache
    cache_key = f"dth:{text.strip()}"
    if cache_key in translation_cache:
        return translation_cache[cache_key]

    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Origin": "https://dichtienghoa.com",
        "Referer": "https://dichtienghoa.com/"
    }
    
    data = {
        "t": text,
        "tt": "vi"
    }
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                DICHTIENGHOA_URL,
                data=data,
                headers=headers
            )
            response.raise_for_status()
            
            result = response.json()
            if result['data']:
                translation_cache[cache_key] = result['data']
                return result['data']
            return text
    except Exception as e:
        print(f"Dichtienghoa translation error: {str(e)}")
        return text

async def translate_chunk_dichtienghoa(chunk: str, semaphore: asyncio.Semaphore) -> str:
    """Translate a chunk using dichtienghoa.com with rate limiting."""
    if not chunk.strip():
        return ""
        
    async with semaphore:
        # Extract text content while preserving HTML structure
        soup = BeautifulSoup(chunk, 'html.parser')
        
        # Collect all text nodes and their positions
        text_nodes = []
        text_contents = []
        for text_node in soup.find_all(text=True):
            if isinstance(text_node, NavigableString) and text_node.strip():
                text_nodes.append(text_node)
                text_contents.append(text_node.strip())
        
        if not text_contents:
            return chunk
            
        # Combine all text with a special separator
        combined_text = "\n[SEP]\n".join(text_contents)
        
        # Translate all text at once
        translated_text = await translate_with_dichtienghoa(combined_text)
        
        # Split the translated text back into individual translations
        if translated_text and translated_text != combined_text:
            translations = translated_text.split("\n[SEP]\n")
            
            # Replace original text with translations
            for text_node, translation in zip(text_nodes, translations):
                if translation.strip():
                    text_node.replace_with(translation.strip())
        
        return str(soup)

async def translate_content(content: str, method: TranslationMethod = "openrouter") -> str:
    """Translate content using specified method."""
    if not content.strip():
        return ""
    
    # Split content into chunks
    chunks = split_html_content(content)
    
    # Create semaphore for concurrent translations
    semaphore = asyncio.Semaphore(CONCURRENT_TRANSLATIONS)
    
    # Choose translation function based on method
    translate_func = translate_chunk_dichtienghoa
    
    # Translate chunks in parallel
    tasks = [translate_func(chunk, semaphore) for chunk in chunks]
    translated_chunks = await asyncio.gather(*tasks)
    
    # Combine translated chunks
    return ''.join(translated_chunks)

async def modify_url(url: str, base_url: str) -> str:
    """Convert external URLs to internal application URLs."""
    if not url:
        return "#"
    
    # Remove @ prefix if present
    if url.startswith('@'):
        url = url[1:]
    # Handle protocol-relative URLs (starting with //)
    if url.startswith('//'):
        url = f"https:{url}"
    
    # Clean up multiple forward slashes and redundant domain names
    parsed = urlparse(url)

    if parsed.netloc:
        # Split path by the domain name to remove redundant occurrences
        path_parts = parsed.path.split(parsed.netloc)
        cleaned_path = path_parts[-1] if path_parts else parsed.path
        # Clean up multiple forward slashes in path
        cleaned_path = re.sub(r'/+', '/', cleaned_path)
        # Reconstruct URL with cleaned path
        url = f"{parsed.scheme}://{parsed.netloc}{cleaned_path}"
        if parsed.query:
            url += f"?{parsed.query}"
        if parsed.fragment:
            url += f"#{parsed.fragment}"
    
    # Handle relative URLs
    parsed = urlparse(url)

    if not parsed.netloc:
        url = urljoin(base_url, url)
    
    print(url)
    return f"/translate?url={url}"

async def translate_with_vietphrase_api(content: str) -> str:
    """Translate content using VietPhrase TranslateHanViet API."""
    vietphrase_translate_url = "https://vietphrase.info/Vietphrase/TranslateVietPhraseS"
    
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Origin": "https://vietphrase.info",
        "Referer": "https://vietphrase.info/"
    }
    
    # Try form data instead of JSON
    payload = {
        "chineseContent": content
    }
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Try POST with form data
            response = await client.post(
                vietphrase_translate_url,
                data=payload,
                headers=headers
            )
            response.raise_for_status()
            
            # Parse the response
            response_text = response.text
            # Check if response is JSON
            if response_text and len(response_text) > 10:
                return response_text
                
    except Exception as e:
        print(f"VietPhrase API translation error: {str(e)}")
        # Fallback to dichtienghoa translation
        try:
            print("Trying dichtienghoa as backup...")
            return await translate_content(content, "base")
        except Exception as fallback_e:
            print(f"Backup translation also failed: {str(fallback_e)}")
            return content

async def extract_content_fallback(url: str) -> str:
    """Extract content manually from URL as fallback for VietPhrase."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    }
    
    try:
        async with httpx.AsyncClient(verify=False, timeout=30.0) as client:
            response = await client.get(url, headers=headers, follow_redirects=True)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find content using the identifier classes
            content = None
            identifierClasses = ['page-content', 'print', 'mybox', 'text-data', 'chapter-content', 'article-content', 'box_con', 'book']
            
            # Special handling for shuhaige.net
            if url.startswith('https://m.shuhaige.net') or url.startswith('https://www.shuhaige.net'):
                identifierClasses = ['headeline', 'pager', 'content']
                merged_content = soup.new_tag('div')
                merged_content['class'] = 'merged-content'
                for class_name in identifierClasses:
                    found_element = soup.find(class_=re.compile(class_name))
                    if found_element:
                        element_copy = found_element.decode_contents()
                        merged_content.append(BeautifulSoup(element_copy, 'html.parser'))
                content = merged_content
            else: 
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    from urllib.parse import urlparse
                    parsed = urlparse(href)
                    original_url = parsed.path
                    if original_url:
                        modified_href = await modify_url(original_url, url)
                        link['href'] = f"{modified_href}&method=vietphrase"
                for class_name in identifierClasses:
                    content = soup.find(class_=re.compile(class_name))
                    if content:
                        break
            return str(content)
            
    except Exception as e:
        print(f"Content extraction fallback error: {str(e)}")
        return ""

async def translate_with_vietphrase(url: str) -> str:
    """Translate using VietPhrase service with fallback to manual extraction."""
    vietphrase_url = "http://vietphrase.info/VietPhrase/Browser"
    params = {
        "url": url,
        "GB2312": "true",
        "script": "true",
        "t": "VP"
    }
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    }
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(vietphrase_url, params=params, headers=headers)
            response.raise_for_status()
            # Parse the response HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find the container element
            content = soup.find(id='html')
            if content:
                # Remove unwanted elements including style tags
                for element in content.find_all(['meta', 'link', 'base', 'style', 'script']):
                    element.decompose()
                
                # Check if content is too short or minimal (like empty div)
                content_text = content.get_text(strip=True)
                if len(content_text) < 50 or not content_text:
                    print("Content is too short or empty, using fallback method...")
                    raise Exception("Content too short, using fallback")
                    
                # Update all VietPhrase links
                for link in content.find_all('a', href=True):
                    href = link['href']
                    if 'vietphrase.info/VietPhrase/Browser' in href:
                        # Extract the original URL from VietPhrase link
                        from urllib.parse import urlparse, parse_qs
                        parsed = urlparse(href)
                        original_url = parse_qs(parsed.query).get('url', [''])[0]
                        if original_url:
                            modified_href = await modify_url(original_url, url)
                            link['href'] = f"{modified_href}&method=vietphrase"
                print(content)
                
                return str(content)
            
            # If no content found, try fallback
            raise Exception("No content found in VietPhrase response")
            
    except Exception as e:
        print(f"VietPhrase Browser translation error: {str(e)}, trying fallback...")
        
        # Fallback: Extract content manually and use VietPhrase API
        try:
            extracted_content = await extract_content_fallback(url)
            if extracted_content:
                translated_content = await translate_with_vietphrase_api(extracted_content)
                return translated_content
            else:
                raise Exception("Could not extract content for translation")
        except Exception as fallback_error:
            print(f"VietPhrase fallback error: {str(fallback_error)}")
            return ""
    
async def extract_content(url: str, method: TranslationMethod = "base") -> tuple:
    """Extract content from the given URL and prepare it for translation."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    }
    
    async with httpx.AsyncClient(verify=False, timeout=30.0) as client:
        try:
            # If using VietPhrase, handle differently
            translated_content = None
            translated_title = None
            if method == "vietphrase":
                extracted_content = await translate_with_vietphrase(url)
                if extracted_content:
                    translated_title = "VietPhrase Translation"
                    translated_content = extracted_content
                else:
                    raise HTTPException(status_code=400, detail="VietPhrase translation failed")
            else:
                # Extract title first
                response = await client.get(url, headers=headers, follow_redirects=True)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract title first
                title = soup.title.string if soup.title else "Unknown Title"
                
                # Find the main content
                content = None
                # The above code appears to be a comment in Python. It starts with a "#" symbol, which
                # indicates that it is a single-line comment. The text "identifierClasses" seems to be a
                # placeholder or a note about the content of the code. Comments are used to provide
                # explanations or notes within the code for better understanding by developers.
                identifierClasses = ['page-content', 'print', 'mybox', 'text-data', 'chapter-content', 'article-content', 'box_con', 'book']
                
                # Special handling for shuhaige.net
                if url.startswith('https://m.shuhaige.net') or url.startswith('https://www.shuhaige.net'):
                    identifierClasses = ['headeline', 'pager', 'content']
                    merged_content = soup.new_tag('div')
                    merged_content['class'] = 'merged-content'
                    for class_name in identifierClasses:
                        found_element = soup.find(class_=re.compile(class_name))
                        if found_element:
                            element_copy = found_element.decode_contents()
                            merged_content.append(BeautifulSoup(element_copy, 'html.parser'))
                    content = merged_content
                else: 
                    for class_name in identifierClasses:
                        content = soup.find(class_=re.compile(class_name))
                        if content:
                            break
                
                if not content:
                    raise HTTPException(status_code=400, detail="Could not find novel content")
                
                # Clean up content and remove duplicates
                content = clean_content(content)
                
                # Update links
                for link in content.find_all('a', href=True):
                    original_href = link['href']
                    modified_href = await modify_url(original_href, url)
                    link['href'] = f"{modified_href}&method={method}"
                    link['data-original-href'] = original_href
                
                # Translate title and content
                translated_title = await translate_content(title, "base")
                translated_content = await translate_content(str(content), "base")
                
                # Clean up any HTML tags in the translated content


           
            # Remove any remaining HTML entities
            translated_content = html.unescape(translated_content)
            translated_content = re.sub(r'<p></p>', '', translated_content)
            return translated_title, translated_content
            
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Error processing content: {str(e)}"
            )

# Add a health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Add startup and shutdown events
@app.on_event("startup")
async def startup_event():
    logger.info("Application starting up...")
    # Log environment variables (excluding sensitive data)
    logger.info(f"PORT: {os.getenv('PORT')}")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Application shutting down...")

# Redirect root to home
@app.get("/")
async def redirect_to_home():
    return RedirectResponse(url="/home")

# Home page
@app.get("/home", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html", 
        {
            "request": request,
            "year": datetime.now().year
        }
    )

@app.get("/translate", response_class=HTMLResponse)
async def translate(request: Request, url: str, method: TranslationMethod = "base"):
    try:
        title, content = await extract_content(url, method)
        return templates.TemplateResponse(
            "translation.html",
            {
                "request": request,
                "title": title,
                "content": content,
                "original_url": url,
                "translation_method": method,
                "year": datetime.now().year
            }
        )
    except HTTPException as e:
        return templates.TemplateResponse(
            "error.html",
            {
                "request": request,
                "error_message": str(e.detail),
                "original_url": url,
                "year": datetime.now().year
            },
            status_code=e.status_code
        )

# Custom 404 handler
@app.exception_handler(StarletteHTTPException)
async def custom_http_exception_handler(request: Request, exc: StarletteHTTPException):
    if exc.status_code == 404:
        return templates.TemplateResponse(
            "404.html", 
            {
                "request": request,
                "year": datetime.now().year
            }, 
            status_code=404
        )
    return templates.TemplateResponse(
        "error.html",
        {
            "request": request,
            "error_message": str(exc.detail),
            "year": datetime.now().year
        },
        status_code=exc.status_code
    ) 
    
@app.get("/img/{filename}")
async def send_file(filename: str):
    return FileResponse(f"{static_dir}/img/{filename}")