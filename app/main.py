from fastapi import FastAPI, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse
import httpx
from bs4 import BeautifulSoup, NavigableString
import os
from dotenv import load_dotenv
import re
from urllib.parse import urljoin, urlparse, quote
import asyncio
import random
from typing import Dict, List, Literal
import html
import json
import logging
from starlette.exceptions import HTTPException as StarletteHTTPException
from datetime import datetime


load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Only mount static files if directory exists
static_dir = "app/static"
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

templates = Jinja2Templates(directory="app/templates")

# API Keys configuration
API_KEYS = [
    os.getenv(f"OPENROUTER_API_KEY_{i}") for i in range(1, 5)
]
current_api_key_index = 0
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
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

async def translate_chunk(chunk: str, semaphore: asyncio.Semaphore) -> str:
    """Translate a chunk using OpenRouter with rate limiting."""
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
        
        # Translate all text at once using OpenRouter
        translated_text = await translate_with_openrouter(combined_text)
        
        # Split and replace translations
        if translated_text and translated_text != combined_text:
            translations = translated_text.split("\n[SEP]\n")
            for text_node, translation in zip(text_nodes, translations):
                if translation.strip():
                    text_node.replace_with(translation.strip())
        
        return str(soup)

async def translate_with_openrouter(text: str) -> str:
    """Translate text using OpenRouter API."""
    if not text.strip():
        return ""
    
    # Check cache
    cache_key = f"or:{text.strip()}"
    if cache_key in translation_cache:
        return translation_cache[cache_key]

    api_key = get_next_api_key()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    data = {
        "model": "deepseek/deepseek-chat:free",
        "messages": [
            {
                "role": "system",
                "content": "You are a Chinese to Vietnamese translator. Translate the following text accurately while preserving any HTML formatting."
            },
            {
                "role": "user",
                "content": text
            }
        ]
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                OPENROUTER_API_URL,
                json=data,
                headers=headers
            )
            print(response)
            response.raise_for_status()
            print(response.text)
            result = response.json()
            
            if result.get('choices') and result['choices'][0].get('message'):
                translation = result['choices'][0]['message']['content']
                translation_cache[cache_key] = translation
                return translation
            return text
    except Exception as e:
        print(f"OpenRouter translation error: {str(e)}")
        return text

async def translate_content(content: str, method: TranslationMethod = "openrouter") -> str:
    """Translate content using specified method."""
    if not content.strip():
        return ""
    
    # Split content into chunks
    chunks = split_html_content(content)
    
    # Create semaphore for concurrent translations
    semaphore = asyncio.Semaphore(CONCURRENT_TRANSLATIONS)
    
    # Choose translation function based on method
    translate_func = translate_chunk if method == "openrouter" else translate_chunk_dichtienghoa
    
    # Translate chunks in parallel
    tasks = [translate_func(chunk, semaphore) for chunk in chunks]
    translated_chunks = await asyncio.gather(*tasks)
    
    # Combine translated chunks
    return ''.join(translated_chunks)

async def modify_url(url: str, base_url: str) -> str:
    """Convert external URLs to internal application URLs."""
    if not url:
        return "#"
    
    # Handle protocol-relative URLs (starting with //)
    if url.startswith('//'):
        url = f"https:{url}"
    
    parsed = urlparse(url)
    if not parsed.netloc:
        url = urljoin(base_url, url)
    
    return f"/translate?url={url}"

async def translate_with_vietphrase(url: str) -> str:
    """Translate using VietPhrase service."""
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
                # Update all VietPhrase links
                for element in content.find_all(['meta', 'link']):
                    element.decompose()
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
                
                return str(content)
            
            return ""
    except Exception as e:
        print(f"VietPhrase translation error: {str(e)}")
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
            if method == "vietphrase":
                translated_content = await translate_with_vietphrase(url)
                if translated_content:
                    return "VietPhrase Translation", translated_content
                raise HTTPException(status_code=400, detail="VietPhrase translation failed")
            
            # Extract title first
            response = await client.get(url, headers=headers, follow_redirects=True)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract title first
            title = soup.title.string if soup.title else "Unknown Title"
            
            # Find the main content
            content = None
            identifierClasses = ['page-content', 'print', 'mybox', 'text-data', 'chapter-content', 'article-content']
            
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
            translated_title = await translate_content(title, method)
            translated_content = await translate_content(str(content), method)
            
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
    logger.info("API Keys configured: %s", bool(API_KEYS[0]))

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
async def translate(request: Request, url: str, method: TranslationMethod = "openrouter"):
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