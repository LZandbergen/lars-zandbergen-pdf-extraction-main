#### IMPORTANT: THIS FILE DOES NOT YET STORE THE TABLES IN A NEAT WAY, IT ONLY SCRAPES THE CONTENT OF THE WEBSITES ####

#%% Import modules
import pandas as pd
import requests
from pathlib import Path
from bs4 import BeautifulSoup
from bs4.element import Comment
from requests.exceptions import RequestException, Timeout
from urllib.parse import urlparse
import asyncio
import aiohttp
import json
import os
from datetime import datetime
from collections import defaultdict
import time
from typing import Dict, List, Tuple, Any, Iterator
import logging


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='scraping.log'
)

#%% Define page parser
def get_html_content(url: str) -> str:
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
    except (RequestException, Timeout) as e:
        print(f"Error fetching URL: {e}")
        return ""
    return response.content
    
def tag_visible(element) -> bool:
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True

def parse_html(html_content) -> str:
    try:
        soup = BeautifulSoup(html_content, "html.parser")
        visible_text = filter(tag_visible, soup.find_all(text=True))
        # Join the filtered content into a single string
        visible_text_str = " ".join(t.strip() for t in visible_text)
        return visible_text_str
    except Exception as e:
        print(f"Error parsing HTML content: {e}")
        return None
    
#%% Get URL List
with open('search_bol.csv', 'r') as f:
    search_bol = pd.read_csv(f)
    unique_urls = set(search_bol['search_url'].tolist())

# Print sample of URLs, and statistics
print(f"Number of unique URLs: {len(unique_urls)}")
print(f"Sample of URLs: {list(unique_urls)[:5]}")

# Print the amount of unique base domains in the URLs, and the average amount of URLs per base domain
base_domains = [urlparse(url).netloc for url in unique_urls]
unique_base_domains = set(base_domains)
avg_urls_per_base_domain = len(unique_urls) / len(unique_base_domains)
print(f"Number of unique base domains: {len(unique_base_domains)}")
print(f"Average amount of URLs per base domain: {avg_urls_per_base_domain}")


#%% Scrape the content of each URL and save per website to a JSON file

class ScrapingStats:
    def __init__(self):
        self.successful = 0
        self.failed = 0
        self.errors: Dict[str, int] = defaultdict(int)

    def add_error(self, error_type: str):
        self.failed += 1
        self.errors[error_type] += 1

    def add_success(self):
        self.successful += 1

    def __str__(self):
        return f"Successful: {self.successful}, Failed: {self.failed}, Errors: {dict(self.errors)}"
    
    def save_to_file(self):
        with open('scraping_stats.json', 'w') as f:
            json.dump({
                'successful': self.successful,
                'failed': self.failed,
                'errors': dict(self.errors)
            }, f)

async def scrape_url(session: aiohttp.ClientSession, url: str, tmp_dir: str, retries: int = 3) -> Tuple[bool, dict]:
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    filename = f"{tmp_dir}/{domain}/{hash(url)}.json"
    
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return True, json.load(f)
    
    for attempt in range(retries):
        try:
            async with session.get(url, timeout=30) as response:
                if response.status != 200:
                    raise Exception(f"HTTP {response.status}")
                
                content = await response.text()
                parsed_content = parse_html(content)
                
                if not parsed_content:
                    raise Exception("Failed to parse HTML content")
                
                soup = BeautifulSoup(content, "html.parser")
                result = {
                    'url': url,
                    'title': soup.title.string if soup.title else None,
                    'content': parsed_content,
                    'timestamp': datetime.now().isoformat()
                }
                
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                with open(filename, 'w') as f:
                    json.dump(result, f)
                    
                return True, result
                
        except Exception as e:
            if attempt == retries - 1:
                return False, {'url': url, 'error': str(e)}
            await asyncio.sleep(2 ** attempt)  # Exponential backoff

async def save_batch_results(batch_results: List[dict], domain: str, output_file: str):
    async with aiolock:  # Define this at module level
        with open(output_file, 'a') as f:
            for result in batch_results:
                f.write(json.dumps({domain: result}) + '\n')

async def scrape_domain(session: aiohttp.ClientSession, urls: List[str], stats: ScrapingStats, 
                       tmp_dir: str, output_file: str) -> None:
    batch_results = []
    for url in urls:
        success, result = await scrape_url(session, url, tmp_dir)
        if success:
            stats.add_success()
            batch_results.append(result)
        else:
            stats.add_error(result['error'])
        
        if len(batch_results) >= 10:  # Save in batches of 10
            await save_batch_results(batch_results, urlparse(urls[0]).netloc, output_file)
            batch_results = []
            
        await asyncio.sleep(1)
    
    if batch_results:  # Save remaining results
        await save_batch_results(batch_results, urlparse(urls[0]).netloc, output_file)

async def main():
    # Add semaphore for connection limiting
    semaphore = asyncio.Semaphore(10)  # Max 10 concurrent connections
    
    # Initialize files
    output_file = 'jsons/scraped_content.jsonl'
    if os.path.exists(output_file):
        os.rename(output_file, f"{output_file}.{int(time.time())}.backup")
    
    # Group URLs by domain
    domain_urls: Dict[str, List[str]] = defaultdict(list)
    for url in unique_urls:
        domain = urlparse(url).netloc
        domain_urls[domain].append(url)

    tmp_dir = 'tmp_scrapes'
    os.makedirs(tmp_dir, exist_ok=True)

    stats = ScrapingStats()
    
    # Configure client session with proper limits
    timeout = aiohttp.ClientTimeout(total=30, connect=10)
    connector = aiohttp.TCPConnector(limit_per_host=2)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = []
        for domain, urls in domain_urls.items():
            task = asyncio.create_task(
                scrape_domain(session, urls, stats, tmp_dir, output_file)
            )
            tasks.append(task)
            if len(tasks) >= 5:  # Process domains in batches
                await asyncio.gather(*tasks)
                tasks = []
                await asyncio.sleep(5)
        
        if tasks:
            await asyncio.gather(*tasks)
    
    stats.save_to_file()
    logging.info(f"Scraping completed. {str(stats)}")

# Define module-level lock for file access
aiolock = asyncio.Lock()

def decode_strings_recursive(obj: Any) -> Any:
    """Recursively decode Unicode escape sequences in strings"""
    if isinstance(obj, str):
        return obj.encode().decode('unicode-escape')
    elif isinstance(obj, dict):
        return {k: decode_strings_recursive(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [decode_strings_recursive(item) for item in obj]
    return obj

def load_scraping_jsonl(file_path: str) -> Iterator[Dict[str, Any]]:
    """Load JSONL file and yield decoded dictionaries line by line"""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
        
    with open(path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                # Parse JSON and decode Unicode
                data = json.loads(line.strip())
                yield decode_strings_recursive(data)
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
            except Exception as e:
                print(f"Unexpected error on line {line_num}: {e}")

if __name__ == "__main__":
    asyncio.run(main())