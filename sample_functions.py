import json
import requests
import time
from bs4 import BeautifulSoup
import logging
import ollama
from transformers import pipeline

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

def query_duckduckgo(query: str) -> str:
    """
    Query the DuckDuckGo Instant Answer API and return the results.
    query: The search query to send to DuckDuckGo.
    Returns:
        A summary of the top result from DuckDuckGo.
    """
    url = "https://api.duckduckgo.com/"
    params = {
        'q': query,
        'format': 'json'
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        # Extracting the abstract or top result
        result = data.get('AbstractText')
        if not result:
            related_topics = data.get('RelatedTopics', [])
            if related_topics:
                result = related_topics[0].get('Text', 'No result found.')
            else:
                result = 'No result found.'
        #print(result)
        return result
    else:
        return "Error querying DuckDuckGo API."

# Example function with description
def get_duckduckgo_result(query: str) -> str:
    """
    Get the top DuckDuckGo search result for the given query.
    query: The search query to send to DuckDuckGo.
    """
    return query_duckduckgo(query)


def do_math(a:int, op:str, b:int)->str:
    """
    Do basic math operations
    a: The first operand
    op: The operation to perform (one of '+', '-', '*', '/')
    b: The second operand
    """
    res = "Nan"
    if op == "+":
        res = str(int(a) + int(b))
    elif op == "-":
        res = str(int(a) - int(b))
    elif op == "*":
        res = str(int(a) * int(b))
    elif op == "/":
        if int(b) != 0:
            res = str(int(a) / int(b))
    return res

def get_current_time() -> str:
    """Get the current time"""
    current_time = time.strftime("%H:%M:%S")
    return f"The current time is {current_time}"

def get_current_weather(city:str) -> str:
    """Get the current weather for a city
    Args:
        city: The city to get the weather for
    """
    base_url = f"http://wttr.in/{city}?format=j1"
    response = requests.get(base_url)
    data = response.json()
    return f"The current temperature in {city} is: {data['current_condition'][0]['temp_F']}°F"


def scrape_and_summarize_locally(url):
    """
    Scrape content from a given URL and generate a summary.
    Args:
        url: The URL of the webpage to scrape and summarize
    """
    try:
        # Fetch the webpage content
        response = requests.get(url)
        response.raise_for_status()

        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract the main content (adjust selectors as needed)
        content = soup.find('article')
        if content:
            text = content.get_text(separator=' ', strip=True)
        else:
            text = soup.get_text(separator=' ', strip=True)

        # Split the text into smaller chunks
        max_chunk_size = 500  # Reduced from 1000 to 500
        chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]

        # Summarize each chunk
        # Note: Ensure you're running this script in a virtual environment with all dependencies installed
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        summaries = []
        for i, chunk in enumerate(chunks):
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    logger.info(f"Summarizing chunk {i+1}/{len(chunks)}")
                    summary = summarizer(chunk, max_length=100, min_length=30, do_sample=False)
                    summaries.append(summary[0]['summary_text'])
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Error summarizing chunk {i+1}. Retrying... (Attempt {attempt+1}/{max_retries})")
                        time.sleep(1)  # Wait for 1 second before retrying
                    else:
                        logger.error(f"Failed to summarize chunk {i+1} after {max_retries} attempts: {str(e)}")
                        summaries.append(f"[Error summarizing chunk {i+1}]")

        # Combine the summaries
        combined_summary = " ".join(summaries)

        # Generate a final summary if the combined summary is still too long
        if len(combined_summary) > max_chunk_size:
            logger.info("Generating final summary")
            try:
                final_summary = summarizer(combined_summary, max_length=100, min_length=50, do_sample=False)
                return final_summary[0]['summary_text']
            except Exception as e:
                logger.error(f"Error generating final summary: {str(e)}")
                # Fallback: return a truncated version of the combined summary
                return combined_summary[:1000] + "... (summary truncated due to processing error)"
        else:
            return combined_summary

    except requests.RequestException as e:
        logger.error(f"Error fetching the webpage: {str(e)}")
        return f"Error fetching the webpage: {str(e)}"
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        return f"An unexpected error occurred: {str(e)}"




def scrape_and_summarize(url):
    """
    Scrape content from a given URL and generate a summary using Ollama.
    Args:
        url: The URL of the webpage to scrape and summarize
    """
    #client = ollama.Client(host='http://monster-1:11434')
    try:
        # Fetch the webpage content
        response = requests.get(url)
        response.raise_for_status()

        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract the main content (adjust selectors as needed)
        content = soup.find('article')
        if content:
            text = content.get_text(separator=' ', strip=True)
        else:
            text = soup.get_text(separator=' ', strip=True)

        prompt = f"Please summarize the following text in a concise manner:\n\n{text[:4000]}"  # Limit to 4000 characters to avoid token limits
        response = ollama.generate(model="llama3.1", prompt=prompt)
        #response = client.generate(model="llama2", prompt=prompt)

        return response['response']

    except requests.RequestException as e:
        logger.error(f"Error fetching the webpage: {str(e)}")
        return f"Error fetching the webpage: {str(e)}"
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        return f"An unexpected error occurred: {str(e)}"

    except requests.RequestException as e:
        logger.error(f"Error fetching the webpage: {str(e)}")
        return f"Error fetching the webpage: {str(e)}"
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        return f"An unexpected error occurred: {str(e)}"
