import os
import shutil
from datetime import datetime
from pathlib import Path
from ollama import chat
from bs4 import BeautifulSoup
import requests
from openai import OpenAI
from dotenv import load_dotenv

from utils.logs import log
from utils.tokenizer.tokenizer import get_sentence_tokenizer
from evaluation import evaluate_results

# Load environment variables from .env file
load_dotenv()

# Configuration
MODEL_NAME = "qwen3:8b"
MAX_CHUNK_SIZE = 900
MAX_RETRIES = 3  # Maximum retry attempts per chunk before skipping
DATA_DIR = Path("data/test")
PROMPTS_DIR = Path("prompts")
RESULTS_DIR = Path("results")
PROMPT_NAME = "baseline"
BY_ENTITY_MODE = False  # If True, uses entity-specific prompts from prompts/by_entity/{PROMPT_NAME}/
API_MODE = False  # If True, uses external API instead of local Ollama
OPENAI_MODE = False  # If True, uses OpenAI API (overrides API_MODE)
OPENAI_API_KEY = ""  # Set your OpenAI API key here or use environment variable
ENTITY_TYPES = ["pessoa", "organizacao", "tempo", "local", "legislacao", "jurisprudencia"]


def chunk_text(text: str, max_chars: int = MAX_CHUNK_SIZE) -> list[str]:
    # Get sentence tokenizer (same as LeNER-Br)
    sentence_tokenizer = get_sentence_tokenizer()

    # Tokenize text into sentences
    sentences = sentence_tokenizer.tokenize(text)

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence)

        # If adding this sentence would exceed max_chars, save current chunk and start new one
        if current_length + sentence_length + 1 > max_chars and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_length = 0

        # Add sentence to current chunk
        current_chunk.append(sentence)
        current_length += sentence_length + 1  # +1 for space

    # Add remaining sentences as final chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


def annotate_chunk(prompt: str, model: str = MODEL_NAME) -> str:
    # Annotate chunk using local Ollama model
    response = chat(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        options={"temperature": 0}
    )
    return response.message.content


def annotate_chunk_api(prompt: str, model: str = MODEL_NAME) -> str:
    # Annotate chunk using external API
    api_endpoint = os.environ.get("API_ENDPOINT")
    project = os.environ.get("API_PROJECT")

    payload = {
        "model": model,
        "project": project,
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post(
        api_endpoint,
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    response.raise_for_status()

    # Parse response: {"choices": [{"message": {"content": "..."}}]}
    response_data = response.json()
    return response_data["choices"][0]["message"]["content"]


def annotate_chunk_openai(prompt: str, model: str = MODEL_NAME) -> str:
    # Annotate chunk using OpenAI API
    # Get API key from config or environment variable
    api_key = os.environ.get("OPENAI_API_KEY", OPENAI_API_KEY)
    if not api_key:
        raise ValueError("OpenAI API key not set. Set OPENAI_API_KEY in config or environment variable.")

    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content


def extract_clean_html(output: str) -> str:
    html = output

    # Remove <think>...</think> blocks (Qwen reasoning)
    if "<think>" in html.lower():
        if "</think>" not in html.lower():
            raise ValueError("Incomplete <think> tag - missing closing tag")
        think_start = html.lower().index("<think>")
        think_end = html.lower().index("</think>") + len("</think>")
        html = html[:think_start] + html[think_end:]

    # Remove common output prefixes
    if "***output***" in html.lower():
        html = html[html.lower().index("***output***")+len("***output***")+1:]
    if "output:" in html.lower():
        html = html[html.lower().index("output:")+len("output:")+1:]
    if "output text" in html.lower():
        html = html[html.lower().index("output text")+len("output text")+1:]
    if "***highlighted text***" in html.lower():
        html = html[html.lower().index("***highlighted text***")+len("***highlighted text***")+1:]

    # Remove HTML wrappers
    if "<body>" in html:
        if "</body>" not in html:
            raise ValueError("Incomplete <body> tag - missing closing tag")
        html = html[html.index("<body>")+6:html.index("</body>")]
    if "<p>" in html:
        if "</p>" not in html:
            raise ValueError("Incomplete <p> tag - missing closing tag")
        html = html[html.index("<p>")+3:html.index("</p>")]

    # Remove markdown code blocks
    if "```html" in html:
        html = html[html.index("```html")+7:]
    elif "```" in html:
        start = html.index("```")
        html = html[start+3:]

    # Remove closing ```
    if "```" in html:
        html = html[:html.index("```")]

    return html.strip()


def merge_entity_htmls(entity_htmls: dict[str, str]) -> str:
    # Entity priority based on dataset frequency (lower number = higher priority)
    ENTITY_PRIORITY = {
        "legislacao": 1,
        "organizacao": 2,
        "pessoa": 3,
        "jurisprudencia": 4,
        "tempo": 5,
        "local": 6
    }

    # Get base text from any non-empty HTML (all have the same text)
    base_text = None
    for html in entity_htmls.values():
        if html.strip():
            soup = BeautifulSoup(html, "html.parser")
            base_text = soup.get_text()
            break

    if not base_text:
        return ""

    # Extract all spans with their character positions
    all_spans = []  # [(start_pos, end_pos, entity_type, span_text), ...]

    for entity_type, html in entity_htmls.items():
        if not html.strip():
            continue

        soup = BeautifulSoup(html, "html.parser")

        # Find all spans for this entity
        for span in soup.find_all("span"):
            span_text = span.get_text()

            # Find position of this span in the base text
            # We'll search for the exact text in base_text
            start_pos = base_text.find(span_text)
            if start_pos != -1:
                end_pos = start_pos + len(span_text)
                all_spans.append((start_pos, end_pos, entity_type, span_text))

    # Resolve conflicts: sort by priority (highest frequency first)
    # When spans overlap, only the highest priority one will be selected
    sorted_spans = sorted(all_spans, key=lambda s: ENTITY_PRIORITY.get(s[2], 99))

    selected_spans = []
    blocked_positions = set()

    for start, end, entity, text in sorted_spans:
        span_positions = set(range(start, end))

        # Check if any position is already used
        if not span_positions.intersection(blocked_positions):
            selected_spans.append((start, end, entity, text))
            blocked_positions.update(span_positions)
        # else: discard completely (lower priority or overlap)

    # Sort by position to rebuild HTML in correct order
    selected_spans.sort(key=lambda s: s[0])

    # Rebuild HTML with selected spans
    if not selected_spans:
        return base_text

    result = ""
    last_pos = 0

    for start, end, entity, span_text in selected_spans:
        # Add text before span
        result += base_text[last_pos:start]
        # Add span
        result += f'<span class="{entity}">{span_text}</span>'
        last_pos = end

    # Add remaining text after last span
    result += base_text[last_pos:]

    return result


def run(documents: list[str], result_dir: Path, prompt_name: str = PROMPT_NAME):
    # Create result structure
    outputs_dir = result_dir / "outputs"
    prompts_output_dir = result_dir / "prompts"
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(prompts_output_dir, exist_ok=True)

    log_file = result_dir / "log.txt"

    log("Starting annotation process", log_file)
    log(f"Model: {MODEL_NAME}", log_file)
    log(f"Max chunk size: {MAX_CHUNK_SIZE} characters", log_file)
    log(f"Prompt: {prompt_name}", log_file)
    log(f"By-entity mode: {BY_ENTITY_MODE}", log_file)
    log(f"OpenAI mode: {OPENAI_MODE}", log_file)
    log(f"API mode: {API_MODE}", log_file)
    log(f"Documents to process: {len(documents)}", log_file)

    # Load prompt template(s) and save copy
    if BY_ENTITY_MODE:
        # Load entity-specific prompts
        entity_prompts = {}
        by_entity_dir = PROMPTS_DIR / "by_entity" / prompt_name
        for entity_type in ENTITY_TYPES:
            entity_prompt_file = by_entity_dir / f"{entity_type}.txt"
            with open(entity_prompt_file, "r", encoding="utf-8") as f:
                entity_prompts[entity_type] = f.read()

            # Save copy of each entity prompt
            shutil.copy(entity_prompt_file, prompts_output_dir / f"{entity_type}.txt")

        log(f"Loaded and saved {len(entity_prompts)} entity-specific prompts from: {by_entity_dir}", log_file)
        prompt_template = None  # Not used in BY_ENTITY_MODE
    else:
        # Load single prompt template
        prompt_file = PROMPTS_DIR / f"{prompt_name}.txt"
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompt_template = f.read()

        # Save copy of prompt used
        shutil.copy(prompt_file, prompts_output_dir / f"{prompt_name}.txt")
        log(f"Loaded and saved prompt template: {prompt_name}", log_file)
        entity_prompts = None  # Not used in standard mode

    # Process each document
    for doc_name in documents:
        doc_path = DATA_DIR / doc_name
        log(f"Processing document: {doc_name}", log_file)

        # Read document text from BIO file (extract only text, ignore tags)
        with open(doc_path, "r", encoding="utf-8") as f:
            text = " ".join([line.split()[0] for line in f.read().splitlines() if line.strip()])

        # Split into chunks
        chunks = chunk_text(text)
        log(f"  Split into {len(chunks)} chunks", log_file)

        # Create output directory for this document
        doc_outputs_dir = outputs_dir / doc_name.replace(".bio", "")
        os.makedirs(doc_outputs_dir, exist_ok=True)

        # Process each chunk
        for i, chunk in enumerate(chunks, start=1):
            log(f"  Processing chunk {i}/{len(chunks)}", log_file)

            # Create chunk directory
            chunk_dir = doc_outputs_dir / f"chunk{i}"
            os.makedirs(chunk_dir, exist_ok=True)

            if BY_ENTITY_MODE:
                # Process chunk once for each entity
                entity_htmls = {}

                for entity_type in ENTITY_TYPES:
                    log(f"    Processing entity: {entity_type}", log_file)

                    retry_count = 0
                    success = False

                    while not success and retry_count < MAX_RETRIES:
                        try:
                            # Format prompt with chunk text
                            formatted_prompt = entity_prompts[entity_type].format(chunk)

                            # Get annotation from model
                            if OPENAI_MODE:
                                output = annotate_chunk_openai(formatted_prompt)
                            elif API_MODE:
                                output = annotate_chunk_api(formatted_prompt)
                            else:
                                output = annotate_chunk(formatted_prompt)

                            # Extract clean HTML
                            clean_output = extract_clean_html(output)

                            # Save entity-specific output
                            with open(chunk_dir / f"{entity_type}.html", "w", encoding="utf-8") as f:
                                f.write(clean_output)

                            entity_htmls[entity_type] = clean_output
                            success = True

                        except Exception as e:
                            retry_count += 1
                            log(f"      Error processing {entity_type} (attempt {retry_count}/{MAX_RETRIES}): {str(e)}", log_file)

                            if retry_count >= MAX_RETRIES:
                                log(f"      Skipping {entity_type} after {MAX_RETRIES} failed attempts", log_file)
                                entity_htmls[entity_type] = ""
                                break

                # Merge all entity HTMLs into single clean_output.html
                merged_html = merge_entity_htmls(entity_htmls)

                # Save merged output
                with open(chunk_dir / "clean_output.html", "w", encoding="utf-8") as f:
                    f.write(merged_html)

                log(f"    Saved chunk {i} with {len([e for e in entity_htmls.values() if e])} entities to {chunk_dir}", log_file)

            else:
                # Standard mode: process chunk once with single prompt
                retry_count = 0
                success = False

                while not success and retry_count < MAX_RETRIES:
                    try:
                        # Format prompt with chunk text
                        formatted_prompt = prompt_template.format(chunk)

                        # Get annotation from model
                        if OPENAI_MODE:
                            output = annotate_chunk_openai(formatted_prompt)
                        elif API_MODE:
                            output = annotate_chunk_api(formatted_prompt)
                        else:
                            output = annotate_chunk(formatted_prompt)

                        # Extract clean HTML
                        clean_output = extract_clean_html(output)

                        # Save formatted prompt
                        with open(chunk_dir / "formatted_prompt.txt", "w", encoding="utf-8") as f:
                            f.write(formatted_prompt)

                        # Save raw output
                        with open(chunk_dir / "output.html", "w", encoding="utf-8") as f:
                            f.write(output)

                        # Save clean output
                        with open(chunk_dir / "clean_output.html", "w", encoding="utf-8") as f:
                            f.write(clean_output)

                        log(f"    Saved chunk {i} to {chunk_dir}", log_file)
                        success = True

                    except Exception as e:
                        retry_count += 1
                        log(f"    Error processing chunk {i} (attempt {retry_count}/{MAX_RETRIES}): {str(e)}", log_file)

                        if retry_count >= MAX_RETRIES:
                            log(f"    Skipping chunk {i} after {MAX_RETRIES} failed attempts", log_file)
                            break

        log(f"  Completed document: {doc_name}", log_file)

    log("Annotation process complete", log_file)


if __name__ == "__main__":
    # Create result directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = RESULTS_DIR / f"result_{timestamp}"
    os.makedirs(result_dir, exist_ok=True)

    log_file = result_dir / "log.txt"

    # Log which data directory is being used
    log(f"Processing documents from DATA_DIR: {DATA_DIR.absolute()}", log_file)

    # Get all .bio files in data directory
    documents = [f.name for f in DATA_DIR.glob("*.bio")]

    if not documents:
        log("No documents found", log_file)
        print("No documents found in data directory")
    else:
        log(f"Found {len(documents)} documents: {', '.join(documents)}", log_file)
        print(f"Found {len(documents)} documents")
        print(f"Result directory: {result_dir}")

        # Run annotation
        run(documents, result_dir)

        # Run evaluation
        log("Starting evaluation", log_file)
        print("\nRunning evaluation...")
        evaluation_dir = evaluate_results(result_dir, DATA_DIR, MAX_CHUNK_SIZE, log_file)
        print(f"\nEvaluation complete!")
        print(f"Results saved to: {evaluation_dir}")
