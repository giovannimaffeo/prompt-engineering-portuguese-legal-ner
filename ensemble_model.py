from pathlib import Path
from bs4 import BeautifulSoup
from datetime import datetime
import os
import shutil

RESULT_DIRS = [
    "results/research_question_4a",
    "results/research_question_4b",
    "results/research_question_4c"
]

OUTPUT_DIR = None
ENTITY_TYPES = ["pessoa", "organizacao", "tempo", "local", "legislacao", "jurisprudencia"]
VOTING_THRESHOLD = 2


def extract_entities_from_html(html_path):
    entities = []

    if not html_path.exists():
        return entities

    with open(html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    soup = BeautifulSoup(html_content, 'html.parser')

    for span in soup.find_all('span'):
        entity_class = span.get('class')
        if entity_class:
            entity_type = entity_class[0] if isinstance(entity_class, list) else entity_class
            entity_text = span.get_text().strip()

            if entity_text and entity_type in ENTITY_TYPES:
                entities.append({
                    'text': entity_text,
                    'type': entity_type,
                    'start': span.sourceline if hasattr(span, 'sourceline') else 0
                })

    return entities


def get_base_text(html_path):
    if not html_path.exists():
        return ""

    with open(html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    soup = BeautifulSoup(html_content, 'html.parser')
    return soup.get_text()


def vote_entities(entities_list):
    entity_votes = {}

    for entities in entities_list:
        for entity in entities:
            key = (entity['text'].lower(), entity['type'])

            if key not in entity_votes:
                entity_votes[key] = {
                    'text': entity['text'],
                    'type': entity['type'],
                    'votes': 0
                }

            entity_votes[key]['votes'] += 1

    voted_entities = []
    for key, entity_info in entity_votes.items():
        if entity_info['votes'] >= VOTING_THRESHOLD:
            voted_entities.append({
                'text': entity_info['text'],
                'type': entity_info['type']
            })

    return voted_entities


def create_html_output(base_text, entities):
    if not entities:
        return ""

    html_parts = []

    for entity in entities:
        html_parts.append(f'<span class="{entity["type"]}">{entity["text"]}</span>')

    return ' '.join(html_parts)


def process_chunk(chunk_paths):
    all_entities = []
    base_text = ""

    for chunk_path in chunk_paths:
        html_file = chunk_path / "clean_output.html"

        if html_file.exists():
            if not base_text:
                base_text = get_base_text(html_file)

            entities = extract_entities_from_html(html_file)
            all_entities.append(entities)

    if not all_entities:
        return None, base_text

    voted_entities = vote_entities(all_entities)
    output_html = create_html_output(base_text, voted_entities)

    return output_html, base_text


def get_common_documents():
    common_docs = None

    for result_dir in RESULT_DIRS:
        outputs_path = Path(result_dir) / "outputs"

        if not outputs_path.exists():
            continue

        docs = set([d.name for d in outputs_path.iterdir() if d.is_dir()])

        if common_docs is None:
            common_docs = docs
        else:
            common_docs = common_docs.intersection(docs)

    return sorted(list(common_docs)) if common_docs else []


def process_document(doc_name):
    print(f"Processing document: {doc_name}")

    result_paths = [Path(result_dir) / "outputs" / doc_name for result_dir in RESULT_DIRS]

    all_chunks = set()
    for result_path in result_paths:
        if result_path.exists():
            chunks = [c.name for c in result_path.glob("chunk*") if c.is_dir()]
            all_chunks.update(chunks)

    if not all_chunks:
        print(f"  No chunks found for {doc_name}")
        return

    output_doc_dir = Path(OUTPUT_DIR) / "outputs" / doc_name
    os.makedirs(output_doc_dir, exist_ok=True)

    for chunk_name in sorted(all_chunks, key=lambda x: int(x.replace('chunk', ''))):
        chunk_paths = [
            result_path / chunk_name
            for result_path in result_paths
            if (result_path / chunk_name).exists()
        ]

        if len(chunk_paths) < VOTING_THRESHOLD:
            continue

        output_html, base_text = process_chunk(chunk_paths)

        if output_html is None:
            continue

        chunk_output_dir = output_doc_dir / chunk_name
        os.makedirs(chunk_output_dir, exist_ok=True)

        with open(chunk_output_dir / "clean_output.html", "w", encoding="utf-8") as f:
            f.write(output_html)

        with open(chunk_output_dir / "chunk.txt", "w", encoding="utf-8") as f:
            f.write(base_text)

    print(f"  Completed: {len(list(output_doc_dir.glob('chunk*')))} chunks")


def copy_prompts_directory():
    source_prompt = Path(RESULT_DIRS[0]) / "prompts"
    target_prompt = Path(OUTPUT_DIR) / "prompts"

    if source_prompt.exists():
        shutil.copytree(source_prompt, target_prompt, dirs_exist_ok=True)


def create_ensemble_log():
    log_file = Path(OUTPUT_DIR) / "log.txt"
    timestamp = datetime.now().strftime("%H:%M:%S")

    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"{timestamp} - Ensemble voting started\n")
        f.write(f"{timestamp} - Model: ensemble\n")
        f.write(f"{timestamp} - Max chunk size: 900 characters\n")
        f.write(f"{timestamp} - Prompt: ensemble_voting\n")
        f.write(f"{timestamp} - By-entity mode: False\n")
        f.write(f"{timestamp} - OpenAI mode: False\n")
        f.write(f"{timestamp} - API mode: False\n")
        f.write(f"{timestamp} - Source results:\n")
        for result_dir in RESULT_DIRS:
            f.write(f"{timestamp} -   {result_dir}\n")
        f.write(f"{timestamp} - Voting threshold: {VOTING_THRESHOLD}/{len(RESULT_DIRS)}\n")
        f.write(f"{timestamp} - Entity types: {', '.join(ENTITY_TYPES)}\n")


def main():
    global OUTPUT_DIR

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR = f"results/result_{timestamp}"

    print("="*70)
    print("ENSEMBLE MODEL - VOTING SYSTEM")
    print("="*70)
    print()

    for result_dir in RESULT_DIRS:
        result_path = Path(result_dir)
        status = "✓" if result_path.exists() else "✗"
        print(f"  {status} {result_dir}")
    print()

    print(f"Voting threshold: {VOTING_THRESHOLD} out of {len(RESULT_DIRS)}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(Path(OUTPUT_DIR) / "outputs", exist_ok=True)
    os.makedirs(Path(OUTPUT_DIR) / "evaluation", exist_ok=True)

    create_ensemble_log()
    copy_prompts_directory()

    common_docs = get_common_documents()

    print(f"Found {len(common_docs)} common documents")
    print()

    for doc_name in common_docs:
        process_document(doc_name)

    print()
    print("="*70)
    print("ENSEMBLE COMPLETE")
    print("="*70)
    print(f"Output saved to: {OUTPUT_DIR}")
    print()
    print("To evaluate, run:")
    print(f"  1. Update RESULT_DIR in unit_evaluate.py to: {OUTPUT_DIR}")
    print(f"  2. Update RESULTS_DIR in entity_extraction_evaluation.py to: Path('{OUTPUT_DIR}')")
    print(f"  3. python unit_evaluate.py")
    print(f"  4. python entity_extraction_evaluation.py")
    print()


if __name__ == "__main__":
    main()
