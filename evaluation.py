"""
Entity extraction evaluation using LeNER-Br exact matching methodology.
This implementation follows the original LeNER-Br evaluation script.
"""

from pathlib import Path
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from utils.tokenizer.tokenizer import word_tokenize
from utils.logs import log
from collections import OrderedDict

ENTITY_TYPES = ["pessoa", "organizacao", "tempo", "local", "legislacao", "jurisprudencia"]


def chunk_text(text: str, max_chars: int = 900) -> list[str]:
    """Import chunk_text function to avoid circular import"""
    from utils.tokenizer.tokenizer import get_sentence_tokenizer

    sentence_tokenizer = get_sentence_tokenizer()
    sentences = sentence_tokenizer.tokenize(text)

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence)

        if current_length + sentence_length + 1 > max_chars and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_length = 0

        current_chunk.append(sentence)
        current_length += sentence_length + 1

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


def _update_chunk(candidate, prev, current_tag, current_chunk, current_pos, prediction=False):
    """Extract entity chunks as position spans (LeNER-Br style)"""
    if candidate == 'B-' + current_tag:
        if len(current_chunk) > 0 and len(current_chunk[-1]) == 1:
            current_chunk[-1].append(current_pos - 1)
        current_chunk.append([current_pos])
    elif candidate == 'I-' + current_tag:
        if prediction and (current_pos == 0 or current_pos > 0 and prev.split('-', 1)[-1] != current_tag):
            current_chunk.append([current_pos])
        if not prediction and (current_pos == 0 or current_pos > 0 and prev == 'O'):
            current_chunk.append([current_pos])
    elif current_pos > 0 and prev.split('-', 1)[-1] == current_tag:
        if len(current_chunk) > 0:
            current_chunk[-1].append(current_pos - 1)


def _update_last_chunk(current_chunk, current_pos):
    """Close the last chunk"""
    if len(current_chunk) > 0 and len(current_chunk[-1]) == 1:
        current_chunk[-1].append(current_pos - 1)


def _tag_precision_recall_f1(tp, fp, fn):
    """Calculate precision, recall, F1"""
    precision, recall, f1 = 0, 0, 0
    if tp + fp > 0:
        precision = tp / (tp + fp) * 100
    if tp + fn > 0:
        recall = tp / (tp + fn) * 100
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def extract_entities_from_html_as_bio(html_path, chunk_text_content):
    """
    Extract entities from HTML and convert to BIO tags matching the chunk text.
    Returns list of BIO tags aligned with tokens.
    """
    if not html_path.exists():
        return None

    with open(html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    if not html_content.strip():
        return None

    soup = BeautifulSoup(html_content, 'html.parser')

    # Tokenize the chunk text
    tokens = word_tokenize(chunk_text_content, language='portuguese')
    bio_tags = ['O'] * len(tokens)

    # Extract entities from HTML
    for span in soup.find_all('span'):
        entity_class = span.get('class')
        if entity_class:
            entity_type = entity_class[0] if isinstance(entity_class, list) else entity_class
            entity_text = span.get_text().strip()

            if entity_text and entity_type in ENTITY_TYPES:
                # Tokenize the entity
                entity_tokens = word_tokenize(entity_text, language='portuguese')

                # Find this sequence in the chunk tokens
                entity_len = len(entity_tokens)
                for i in range(len(tokens) - entity_len + 1):
                    # Check if tokens match
                    if tokens[i:i+entity_len] == entity_tokens:
                        # Tag with BIO
                        bio_tags[i] = f'B-{entity_type.upper()}'
                        for j in range(1, entity_len):
                            bio_tags[i+j] = f'I-{entity_type.upper()}'
                        break

    return bio_tags


def extract_entities_from_bio(bio_path, chunk_number, chunk_size):
    """
    Extract gold BIO tags for a specific chunk.
    Returns list of BIO tags.
    """
    with open(bio_path, "r", encoding="utf-8") as f:
        lines = [line.strip().split() for line in f.readlines() if line.strip()]

    all_tokens = [line[0] for line in lines]
    all_tags = [line[1] if len(line) > 1 else 'O' for line in lines]

    # Get chunk boundaries
    text = ' '.join(all_tokens)
    chunks = chunk_text(text, chunk_size)

    if chunk_number > len(chunks):
        return None, None

    chunk = chunks[chunk_number - 1]
    chunk_start_char = sum(len(chunks[i]) + 1 for i in range(chunk_number - 1))
    chunk_end_char = chunk_start_char + len(chunk)

    # Find token indices for this chunk
    current_char = 0
    chunk_start_idx = None
    chunk_end_idx = None

    for i, token in enumerate(all_tokens):
        if current_char >= chunk_start_char and chunk_start_idx is None:
            chunk_start_idx = i

        current_char += len(token)

        if current_char >= chunk_end_char and chunk_end_idx is None:
            chunk_end_idx = i + 1
            break

        current_char += 1

    if chunk_start_idx is None:
        chunk_start_idx = 0
    if chunk_end_idx is None:
        chunk_end_idx = len(all_tokens)

    # Extract BIO tags for this chunk
    chunk_tags = all_tags[chunk_start_idx:chunk_end_idx]
    chunk_text_content = chunk

    return chunk_tags, chunk_text_content


def precision_recall_f1_per_type(y_true, y_pred, entity_types):
    """
    Calculate precision, recall, F1 per entity type using LeNER-Br methodology.
    """
    results = OrderedDict((tag, OrderedDict()) for tag in entity_types)
    n_tokens = len(y_true)
    total_correct = 0

    for tag in entity_types:
        tag_upper = tag.upper()
        true_chunk = list()
        predicted_chunk = list()

        for position in range(n_tokens):
            _update_chunk(y_true[position], y_true[position - 1] if position > 0 else 'O',
                         tag_upper, true_chunk, position)
            _update_chunk(y_pred[position], y_pred[position - 1] if position > 0 else 'O',
                         tag_upper, predicted_chunk, position, True)

        _update_last_chunk(true_chunk, position)
        _update_last_chunk(predicted_chunk, position)

        # Calculate TP, FP, FN
        tp = sum(chunk in predicted_chunk for chunk in true_chunk)
        total_correct += tp
        fn = len(true_chunk) - tp
        fp = len(predicted_chunk) - tp

        precision, recall, f1 = _tag_precision_recall_f1(tp, fp, fn)

        results[tag]['precision'] = precision
        results[tag]['recall'] = recall
        results[tag]['f1'] = f1
        results[tag]['tp'] = tp
        results[tag]['fp'] = fp
        results[tag]['fn'] = fn
        results[tag]['n_predicted_entities'] = len(predicted_chunk)
        results[tag]['n_true_entities'] = len(true_chunk)

    return results, total_correct


def evaluate_chunk(chunk_path, bio_path, chunk_number, chunk_size):
    """Evaluate a single chunk"""
    clean_output = chunk_path / "clean_output.html"

    # Extract gold tags and chunk text
    gold_tags, chunk_text_content = extract_entities_from_bio(bio_path, chunk_number, chunk_size)

    if gold_tags is None or chunk_text_content is None:
        return None

    # Extract predicted tags
    pred_tags = extract_entities_from_html_as_bio(clean_output, chunk_text_content)

    if pred_tags is None:
        # No predictions, all O tags
        pred_tags = ['O'] * len(gold_tags)

    # Ensure same length
    if len(pred_tags) != len(gold_tags):
        # Pad or truncate
        if len(pred_tags) < len(gold_tags):
            pred_tags.extend(['O'] * (len(gold_tags) - len(pred_tags)))
        else:
            pred_tags = pred_tags[:len(gold_tags)]

    # Calculate metrics using LeNER-Br methodology
    results, total_correct = precision_recall_f1_per_type(gold_tags, pred_tags, ENTITY_TYPES)

    return results


def generate_reports(global_stats, output_dir):
    """Generate classification report and confusion matrix"""
    df_confusion_matrix = pd.DataFrame.from_dict(global_stats, orient="index")
    df_confusion_matrix.to_csv(output_dir / "confusion_matrix.csv", encoding="utf-8", index_label="entity")

    # Build arrays for sklearn classification_report
    y_true = []
    y_pred = []

    for ent_type, stats in global_stats.items():
        TP = stats["tp"]
        FP = stats["fp"]
        FN = stats["fn"]

        # True positives
        y_true += [ent_type] * TP
        y_pred += [ent_type] * TP

        # False positives
        y_true += ["__other__"] * FP
        y_pred += [ent_type] * FP

        # False negatives
        y_true += [ent_type] * FN
        y_pred += ["__other__"] * FN

    labels = sorted(global_stats.keys())
    report_text = classification_report(y_true, y_pred, labels=labels, zero_division=0, digits=3)

    with open(output_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    report_dict = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    df_report = pd.DataFrame(report_dict).transpose()

    # Try to create plot (may fail if no data)
    try:
        metrics_df = df_report.drop(index=["micro avg", "macro avg", "weighted avg"], errors='ignore')
        metrics_df = metrics_df.reset_index().rename(columns={"index": "entity"})
        metrics_df = metrics_df[["entity", "precision", "recall", "f1-score"]]

        plt.figure(figsize=(12, 10))
        x = range(len(metrics_df))
        bar_width = 0.25

        plt.bar([i - bar_width for i in x], metrics_df["precision"], width=bar_width, label="Precision")
        plt.bar(x, metrics_df["recall"], width=bar_width, label="Recall")
        plt.bar([i + bar_width for i in x], metrics_df["f1-score"], width=bar_width, label="F1 Score")

        plt.xticks(ticks=x, labels=metrics_df["entity"], rotation=90)
        plt.ylabel("Score")
        plt.ylim(0, 1.05)
        plt.title("Entity-Level Evaluation Results (LeNER-Br Methodology)")
        plt.legend()
        plt.tight_layout()

        plt.savefig(output_dir / "evaluation_plot.png", dpi=300)
        plt.close()
    except Exception as e:
        print(f"Warning: Could not generate plot: {e}")


def evaluate_results(result_dir: Path, data_dir: Path, chunk_size: int, log_file: Path = None):
    """
    Evaluate NER results using LeNER-Br exact match methodology.

    Args:
        result_dir: Directory containing model outputs
        data_dir: Directory containing gold standard .bio files
        chunk_size: Maximum chunk size used during annotation
        log_file: Optional log file path

    Returns:
        Path to evaluation directory
    """
    log("Starting entity-level evaluation (LeNER-Br methodology)", log_file)

    outputs_dir = result_dir / "outputs"
    evaluation_dir = result_dir / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    # Initialize global stats
    global_stats = {
        entity_type: {'tp': 0, 'fp': 0, 'fn': 0, 'precision': 0, 'recall': 0, 'f1': 0}
        for entity_type in ENTITY_TYPES
    }

    # Process all documents
    for doc_dir in sorted(outputs_dir.iterdir()):
        if not doc_dir.is_dir():
            continue

        document_name = doc_dir.name
        bio_path = data_dir / f"{document_name}.bio"

        if not bio_path.exists():
            log(f"Warning: Gold standard not found for {document_name}", log_file)
            continue

        log(f"Evaluating document: {document_name}", log_file)

        chunk_dirs = sorted([d for d in doc_dir.iterdir() if d.is_dir() and d.name.startswith("chunk")],
                           key=lambda x: int(x.name.replace("chunk", "")))

        for chunk_dir in chunk_dirs:
            chunk_number = int(chunk_dir.name.replace("chunk", ""))
            results = evaluate_chunk(chunk_dir, bio_path, chunk_number, chunk_size)

            if results is None:
                continue

            # Aggregate results
            for entity_type in ENTITY_TYPES:
                global_stats[entity_type]['tp'] += results[entity_type]['tp']
                global_stats[entity_type]['fp'] += results[entity_type]['fp']
                global_stats[entity_type]['fn'] += results[entity_type]['fn']

    # Calculate final metrics
    for entity_type in ENTITY_TYPES:
        tp = global_stats[entity_type]['tp']
        fp = global_stats[entity_type]['fp']
        fn = global_stats[entity_type]['fn']
        precision, recall, f1 = _tag_precision_recall_f1(tp, fp, fn)
        global_stats[entity_type]['precision'] = precision
        global_stats[entity_type]['recall'] = recall
        global_stats[entity_type]['f1'] = f1

    # Generate reports
    generate_reports(global_stats, evaluation_dir)

    log(f"Evaluation results saved to: {evaluation_dir}", log_file)
    log(f"  - classification_report.txt", log_file)
    log(f"  - confusion_matrix.csv", log_file)
    log(f"  - evaluation_plot.png", log_file)

    return evaluation_dir
