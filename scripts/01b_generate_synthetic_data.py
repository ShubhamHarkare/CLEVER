#!/usr/bin/env python3
"""
01b_generate_synthetic_data.py — Generate synthetic query data for local testing.

Use this to develop and test the pipeline when LMSYS-Chat-1M access is pending.
Generates realistic-looking queries across multiple topics.

Usage:
    python scripts/01b_generate_synthetic_data.py --output data/ --num-queries 10000
"""

import logging
import random
import sys
from pathlib import Path

import click
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.preprocessor import preprocess_queries, save_processed_queries
from src.data.sampler import create_subsets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Topic templates for generating diverse, realistic queries
QUERY_TEMPLATES = {
    "coding": [
        "How do I {action} in {language}?",
        "What is the difference between {concept_a} and {concept_b} in {language}?",
        "Write a {language} function to {task}",
        "Explain {concept} in {language} with examples",
        "How to fix {error} error in {language}?",
        "Best practices for {topic} in {language}",
        "Can you help me debug this {language} code that {problem}?",
        "What is the time complexity of {algorithm}?",
    ],
    "science": [
        "What is {concept} and how does it work?",
        "Explain the process of {process} in simple terms",
        "What are the main differences between {thing_a} and {thing_b}?",
        "How does {phenomenon} affect {subject}?",
        "What are the latest discoveries in {field}?",
        "Can you explain {theory} theory to a beginner?",
        "What causes {phenomenon} to occur?",
        "How is {technology} used in {application}?",
    ],
    "writing": [
        "Write a short story about {topic}",
        "Help me write an email to {recipient} about {subject}",
        "Can you proofread this paragraph and suggest improvements?",
        "Write a poem about {topic}",
        "Help me create a cover letter for a {job} position",
        "Summarize the key points of {topic}",
        "Write a persuasive essay about {topic}",
        "How do I improve my writing about {subject}?",
    ],
    "general": [
        "What are the best {items} for {purpose}?",
        "How do I {task} step by step?",
        "What should I know about {topic} before {action}?",
        "Can you compare {option_a} and {option_b}?",
        "What are the pros and cons of {topic}?",
        "Recommend some {category} for {audience}",
        "How much does {item} typically cost?",
        "What is the history of {topic}?",
    ],
    "math": [
        "Solve the equation {equation}",
        "What is the integral of {function}?",
        "Explain {concept} in linear algebra",
        "How do I calculate {metric} for {context}?",
        "What is the probability of {event}?",
        "Prove that {statement}",
        "Help me understand {topic} in statistics",
        "What is the difference between {concept_a} and {concept_b} in math?",
    ],
}

FILL_VALUES = {
    "language": ["Python", "JavaScript", "Java", "C++", "Rust", "Go", "TypeScript", "Ruby"],
    "action": ["sort a list", "read a file", "connect to a database", "handle errors", "parse JSON",
               "create a class", "use async/await", "implement a REST API"],
    "concept": ["polymorphism", "recursion", "closures", "decorators", "generators",
                "photosynthesis", "quantum entanglement", "machine learning", "neural networks"],
    "concept_a": ["a list", "TCP", "supervised learning", "stack", "HTTP", "mean"],
    "concept_b": ["a tuple", "UDP", "unsupervised learning", "queue", "HTTPS", "median"],
    "task": ["sort an array", "find prime numbers", "reverse a string", "build a calculator",
             "implement binary search", "create a linked list"],
    "error": ["IndexOutOfBounds", "NullPointer", "TypeError", "SegFault", "MemoryLeak"],
    "topic": ["artificial intelligence", "climate change", "space exploration", "renewable energy",
              "blockchain", "cybersecurity", "data structures", "web development",
              "ancient civilizations", "modern art", "cooking techniques", "fitness training"],
    "algorithm": ["quicksort", "binary search", "BFS", "DFS", "dijkstra's algorithm", "merge sort"],
    "process": ["photosynthesis", "cellular respiration", "DNA replication", "protein synthesis"],
    "thing_a": ["mitosis", "DNA", "classical physics", "bacteria"],
    "thing_b": ["meiosis", "RNA", "quantum physics", "viruses"],
    "phenomenon": ["gravity", "magnetism", "climate change", "evolution"],
    "subject": ["the environment", "human health", "technology", "society"],
    "field": ["astronomy", "genetics", "neuroscience", "materials science"],
    "theory": ["relativity", "evolution", "string", "game"],
    "technology": ["CRISPR", "AI", "blockchain", "3D printing"],
    "application": ["medicine", "agriculture", "manufacturing", "education"],
    "recipient": ["my professor", "my manager", "a client", "HR department"],
    "job": ["software engineer", "data scientist", "product manager", "researcher"],
    "items": ["books", "tools", "resources", "apps", "courses"],
    "purpose": ["learning Python", "starting a business", "personal finance", "home improvement"],
    "option_a": ["buying", "renting", "freelancing", "a Mac"],
    "option_b": ["leasing", "owning", "full-time employment", "a PC"],
    "category": ["movies", "podcasts", "books", "documentaries"],
    "audience": ["beginners", "teenagers", "professionals", "families"],
    "item": ["a laptop", "a house", "college tuition", "a used car"],
    "equation": ["x^2 + 3x - 4 = 0", "2x + 5 = 15", "e^x = 10"],
    "function": ["x^2 * sin(x)", "e^(-x^2)", "ln(x)/x"],
    "metric": ["standard deviation", "variance", "correlation", "the mean"],
    "context": ["a dataset", "a sample", "a population", "stock returns"],
    "event": ["rolling a 6 twice", "drawing two aces", "rain tomorrow"],
    "statement": ["sqrt(2) is irrational", "the sum of angles is 180", "there are infinite primes"],
    "problem": ["keeps crashing", "returns wrong output", "is too slow", "has a memory leak"],
}


def generate_query(rng: random.Random) -> str:
    """Generate a single synthetic query."""
    category = rng.choice(list(QUERY_TEMPLATES.keys()))
    template = rng.choice(QUERY_TEMPLATES[category])

    # Fill in placeholders
    result = template
    for key, values in FILL_VALUES.items():
        placeholder = "{" + key + "}"
        while placeholder in result:
            result = result.replace(placeholder, rng.choice(values), 1)

    return result


def generate_synthetic_queries(num_queries: int, seed: int = 42) -> pd.DataFrame:
    """Generate a DataFrame of synthetic queries."""
    rng = random.Random(seed)

    queries = []
    for i in range(num_queries):
        query = generate_query(rng)
        queries.append({
            "query_id": i,
            "query_text": query,
            "original_index": i,
        })

    # Add some intentional duplicates (~5%)
    num_dupes = int(num_queries * 0.05)
    for i in range(num_dupes):
        source = rng.choice(queries)
        queries.append({
            "query_id": num_queries + i,
            "query_text": source["query_text"],
            "original_index": num_queries + i,
        })

    rng.shuffle(queries)
    df = pd.DataFrame(queries)
    df["query_id"] = range(len(df))
    return df


@click.command()
@click.option("--output", type=click.Path(), default="data/",
              help="Output directory.")
@click.option("--num-queries", type=int, default=10000,
              help="Number of queries to generate.")
@click.option("--seed", type=int, default=42,
              help="Random seed.")
@click.option("--skip-subsets", is_flag=True, default=False,
              help="Skip creating scale subsets.")
def main(output, num_queries, seed, skip_subsets):
    """Generate synthetic query data for local development and testing."""
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Generate synthetic queries
    logger.info("=" * 60)
    logger.info(f"Step 1: Generating {num_queries} synthetic queries (seed={seed})")
    logger.info("=" * 60)
    raw_df = generate_synthetic_queries(num_queries, seed=seed)
    raw_path = output_dir / "raw_queries.parquet"
    raw_df.to_parquet(raw_path, index=False)
    logger.info(f"Generated {len(raw_df)} queries (including ~5% duplicates)")

    # Step 2: Preprocess
    logger.info("=" * 60)
    logger.info("Step 2: Preprocessing queries")
    logger.info("=" * 60)
    processed_df = preprocess_queries(raw_df)
    processed_path = output_dir / "processed_queries.parquet"
    save_processed_queries(processed_df, processed_path)

    # Step 3: Create subsets
    if not skip_subsets:
        logger.info("=" * 60)
        logger.info("Step 3: Creating scale subsets")
        logger.info("=" * 60)
        create_subsets(processed_df, output_dir=output_dir, seed=seed)

    # Summary
    logger.info("=" * 60)
    logger.info("DONE — Synthetic Data Summary")
    logger.info("=" * 60)
    logger.info(f"  Raw queries:       {len(raw_df)}")
    logger.info(f"  After processing:  {len(processed_df)}")
    logger.info(f"  Output directory:  {output_dir}")
    logger.info("")
    logger.info("Next step: python scripts/02_generate_embeddings.py \\")
    logger.info(f"    --input {processed_path} --output results/embeddings/ \\")
    logger.info("    --device cpu --batch-size 64 --sizes 10k")


if __name__ == "__main__":
    main()
