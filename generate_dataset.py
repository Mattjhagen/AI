#!/usr/bin/env python3
"""
Generate training datasets from Shaggoth knowledge base for DeepSeek-R1 fine-tuning.
Creates instruction-response pairs in JSONL format from knowledge files.
"""

import json
import os
import re
from pathlib import Path
from typing import List, Dict
import random

# Knowledge base path
KNOWLEDGE_DIR = Path("/home/matt/Shaggoth-a1/data/knowledge")
OUTPUT_DIR = Path("/home/matt/AI/datasets")

def clean_topic_name(filename: str) -> str:
    """Convert filename to readable topic name"""
    name = filename.replace('.md', '').replace('-', ' ').replace('_', ' ')
    # Remove "part-1" etc suffixes
    name = re.sub(r'\s+part\s+\d+$', '', name, flags=re.IGNORECASE)
    return name.title()

def read_knowledge_file(filepath: Path) -> str:
    """Read and clean knowledge file content"""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read().strip()
        # Remove markdown artifacts
        content = re.sub(r'\[edit\]', '', content)
        content = re.sub(r'\s+', ' ', content)
        return content
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return ""

def generate_instruction_variants(topic: str) -> List[str]:
    """Generate diverse instruction prompts for a topic"""
    instructions = [
        f"Explain {topic}",
        f"What is {topic}?",
        f"Describe {topic}",
        f"Tell me about {topic}",
        f"Give me an overview of {topic}",
        f"Provide information about {topic}",
        f"Can you explain {topic}?",
        f"What do you know about {topic}?",
        f"Help me understand {topic}",
        f"Teach me about {topic}",
    ]
    return instructions

def create_training_example(topic: str, content: str, instruction_type: str = "explain") -> Dict:
    """Create a single training example"""
    instructions = generate_instruction_variants(topic)

    if instruction_type == "explain":
        instruction = random.choice(instructions)
    elif instruction_type == "definition":
        instruction = f"What is {topic}?"
    elif instruction_type == "detailed":
        instruction = f"Provide a detailed explanation of {topic}"
    elif instruction_type == "simple":
        instruction = f"Explain {topic} in simple terms"
    else:
        instruction = random.choice(instructions)

    return {
        "instruction": instruction,
        "input": "",
        "output": content
    }

def create_reasoning_example(topic: str, content: str) -> Dict:
    """Create reasoning/analytical training example"""
    reasoning_prompts = [
        f"Why is {topic} important?",
        f"What are the key concepts in {topic}?",
        f"How does {topic} work?",
        f"What are the applications of {topic}?",
        f"Explain the significance of {topic}",
    ]

    # Extract first few sentences for reasoning response
    sentences = content.split('. ')[:3]
    reasoning_response = '. '.join(sentences) + '.'

    return {
        "instruction": random.choice(reasoning_prompts),
        "input": "",
        "output": reasoning_response
    }

def create_comparison_example(topics: List[tuple]) -> Dict:
    """Create comparison training examples between related topics"""
    if len(topics) < 2:
        return None

    topic1_name, topic1_content = topics[0]
    topic2_name, topic2_content = topics[1]

    comparison_response = f"Let me compare {topic1_name} and {topic2_name}. "
    comparison_response += f"{topic1_name}: {topic1_content[:200]}... "
    comparison_response += f"{topic2_name}: {topic2_content[:200]}..."

    return {
        "instruction": f"Compare {topic1_name} and {topic2_name}",
        "input": "",
        "output": comparison_response
    }

def create_shaggoth_specific_examples() -> List[Dict]:
    """Create Shaggoth-specific training examples"""
    examples = [
        {
            "instruction": "What is Shaggoth AI?",
            "input": "",
            "output": "Shaggoth is a self-hosted conversational AI platform that runs on your own hardware with zero required dependencies. It features a knowledge-first answering system, autonomous learning agents, persistent memory, and hybrid inference supporting both local models (Markov, stdlib) and cloud models (Gemini, Cloudflare Workers AI, OpenAI). The system includes onboard AI agents for research, curation, and training, all running locally and optionally."
        },
        {
            "instruction": "How does Shaggoth's knowledge base work?",
            "input": "",
            "output": "Shaggoth uses a BM25-ranked knowledge base built from web research. When asked 'what is X', it retrieves the real definition from its knowledge base with attribution. If it doesn't know something, it explicitly says so and can autonomously research the topic to expand its knowledge. The knowledge base supports feedback loops where thumbs up/down reactions implicate specific entries for re-research before age-based updates."
        },
        {
            "instruction": "Explain Shaggoth's autonomous learning system",
            "input": "",
            "output": "Shaggoth includes onboard AI agents that run alongside the server: a researcher that converts conversations into research topics, a grader that self-evaluates past answers during idle time, a curator that maintains knowledge base quality, a gatherer that reads crawl-permissive sources, and a trainer that retrains the model behind a quality gate. All agents are local, free, and optional, enabling continuous self-improvement without external dependencies."
        },
        {
            "instruction": "What models does Shaggoth support?",
            "input": "",
            "output": "Shaggoth supports hybrid inference with multiple model options: local inference using Markov chains and Python standard library (zero dependencies), free-tier cloud models including Gemini and Cloudflare Workers AI (using plain urllib, no SDK), paid OpenAI backends, and now fine-tuned DeepSeek-R1 for advanced reasoning tasks. All backends are swappable through a unified interface."
        },
        {
            "instruction": "How is Shaggoth deployed?",
            "input": "",
            "output": "Shaggoth runs on AWS EC2 (production API on t3.small), with a Dell R510 homelab for development running Ollama and OpenCode. The production setup uses Cloudflare tunnels for secure access at ai.relayapp.pro, includes mobile apps for iOS and Android, and features an orbital command center dashboard for monitoring. The system is designed to run identically on a laptop, cloud sandbox, or rack server."
        }
    ]
    return examples

def generate_datasets():
    """Main dataset generation function"""
    OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"📚 Reading knowledge base from {KNOWLEDGE_DIR}")

    # Collect all knowledge files
    knowledge_files = list(KNOWLEDGE_DIR.glob("*.md"))
    print(f"Found {len(knowledge_files)} knowledge files")

    # Different dataset types
    datasets = {
        "general": [],
        "reasoning": [],
        "shaggoth": create_shaggoth_specific_examples(),
        "mixed": []
    }

    topic_contents = []

    # Process each knowledge file
    for i, filepath in enumerate(knowledge_files[:500], 1):  # Limit to 500 for initial dataset
        if i % 50 == 0:
            print(f"  Processed {i}/{min(500, len(knowledge_files))} files...")

        topic = clean_topic_name(filepath.name)
        content = read_knowledge_file(filepath)

        if not content or len(content) < 50:
            continue

        topic_contents.append((topic, content))

        # Generate multiple example types
        datasets["general"].append(create_training_example(topic, content, "explain"))
        datasets["general"].append(create_training_example(topic, content, "definition"))

        if len(content) > 200:
            datasets["reasoning"].append(create_reasoning_example(topic, content))

    print(f"✓ Processed {len(topic_contents)} valid knowledge entries")

    # Create comparison examples
    print("Creating comparison examples...")
    for i in range(0, min(100, len(topic_contents) - 1), 2):
        comp = create_comparison_example(topic_contents[i:i+2])
        if comp:
            datasets["reasoning"].append(comp)

    # Create mixed dataset
    print("Creating mixed dataset...")
    datasets["mixed"] = (
        datasets["general"][:300] +
        datasets["reasoning"][:100] +
        datasets["shaggoth"]
    )
    random.shuffle(datasets["mixed"])

    # Write datasets
    print("\n💾 Writing datasets...")
    for name, examples in datasets.items():
        if not examples:
            continue

        output_file = OUTPUT_DIR / f"shaggoth_{name}.jsonl"
        with open(output_file, 'w') as f:
            for example in examples:
                f.write(json.dumps(example) + '\n')

        print(f"  ✓ {output_file.name}: {len(examples)} examples")

        # Also create a preview file
        preview_file = OUTPUT_DIR / f"shaggoth_{name}_preview.json"
        with open(preview_file, 'w') as f:
            json.dump(examples[:5], f, indent=2)

    # Summary statistics
    print("\n" + "="*60)
    print("📊 DATASET SUMMARY")
    print("="*60)
    print(f"Total knowledge files processed: {len(topic_contents)}")
    print(f"General knowledge examples: {len(datasets['general'])}")
    print(f"Reasoning examples: {len(datasets['reasoning'])}")
    print(f"Shaggoth-specific examples: {len(datasets['shaggoth'])}")
    print(f"Mixed training dataset: {len(datasets['mixed'])}")
    print(f"\nDatasets saved to: {OUTPUT_DIR}")
    print("="*60)

    return datasets

if __name__ == "__main__":
    print("🚀 Shaggoth Training Dataset Generator")
    print("="*60)
    datasets = generate_datasets()
    print("\n✅ Dataset generation complete!")
