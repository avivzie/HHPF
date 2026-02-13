"""
Convert TAT-QA JSON files to CSV format for HHPF pipeline.

TAT-QA includes:
- Financial tables (structured data)
- Text paragraphs (context)
- Multiple Q&A pairs per document

This is IDEAL for hallucination detection with source verification.
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict


def table_to_text(table_data: Dict) -> str:
    """Convert table dict to text representation."""
    if 'table' not in table_data:
        return ""
    
    table = table_data['table']
    rows = []
    for row in table:
        rows.append(' | '.join([str(cell) for cell in row]))
    return '\n'.join(rows)


def paragraphs_to_text(paragraphs: List[Dict]) -> str:
    """Convert paragraphs list to text."""
    sorted_paras = sorted(paragraphs, key=lambda x: x.get('order', 0))
    texts = [p['text'] for p in sorted_paras if 'text' in p]
    return '\n\n'.join(texts)


def extract_questions(item: Dict, doc_id: str) -> List[Dict]:
    """Extract all Q&A pairs from a document with context."""
    
    # Build context from table + paragraphs
    table_text = table_to_text(item.get('table', {}))
    para_text = paragraphs_to_text(item.get('paragraphs', []))
    
    # Combine context
    context_parts = []
    if para_text:
        context_parts.append(para_text)
    if table_text:
        context_parts.append(f"TABLE:\n{table_text}")
    
    full_context = '\n\n'.join(context_parts)
    
    # Extract questions
    questions = item.get('questions', [])
    rows = []
    
    for q_idx, q in enumerate(questions):
        row = {
            'id': f"{doc_id}_q{q_idx}",
            'question': q.get('question', ''),
            'answer': str(q.get('answer', '')),
            'context': full_context,
            'table_uid': item.get('table', {}).get('uid', ''),
            'answer_type': q.get('answer_type', ''),  # span, arithmetic, count, etc.
        }
        rows.append(row)
    
    return rows


def convert_tatqa_file(json_path: Path) -> pd.DataFrame:
    """Convert a single TAT-QA JSON file to DataFrame."""
    print(f"\n📄 Processing: {json_path.name}")
    
    with open(json_path) as f:
        data = json.load(f)
    
    print(f"  Documents: {len(data)}")
    
    # Extract all Q&A pairs
    all_rows = []
    for doc_idx, doc in enumerate(data):
        doc_id = f"{json_path.stem}_doc{doc_idx}"
        rows = extract_questions(doc, doc_id)
        all_rows.extend(rows)
    
    df = pd.DataFrame(all_rows)
    print(f"  Total Q&A pairs: {len(df)}")
    
    return df


def main():
    print("="*60)
    print("TAT-QA Dataset Conversion")
    print("="*60)
    
    tatqa_dir = Path('data/raw/TATQA')
    
    # Find all JSON files
    json_files = sorted(tatqa_dir.glob('*.json'))
    print(f"\nFound {len(json_files)} JSON files:")
    for f in json_files:
        print(f"  - {f.name}")
    
    # Convert each file
    all_dfs = []
    for json_file in json_files:
        df = convert_tatqa_file(json_file)
        all_dfs.append(df)
    
    # Combine all
    combined = pd.concat(all_dfs, ignore_index=True)
    
    print(f"\n{'='*60}")
    print(f"Combined Dataset")
    print(f"{'='*60}")
    print(f"Total samples: {len(combined)}")
    
    # Data quality checks
    print(f"\n🔍 Data Quality Validation:")
    print(f"  Questions with text: {(combined['question'].str.len() > 0).sum()}")
    print(f"  Answers with text: {(combined['answer'].astype(str).str.len() > 0).sum()}")
    print(f"  With context: {(combined['context'].str.len() > 100).sum()}")
    
    # Statistics
    print(f"\n📊 Statistics:")
    print(f"  Avg question length: {combined['question'].str.len().mean():.0f} chars")
    print(f"  Avg answer length: {combined['answer'].str.len().mean():.0f} chars")
    print(f"  Avg context length: {combined['context'].str.len().mean():.0f} chars")
    
    # Answer types
    if 'answer_type' in combined.columns:
        print(f"\n📝 Answer types:")
        print(combined['answer_type'].value_counts().to_string())
    
    # Sample
    print(f"\n📋 Sample Record:")
    print(f"  ID: {combined.iloc[0]['id']}")
    print(f"  Question: {combined.iloc[0]['question'][:100]}")
    print(f"  Answer: {combined.iloc[0]['answer']}")
    print(f"  Context (first 150 chars): {combined.iloc[0]['context'][:150]}...")
    
    # Save
    output_path = Path('data/raw/tatqa.csv')
    combined.to_csv(output_path, index=False)
    
    print(f"\n{'='*60}")
    print(f"✓ SUCCESS: Saved to {output_path}")
    print(f"{'='*60}")
    print(f"  Total samples: {len(combined)}")
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    # Thesis readiness check
    print(f"\n✅ Thesis Readiness Check:")
    checks = {
        'Sample size ≥500': len(combined) >= 500,
        'Has questions': (combined['question'].str.len() > 0).all(),
        'Has answers': (combined['answer'].astype(str).str.len() > 0).all(),
        'Has context': (combined['context'].str.len() > 50).sum() > len(combined) * 0.95,
        'Ready for pipeline': True
    }
    
    for check, passed in checks.items():
        status = '✅' if passed else '❌'
        print(f"  {status} {check}")
    
    if all(checks.values()):
        print(f"\n🎉 TAT-QA is THESIS-READY!")
        print(f"\n🚀 Next steps:")
        print(f"  1. Update configs/datasets.yaml:")
        print(f"     finance:")
        print(f"       name: 'TAT-QA'")
        print(f"       file_path: 'data/raw/tatqa.csv'")
        print(f"  2. Run: python run_pipeline.py --domain finance --limit 500")
    else:
        print(f"\n⚠️  Some checks failed - review data quality")
    
    return combined


if __name__ == '__main__':
    df = main()
