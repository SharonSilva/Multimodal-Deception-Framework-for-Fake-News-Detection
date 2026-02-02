"""
BATCH PROCESSOR - Command Line Utility
======================================
Process multiple posts from CSV files without running the full web server.
"""

import argparse
import csv
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from standalone_detector import StandaloneFakeNewsDetector


def process_batch_file(input_csv, output_dir='batch_results', save_individual=False):
    """
    Process a batch of posts from CSV file
    
    Args:
        input_csv: Path to input CSV file
        output_dir: Directory to save results
        save_individual: Whether to save individual results (in addition to batch file)
    """
    print("="*70)
    print("BATCH FAKE NEWS DETECTOR")
    print("="*70)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Initialize detector
    print("\n🚀 Initializing detector...")
    detector = StandaloneFakeNewsDetector()
    
    # Read CSV
    print(f"\n📂 Reading CSV: {input_csv}")
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"   Found {len(rows)} posts to analyze\n")
    
    # Process each row
    results = []
    stats = {'total': 0, 'fake': 0, 'real': 0, 'unknown': 0}
    
    for row in tqdm(rows, desc="Processing", unit="post"):
        text = row.get('text', '').strip()
        if not text:
            continue
        
        username = row.get('username', 'anonymous')
        image_path = row.get('image_path', None)
        
        # Validate image path
        if image_path and not Path(image_path).exists():
            print(f"   ⚠️  Image not found: {image_path}")
            image_path = None
        
        # Run prediction
        try:
            result = detector.predict(
                text=text,
                image_path=image_path,
                username=username
            )
            result['timestamp'] = datetime.now().isoformat()
            
            results.append(result)
            stats['total'] += 1
            stats[result['verdict'].lower()] = stats.get(result['verdict'].lower(), 0) + 1
            
            # Save individual result if requested
            if save_individual:
                individual_file = output_path / f"{result['post_id']}.json"
                with open(individual_file, 'w') as f:
                    json.dump(result, f, indent=2)
            
        except Exception as e:
            print(f"\n   ❌ Error processing post: {str(e)}")
            continue
    
    # Save batch results
    batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_file = output_path / f"batch_{batch_id}.json"
    
    batch_data = {
        'batch_id': batch_id,
        'timestamp': datetime.now().isoformat(),
        'input_file': str(input_csv),
        'results': results,
        'statistics': stats
    }
    
    with open(batch_file, 'w') as f:
        json.dump(batch_data, f, indent=2)
    
    # Print summary
    print("\n" + "="*70)
    print("BATCH COMPLETE")
    print("="*70)
    print(f"\n📊 Statistics:")
    print(f"   Total Processed: {stats['total']}")
    print(f"   Fake Detected:   {stats.get('fake', 0)} ({stats.get('fake', 0)/stats['total']*100:.1f}%)")
    print(f"   Real Content:    {stats.get('real', 0)} ({stats.get('real', 0)/stats['total']*100:.1f}%)")
    print(f"   Unknown:         {stats.get('unknown', 0)} ({stats.get('unknown', 0)/stats['total']*100:.1f}%)")
    print(f"\n💾 Results saved to: {batch_file}")
    
    if save_individual:
        print(f"   Individual files: {output_path}/")
    
    print("\n" + "="*70)
    
    return batch_data


def generate_report(batch_file, output_format='text'):
    """
    Generate a human-readable report from batch results
    
    Args:
        batch_file: Path to batch JSON file
        output_format: 'text', 'html', or 'markdown'
    """
    with open(batch_file, 'r') as f:
        batch_data = json.load(f)
    
    stats = batch_data['statistics']
    results = batch_data['results']
    
    if output_format == 'text':
        report = f"""
FAKE NEWS DETECTION REPORT
{'='*70}

Batch ID: {batch_data['batch_id']}
Generated: {batch_data['timestamp']}
Input File: {batch_data['input_file']}

SUMMARY STATISTICS
{'='*70}
Total Posts Analyzed: {stats['total']}
Fake News Detected:   {stats.get('fake', 0)} ({stats.get('fake', 0)/stats['total']*100:.1f}%)
Real Content:         {stats.get('real', 0)} ({stats.get('real', 0)/stats['total']*100:.1f}%)
Unknown:              {stats.get('unknown', 0)} ({stats.get('unknown', 0)/stats['total']*100:.1f}%)

DETAILED RESULTS
{'='*70}
"""
        
        for i, result in enumerate(results, 1):
            report += f"""
Post #{i} - {result['post_id']}
{'-'*70}
Verdict:    {result['verdict']}
Score:      {result['score']:.4f}
Confidence: {result['confidence']*100:.1f}%
Username:   @{result['username']}
Text:       {result['text'][:100]}...

Emotion (VAD):
  Valence:   {result['vad_analysis']['valence']:.2f}
  Arousal:   {result['vad_analysis']['arousal']:.2f}
  Dominance: {result['vad_analysis']['dominance']:.2f}

Content Metadata:
  Hashtags: {result['metadata']['hashtags']}
  Mentions: {result['metadata']['mentions']}
  URLs:     {result['metadata']['urls']}
  Emojis:   {result['metadata']['emojis']}

"""
        
        return report
    
    elif output_format == 'markdown':
        report = f"""# Fake News Detection Report

**Batch ID:** {batch_data['batch_id']}  
**Generated:** {batch_data['timestamp']}  
**Input File:** {batch_data['input_file']}

## Summary Statistics

| Metric | Count | Percentage |
|--------|-------|------------|
| Total Posts | {stats['total']} | 100% |
| Fake News | {stats.get('fake', 0)} | {stats.get('fake', 0)/stats['total']*100:.1f}% |
| Real Content | {stats.get('real', 0)} | {stats.get('real', 0)/stats['total']*100:.1f}% |
| Unknown | {stats.get('unknown', 0)} | {stats.get('unknown', 0)/stats['total']*100:.1f}% |

## Detailed Results

"""
        
        for i, result in enumerate(results, 1):
            verdict_emoji = "🚨" if result['verdict'] == 'FAKE' else "✅" if result['verdict'] == 'REAL' else "🟡"
            
            report += f"""
### Post #{i} - {result['post_id']} {verdict_emoji}

**Verdict:** {result['verdict']}  
**Score:** {result['score']:.4f}  
**Confidence:** {result['confidence']*100:.1f}%  
**Username:** @{result['username']}

**Text Preview:**  
> {result['text'][:200]}...

**Emotion Analysis (VAD):**
- Valence: {result['vad_analysis']['valence']:.2f}
- Arousal: {result['vad_analysis']['arousal']:.2f}
- Dominance: {result['vad_analysis']['dominance']:.2f}

**Content Metadata:**
- Hashtags: {result['metadata']['hashtags']}
- Mentions: {result['metadata']['mentions']}
- URLs: {result['metadata']['urls']}
- Emojis: {result['metadata']['emojis']}

---

"""
        
        return report


def main():
    parser = argparse.ArgumentParser(
        description='Batch process posts for fake news detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a CSV file
  python batch_processor.py posts.csv
  
  # Process with custom output directory
  python batch_processor.py posts.csv --output results/
  
  # Save individual result files
  python batch_processor.py posts.csv --save-individual
  
  # Generate a report
  python batch_processor.py --report batch_results/batch_20240201_103045.json
  
  # Generate markdown report
  python batch_processor.py --report batch.json --format markdown > report.md
        """
    )
    
    parser.add_argument('input', nargs='?', help='Input CSV file to process')
    parser.add_argument('-o', '--output', default='batch_results', 
                       help='Output directory (default: batch_results)')
    parser.add_argument('-s', '--save-individual', action='store_true',
                       help='Save individual result files')
    parser.add_argument('-r', '--report', help='Generate report from batch file')
    parser.add_argument('-f', '--format', choices=['text', 'markdown', 'html'],
                       default='text', help='Report format (default: text)')
    
    args = parser.parse_args()
    
    if args.report:
        # Generate report mode
        report = generate_report(args.report, args.format)
        print(report)
    elif args.input:
        # Batch processing mode
        process_batch_file(
            input_csv=args.input,
            output_dir=args.output,
            save_individual=args.save_individual
        )
    else:
        parser.print_help()


if __name__ == '__main__':
    main()