#!/usr/bin/env python3
"""
Generate synthetic structural degeneracy samples for ASV evaluation.

This script creates samples with:
1. Loops: Repeated phrases/sentences
2. Semantic drift: Topic jumping mid-response
3. Incoherence: Contradictory statements
4. Repetition: Same words/patterns without exact loops

Also generates normal samples for contrast.
"""

import json
import random
import sys
from pathlib import Path
from typing import List, Dict, Tuple

# Add agent src to path for reusing utilities
sys.path.insert(0, str(Path(__file__).parent.parent / "agent" / "src"))


class DegeneracyGenerator:
    """Generate synthetic samples with structural degeneracy."""

    def __init__(self, seed: int = 42):
        self.random = random.Random(seed)

        # Base templates for normal text
        self.normal_templates = [
            "The {subject} is a fascinating topic in {field}. It involves {concept1}, {concept2}, and {concept3}. Researchers have found that {finding}. This has important implications for {application}.",
            "Recent advances in {field} have led to significant breakthroughs. For example, {subject} has been shown to {finding}. This discovery opens new possibilities for {application}.",
            "Understanding {subject} requires knowledge of {concept1} and {concept2}. The relationship between these concepts is crucial for {application}. Studies indicate that {finding}.",
            "{subject} plays a vital role in {field}. Scientists have discovered that {concept1} interacts with {concept2} to produce {finding}. This understanding enables {application}.",
            "The study of {subject} has revealed {finding}. This is particularly relevant for {application}, where {concept1} and {concept2} are key factors. Further research in {field} is ongoing.",
        ]

        # Vocabulary for filling templates
        self.vocab = {
            'subject': ['quantum entanglement', 'neural networks', 'photosynthesis', 'gene expression',
                       'climate patterns', 'market dynamics', 'social behavior', 'protein folding'],
            'field': ['physics', 'computer science', 'biology', 'genetics', 'environmental science',
                     'economics', 'psychology', 'biochemistry'],
            'concept1': ['energy transfer', 'pattern recognition', 'feedback loops', 'regulatory mechanisms',
                        'optimization', 'adaptation', 'signal processing', 'structural stability'],
            'concept2': ['temporal dynamics', 'spatial organization', 'information flow', 'resource allocation',
                        'equilibrium states', 'phase transitions', 'network effects', 'emergent properties'],
            'concept3': ['hierarchical organization', 'stochastic processes', 'threshold effects', 'scaling laws',
                        'collective behavior', 'adaptive strategies', 'robustness', 'efficiency tradeoffs'],
            'finding': ['efficiency increases with scale', 'nonlinear relationships exist', 'critical thresholds matter',
                       'network topology affects outcomes', 'randomness plays a key role', 'stability emerges from chaos',
                       'simple rules create complexity', 'diversity enhances resilience'],
            'application': ['predicting future trends', 'optimizing system performance', 'developing new therapies',
                          'designing better algorithms', 'mitigating environmental risks', 'improving decision-making',
                          'engineering robust systems', 'understanding complex phenomena']
        }

        # Sentences for loop generation
        self.loop_sentences = [
            "The process continues iteratively.",
            "This pattern repeats throughout the system.",
            "The cycle begins again.",
            "We observe the same behavior.",
            "The mechanism operates continuously.",
        ]

        # Topic pairs for drift generation (start → end)
        self.drift_topics = [
            ("artificial intelligence", "breakfast cereals"),
            ("quantum mechanics", "gardening tips"),
            ("climate change", "medieval history"),
            ("machine learning", "cooking recipes"),
            ("blockchain technology", "astronomy"),
            ("genetic engineering", "fashion trends"),
        ]

        # Contradictory statement pairs
        self.contradictions = [
            ("The temperature increases significantly.", "The temperature remains constant."),
            ("This approach is highly effective.", "This method has proven ineffective."),
            ("The system is stable.", "The system exhibits chaotic behavior."),
            ("Growth rates are accelerating.", "Growth has stagnated."),
            ("The correlation is strong and positive.", "No correlation was found."),
            ("This is a well-established fact.", "Recent studies have disproven this entirely."),
        ]

    def generate_normal(self, n: int = 500) -> List[Dict]:
        """Generate normal, coherent samples."""
        samples = []
        for i in range(n):
            template = self.random.choice(self.normal_templates)

            # Fill template with vocabulary
            text = template.format(
                subject=self.random.choice(self.vocab['subject']),
                field=self.random.choice(self.vocab['field']),
                concept1=self.random.choice(self.vocab['concept1']),
                concept2=self.random.choice(self.vocab['concept2']),
                concept3=self.random.choice(self.vocab['concept3']),
                finding=self.random.choice(self.vocab['finding']),
                application=self.random.choice(self.vocab['application'])
            )

            samples.append({
                'id': f'normal_{i:04d}',
                'type': 'normal',
                'text': text,
                'is_degenerate': False,
                'metadata': {
                    'degeneracy_type': None,
                    'severity': 0.0
                }
            })

        return samples

    def generate_loops(self, n: int = 125) -> List[Dict]:
        """Generate samples with repetitive loops."""
        samples = []
        for i in range(n):
            # Base sentence + repeated phrase
            base = self.random.choice(self.loop_sentences)
            num_repeats = self.random.randint(10, 50)

            # Create loop with slight variations sometimes
            if self.random.random() < 0.7:  # Exact repetition
                text = " ".join([base] * num_repeats)
                severity = 'high'
            else:  # Slight variations
                variations = [base, base.rstrip('.') + ' again.', base.rstrip('.') + ' once more.']
                text = " ".join([self.random.choice(variations) for _ in range(num_repeats)])
                severity = 'medium'

            samples.append({
                'id': f'loop_{i:04d}',
                'type': 'loop',
                'text': text,
                'is_degenerate': True,
                'metadata': {
                    'degeneracy_type': 'loop',
                    'severity': severity,
                    'num_repeats': num_repeats
                }
            })

        return samples

    def generate_drift(self, n: int = 125) -> List[Dict]:
        """Generate samples with semantic drift."""
        samples = []
        for i in range(n):
            topic_start, topic_end = self.random.choice(self.drift_topics)

            # Generate gradually drifting text
            sentences = []

            # Start with topic A
            sentences.append(f"Let's discuss {topic_start}. This is a fascinating area of study.")
            sentences.append(f"Recent developments in {topic_start} have been significant.")

            # Transition (abrupt or gradual)
            if self.random.random() < 0.5:  # Abrupt drift
                sentences.append(f"Speaking of which, {topic_end} is also interesting.")
                severity = 'high'
            else:  # Gradual drift
                sentences.append("This reminds me of something else entirely.")
                sentences.append(f"Consider, for example, {topic_end}.")
                severity = 'medium'

            # End with topic B
            sentences.append(f"The key aspects of {topic_end} are worth exploring.")
            sentences.append(f"In conclusion, {topic_end} deserves more attention.")

            text = " ".join(sentences)

            samples.append({
                'id': f'drift_{i:04d}',
                'type': 'drift',
                'text': text,
                'is_degenerate': True,
                'metadata': {
                    'degeneracy_type': 'semantic_drift',
                    'severity': severity,
                    'topic_start': topic_start,
                    'topic_end': topic_end
                }
            })

        return samples

    def generate_incoherence(self, n: int = 125) -> List[Dict]:
        """Generate samples with contradictions and incoherence."""
        samples = []
        for i in range(n):
            # Start with a normal sentence
            intro = "Let me explain this phenomenon."

            # Add contradictory statements
            num_contradictions = self.random.randint(2, 4)
            statements = []
            for _ in range(num_contradictions):
                s1, s2 = self.random.choice(self.contradictions)
                statements.append(s1)
                statements.append(s2)

            # Add filler
            filler = "This is an important observation."

            # Combine
            all_sentences = [intro] + statements + [filler]
            self.random.shuffle(statements)  # Mix up the order
            text = " ".join([intro] + statements + [filler])

            severity = 'high' if num_contradictions >= 3 else 'medium'

            samples.append({
                'id': f'incoherent_{i:04d}',
                'type': 'incoherence',
                'text': text,
                'is_degenerate': True,
                'metadata': {
                    'degeneracy_type': 'incoherence',
                    'severity': severity,
                    'num_contradictions': num_contradictions
                }
            })

        return samples

    def generate_repetition(self, n: int = 125) -> List[Dict]:
        """Generate samples with word/phrase repetition (not exact loops)."""
        samples = []
        for i in range(n):
            # Pick a word to repeat excessively
            repeated_words = ['very', 'really', 'actually', 'basically', 'literally', 'essentially']
            word = self.random.choice(repeated_words)

            # Create sentence with excessive repetition
            base_sentences = [
                f"This is {word} {word} {word} important to understand.",
                f"The system is {word} complex and {word} {word} difficult to analyze.",
                f"We need to {word} focus on this {word} {word} critical aspect.",
            ]

            num_repeats = self.random.randint(5, 15)
            text = " ".join([self.random.choice(base_sentences) for _ in range(num_repeats)])

            severity = 'high' if num_repeats >= 10 else 'medium'

            samples.append({
                'id': f'repetition_{i:04d}',
                'type': 'repetition',
                'text': text,
                'is_degenerate': True,
                'metadata': {
                    'degeneracy_type': 'repetition',
                    'severity': severity,
                    'repeated_word': word,
                    'num_repeats': num_repeats
                }
            })

        return samples


def main():
    """Generate degeneracy dataset."""
    print("=== Structural Degeneracy Dataset Generator ===\n")

    # Create output directory
    output_dir = Path('data/benchmarks/degeneracy')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize generator
    generator = DegeneracyGenerator(seed=42)

    # Generate samples
    print("Generating samples...")
    print("- Normal samples (500)...")
    normal_samples = generator.generate_normal(500)

    print("- Loop samples (125)...")
    loop_samples = generator.generate_loops(125)

    print("- Semantic drift samples (125)...")
    drift_samples = generator.generate_drift(125)

    print("- Incoherence samples (125)...")
    incoherent_samples = generator.generate_incoherence(125)

    print("- Repetition samples (125)...")
    repetition_samples = generator.generate_repetition(125)

    # Combine all samples
    all_samples = (
        normal_samples +
        loop_samples +
        drift_samples +
        incoherent_samples +
        repetition_samples
    )

    # Shuffle
    random.Random(42).shuffle(all_samples)

    # Save as JSONL
    output_path = output_dir / 'degeneracy_synthetic.jsonl'
    with open(output_path, 'w') as f:
        for sample in all_samples:
            f.write(json.dumps(sample) + '\n')

    # Statistics
    print(f"\n✅ Generated {len(all_samples)} samples")
    print(f"   Saved to: {output_path}")

    print("\nBreakdown:")
    print(f"  - Normal:      {len(normal_samples):4d} (is_degenerate=False)")
    print(f"  - Loops:       {len(loop_samples):4d} (is_degenerate=True)")
    print(f"  - Drift:       {len(drift_samples):4d} (is_degenerate=True)")
    print(f"  - Incoherence: {len(incoherent_samples):4d} (is_degenerate=True)")
    print(f"  - Repetition:  {len(repetition_samples):4d} (is_degenerate=True)")

    degenerate_count = sum(1 for s in all_samples if s['is_degenerate'])
    print(f"\nTotal degenerate: {degenerate_count}/{len(all_samples)} ({degenerate_count/len(all_samples)*100:.1f}%)")

    # Show examples
    print("\n=== Sample Examples ===")
    for sample_type in ['normal', 'loop', 'drift', 'incoherence', 'repetition']:
        sample = next(s for s in all_samples if s['type'] == sample_type)
        print(f"\n{sample_type.upper()}:")
        print(f"  ID: {sample['id']}")
        print(f"  Text: {sample['text'][:150]}...")
        print(f"  Degenerate: {sample['is_degenerate']}")


if __name__ == '__main__':
    main()
