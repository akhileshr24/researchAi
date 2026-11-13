# ResearchAI RAG Evaluation

Generated: 2026-11-12T18:14:16.619419+00:00

## Reproducibility

- Git commit SHA: `c80a1cc1ba408f8df0141b0d5542987bff75ba28`
- Python version: `3.13.16`
- Lock file SHA-256: `45aae886568727a19bd9cccf963e2a4cfa53f614625c1cfd5f6927c1a5265996`
- Dataset SHA-256: `f3b5c9fd41e16e9cf0e2526b85d42bd262f0b3f3f7c9151db94dc0af4c66abfb`
- Dataset size: 17 questions
- Corpus chunks: 604
- top_k: 5
- candidate_limit: 20
- vector_min_score: 0.0

## Headline metrics

| Method | P@5 | R@5 | MRR | nDCG@5 | Mean latency (ms) |
|---|---:|---:|---:|---:|---:|
| Vector | 0.3176 | 0.6255 | 0.8235 | 0.6405 | 297.2240 |
| BM25 | 0.3529 | 0.6941 | 0.8137 | 0.7309 | 87.1160 |
| Hybrid RRF | 0.3647 | 0.7059 | 0.8529 | 0.7229 | 112.0000 |
| Hybrid + CrossEncoder | 0.4235 | 0.8137 | 0.8824 | 0.8071 | 1193.1010 |

## Pipeline comparison

| Comparison | Recall@5 delta | nDCG@5 delta |
|---|---:|---:|
| BM25 vs Vector | +0.0686 | +0.0904 |
| Hybrid RRF vs BM25 | +0.0118 | -0.0080 |
| Hybrid + CrossEncoder vs Hybrid RRF | +0.1078 | +0.0842 |

Note: deltas are raw differences on a small dataset (17 questions); no statistical significance is claimed.

## Per-question breakdown

| Question ID | Vector R@5 | BM25 R@5 | Hybrid RRF R@5 | Hybrid + CrossEncoder R@5 | Vector first rel. rank | BM25 first rel. rank | Hybrid RRF first rel. rank | Hybrid + CrossEncoder first rel. rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ai-ml-dl-difference | 1.0000 | 0.6667 | 1.0000 | 1.0000 | 1 | 3 | 2 | 1 |
| overfitting-test-set | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 1 | 1 | 1 | 1 |
| rag-pipeline | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1 | 1 | 1 | 1 |
| classification-regression-metrics | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1 | 1 | 1 | 1 |
| machine-learning-types | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1 | 1 | 1 | 1 |
| rag-evaluation | 0.5000 | 0.5000 | 1.0000 | 1.0000 | 1 | 2 | 1 | 1 |
| llm-limitations | 0.5000 | 0.0000 | 0.0000 | 0.5000 | 2 | - | - | 2 |
| deep-work-hypothesis | 0.5000 | 1.0000 | 1.0000 | 1.0000 | 2 | 2 | 1 | 1 |
| productivity-formula | 0.3333 | 0.6667 | 0.6667 | 0.6667 | 1 | 1 | 1 | 1 |
| attention-residue | 0.5000 | 0.7500 | 0.7500 | 1.0000 | 1 | 1 | 1 | 1 |
| deliberate-practice | 0.5000 | 0.2500 | 0.2500 | 0.5000 | 2 | 2 | 2 | 2 |
| principle-of-least-resistance | 0.0000 | 0.6667 | 0.3333 | 0.6667 | - | 1 | 2 | 1 |
| monastic-philosophy | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 1 | 1 | 1 | 1 |
| bimodal-philosophy | 0.5000 | 1.0000 | 0.5000 | 1.0000 | 1 | 1 | 1 | 1 |
| rhythmic-philosophy | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 2 | 1 | 1 | 2 |
| fixed-schedule-productivity | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1 | 1 | 1 | 2 |
| any-benefit-vs-craftsman | 0.8000 | 0.8000 | 1.0000 | 1.0000 | 1 | 1 | 1 | 1 |

## Footer

- Dataset size: 17 questions
- Labels verified: Yes
