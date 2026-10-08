// Realistic API responses (same shape as SummaryResponse in backend/app/api/schemas.py).

const SENTENCES = [
  'Solar power capacity grew faster than any other energy source last year.',
  'The weather was pleasant in Paris when the report was released.',
  'China accounted for more than half of all new solar capacity.',
  'Several journalists attended the launch event downtown.',
]

export const CONFIG = {
  methods: [
    { id: 'tfidf', name: 'TF-IDF', type: 'extractive', description: 'Centroid similarity.' },
    { id: 'textrank', name: 'TextRank', type: 'extractive', description: 'PageRank over sentences.' },
    { id: 'bart', name: 'BART', type: 'abstractive', description: 'Transformer.' },
    { id: 'hybrid', name: 'Hybrid', type: 'hybrid', description: 'TextRank then BART.' },
  ],
  lengths: { short: 0.12, medium: 0.22, long: 0.32 },
  supported_file_types: ['.txt', '.pdf', '.docx'],
  max_upload_mb: 10,
  max_input_chars: 500000,
  min_input_words: 40,
  abstractive_max_input_words: 20000,
  abstractive_model: 'facebook/bart-large-cnn',
  faithfulness_check: true,
}

const base = {
  length: 'short',
  original_word_count: 44,
  processing_time: 0.05,
  intermediate_summaries: [],
  warnings: [],
  source: { type: 'text' },
  timings: { preprocess: 0.03, extractive_summarize: 0.01, postprocess: 0, evaluate: 0.01 },
}

export const EXTRACTIVE_RESULT = {
  ...base,
  summary: `${SENTENCES[0]} ${SENTENCES[2]}`,
  method: 'textrank',
  summary_word_count: 22,
  compression_ratio: 50,
  strategy: 'extractive',
  path: ['preprocess', 'extractive_summarize', 'postprocess', 'evaluate'],
  metrics: {
    original_word_count: 44,
    summary_word_count: 22,
    compression_ratio: 50,
    num_sentences: 4,
    num_selected_sentences: 2,
    processing_time: 0.05,
    rouge1: null,
    rouge2: null,
    rougeL: null,
    rouge: null,
    rouge_note: 'ROUGE needs a human-written reference summary to compare against.',
  },
  sentences: SENTENCES.map((text, index) => ({
    index,
    text,
    page: index < 2 ? 1 : 2,
    score: [0.4, 0.1, 0.35, 0.15][index],
    selected: index === 0 || index === 2,
  })),
  faithfulness: { applicable: false, label: 'Potentially unsupported content (experimental)', reason: 'Extractive.' },
  metadata: { num_sentences: 4, num_selected: 2, graph: { nodes: 4, edges: 3, density: 0.5, isolated_sentences: 1 } },
}

const rouge = (p, r, f) => ({ precision: p, recall: r, f1: f })

export const ABSTRACTIVE_RESULT = {
  ...base,
  summary: 'Solar capacity grew fastest. Installations rose by 15 percent.',
  method: 'bart',
  summary_word_count: 9,
  compression_ratio: 79.55,
  strategy: 'single_pass',
  path: ['preprocess', 'check_length', 'abstractive_single_pass', 'postprocess', 'evaluate'],
  timings: { preprocess: 0.03, check_length: 0.2, abstractive_single_pass: 4.1, postprocess: 0, evaluate: 0.2 },
  metrics: {
    ...EXTRACTIVE_RESULT.metrics,
    summary_word_count: 9,
    compression_ratio: 79.55,
    num_selected_sentences: null,
    rouge1: 0.5,
    rouge2: 0.25,
    rougeL: 0.45,
    rouge: { rouge1: rouge(0.6, 0.43, 0.5), rouge2: rouge(0.3, 0.21, 0.25), rougeL: rouge(0.55, 0.38, 0.45), rougeLsum: rouge(0.55, 0.38, 0.45) },
    rouge_note: null,
  },
  sentences: SENTENCES.map((text, index) => ({ index, text, page: null, score: null, selected: false })),
  faithfulness: {
    applicable: true,
    label: 'Potentially unsupported content (experimental)',
    note: 'Heuristic check; not a guarantee of correctness.',
    flagged_count: 1,
    sentences: [
      { sentence: 'Solar capacity grew fastest.', coverage: 1, similarity: 0.8, closest_source: [0], unsupported_numbers: [], unsupported_entities: [], flagged: false, reasons: [] },
      {
        sentence: 'Installations rose by 15 percent.',
        coverage: 1,
        similarity: 0.6,
        closest_source: [2],
        unsupported_numbers: ['15'],
        unsupported_entities: [],
        flagged: true,
        reasons: ['number(s) not found in the source: 15'],
      },
    ],
  },
  metadata: { model: 'facebook/bart-large-cnn', device: 'cpu', input_tokens: 60, max_input_tokens: 1024, target_words: 5, chunks: 1 },
}

export const LONG_TEXT = Array.from({ length: 12 }, (_, i) => `Sentence number ${i} talks about solar power growth.`).join(' ')
