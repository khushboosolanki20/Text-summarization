import { useEffect, useState } from 'react'
import { getConfig } from '../services/api'

// Used until /api/config answers (or if the backend is offline), so the form
// still renders. The server's values always win once loaded.
const FALLBACK = {
  methods: [
    { id: 'tfidf', name: 'TF-IDF', type: 'extractive', description: 'Ranks sentences by similarity to the document centroid.' },
    { id: 'textrank', name: 'TextRank', type: 'extractive', description: 'Ranks sentences with PageRank over a similarity graph.' },
    { id: 'bart', name: 'BART', type: 'abstractive', description: 'A Transformer writes a new summary.' },
    { id: 'hybrid', name: 'Hybrid', type: 'hybrid', description: 'TextRank selects key sentences, BART rewrites them.' },
  ],
  lengths: { short: 0.12, medium: 0.22, long: 0.32 },
  supported_file_types: ['.txt', '.pdf', '.docx'],
  max_upload_mb: 10,
  max_input_chars: 500000,
  min_input_words: 40,
  abstractive_max_input_words: 20000,
  faithfulness_check: true,
}

export function useConfig() {
  const [config, setConfig] = useState(FALLBACK)
  useEffect(() => {
    let cancelled = false
    getConfig()
      .then((c) => !cancelled && setConfig(c))
      .catch(() => {})
    return () => {
      cancelled = true
    }
  }, [])
  return config
}
