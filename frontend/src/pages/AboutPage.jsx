const METHODS = [
  {
    name: 'TF-IDF',
    kind: 'Extractive',
    points: [
      'Each sentence becomes a vector of TF-IDF weights: words frequent in the sentence but rare across the document weigh more.',
      'Averaging all sentence vectors gives the document centroid, i.e. what the document is about.',
      'Sentences most similar (cosine) to the centroid are selected and kept in their original order.',
    ],
  },
  {
    name: 'TextRank',
    kind: 'Extractive',
    points: [
      'Sentences become nodes of a graph; edges connect sentences that share vocabulary, weighted by similarity.',
      'PageRank scores a sentence highly when it is similar to many other highly scored sentences.',
      'The top-ranked sentences form the summary; near-duplicates are skipped.',
    ],
  },
  {
    name: 'BART',
    kind: 'Abstractive',
    points: [
      'A Transformer encoder-decoder (facebook/bart-large-cnn) reads the text and writes a new summary.',
      'Attention lets every word look at every other word, and the decoder consult the most relevant source parts while writing.',
      'Texts longer than its 1,024-token window are chunked, summarized per chunk, then fused, and never truncated.',
    ],
  },
  {
    name: 'Hybrid',
    kind: 'Extractive → Abstractive',
    points: [
      'TextRank first selects the most central sentences (about 3× the summary length).',
      'BART then rewrites only that selection, often in a single pass instead of many chunks.',
      'Faster on long documents and focused on central content; TextRank mistakes carry over.',
    ],
  },
]

export default function AboutPage() {
  return (
    <div className="mx-auto max-w-4xl px-4 py-10 sm:px-6">
      <h1 className="text-2xl font-bold text-slate-900">How IntelliSum works</h1>
      <p className="mt-2 text-slate-600">
        Four independently implemented methods, orchestrated by a LangGraph workflow and evaluated with ROUGE. All
        summarization runs locally, with no external AI service.
      </p>

      <div className="mt-8 grid gap-4 sm:grid-cols-2">
        {METHODS.map((m) => (
          <section key={m.name} className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
            <p className="text-xs font-semibold uppercase tracking-wide text-indigo-600">{m.kind}</p>
            <h2 className="mt-1 text-lg font-semibold text-slate-900">{m.name}</h2>
            <ul className="mt-3 list-disc space-y-1.5 pl-5 text-sm text-slate-600">
              {m.points.map((p) => (
                <li key={p}>{p}</li>
              ))}
            </ul>
          </section>
        ))}
      </div>

      <section className="mt-8 space-y-4 rounded-2xl border border-slate-200 bg-white p-6 text-sm leading-relaxed text-slate-600 shadow-sm">
        <h2 className="text-lg font-semibold text-slate-900">Evaluation and its limits</h2>
        <p>
          <span className="font-medium text-slate-800">ROUGE</span> counts overlapping words (ROUGE-1), word pairs
          (ROUGE-2) and the longest common word sequence (ROUGE-L) between a summary and a human-written reference. It
          needs a reference, so for your own documents it is only shown when you provide one. It measures overlap, not
          meaning or correctness.
        </p>
        <p>
          <span className="font-medium text-slate-800">Potentially unsupported content</span> is an experimental check
          for BART and Hybrid summaries: sentences whose words, numbers or names do not appear in the source are flagged
          for review. It can miss errors and flag correct paraphrases.
        </p>
        <p>
          <span className="font-medium text-slate-800">Limitations:</span> English only; scanned PDFs need OCR, which is
          not supported; BART on a CPU is slow for long documents.
        </p>
      </section>
    </div>
  )
}
