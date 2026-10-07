const METHODS = [
  { name: 'TF-IDF', kind: 'Extractive', blurb: 'Ranks sentences by the statistical importance of their words.' },
  { name: 'TextRank', kind: 'Extractive', blurb: 'Ranks sentences with PageRank over a sentence-similarity graph.' },
  { name: 'BART', kind: 'Abstractive', blurb: 'A Transformer that writes a new summary in its own words.' },
  { name: 'Hybrid', kind: 'Extractive → Abstractive', blurb: 'TextRank picks key sentences, BART rewrites them.' },
]

/**
 * Landing page shell (Phase 1). The upload form, method selector and results
 * dashboard are added in Phase 12 once the summarisation API exists.
 */
export default function HomePage() {
  return (
    <section className="mx-auto max-w-5xl px-4 py-16 sm:px-6">
      <div className="text-center">
        <h1 className="text-4xl font-bold tracking-tight text-slate-900 sm:text-5xl">IntelliSum</h1>
        <p className="mt-3 text-lg text-slate-600">Intelligent Text Summarization</p>
        <p className="mx-auto mt-4 max-w-2xl text-sm text-slate-500">
          Compare classical extractive NLP with Transformer-based abstractive summarization on plain text,
          PDF and DOCX documents, evaluated with ROUGE.
        </p>
      </div>

      <div className="mt-12 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {METHODS.map((m) => (
          <div key={m.name} className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
            <p className="text-xs font-semibold uppercase tracking-wide text-indigo-600">{m.kind}</p>
            <h2 className="mt-1 text-lg font-semibold text-slate-900">{m.name}</h2>
            <p className="mt-2 text-sm text-slate-600">{m.blurb}</p>
          </div>
        ))}
      </div>
    </section>
  )
}
