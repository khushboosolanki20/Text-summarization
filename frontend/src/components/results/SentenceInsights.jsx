import { useState } from 'react'
import { formatNumber, formatPercent } from '../../utils/format'
import { Badge, Card, Stat } from '../ui/primitives'
import { SentenceImportanceChart } from './charts'

const INITIAL_ROWS = 150

function MethodExplanation({ result }) {
  const meta = result.metadata
  if (result.method === 'tfidf') {
    return (
      <div>
        <p className="text-sm text-slate-600">
          Each sentence is scored by its cosine similarity to the document's TF-IDF centroid: the average of all
          sentence vectors, dominated by the document's most characteristic words:
        </p>
        <div className="mt-2 flex flex-wrap gap-1.5">
          {(meta.top_keywords ?? []).map((k) => (
            <Badge key={k.term} tone="indigo">
              {k.term}
            </Badge>
          ))}
        </div>
      </div>
    )
  }
  const graph = meta.graph ?? meta.extractive_stage?.graph
  return (
    <div className="space-y-3">
      <p className="text-sm text-slate-600">
        {result.method === 'hybrid'
          ? 'TextRank ranked every sentence; the highest-ranked ones (about 3× the summary length) were passed to BART, which wrote the summary.'
          : 'Sentences are nodes of a similarity graph; PageRank scores a sentence highly when it is similar to many other highly-scored sentences.'}
      </p>
      {graph && (
        <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
          <Stat label="Graph nodes (sentences)" value={formatNumber(graph.nodes)} />
          <Stat label="Edges (similar pairs)" value={formatNumber(graph.edges)} />
          <Stat label="Isolated sentences" value={formatNumber(graph.isolated_sentences)} />
          {result.method === 'hybrid' ? (
            <Stat
              label="Input BART was spared"
              value={formatPercent(meta.extractive_stage?.input_reduction)}
              hint={`${formatNumber(meta.extractive_stage?.selected_words)} words passed on`}
            />
          ) : (
            <Stat label="Graph density" value={graph.density?.toFixed(3)} />
          )}
        </div>
      )}
    </div>
  )
}

/** Explains extractive / hybrid selections: score chart + highlighted document. */
export default function SentenceInsights({ result }) {
  const [onlySelected, setOnlySelected] = useState(false)
  const [showAll, setShowAll] = useState(false)
  const selectedLabel = result.method === 'hybrid' ? 'Passed to BART' : 'In the summary'

  const rows = result.sentences.filter((s) => !onlySelected || s.selected)
  const visible = showAll ? rows : rows.slice(0, INITIAL_ROWS)

  return (
    <Card
      title="Why these sentences?"
      subtitle={result.method === 'hybrid' ? 'Stage 1: TextRank content selection' : 'How the extractive method chose'}
    >
      <div className="space-y-6">
        <MethodExplanation result={result} />
        <SentenceImportanceChart sentences={result.sentences} selectedLabel={selectedLabel} />

        <div>
          <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
            <h3 className="text-sm font-semibold text-slate-800">Document</h3>
            <label className="flex items-center gap-2 text-sm text-slate-600">
              <input
                type="checkbox"
                checked={onlySelected}
                onChange={(e) => setOnlySelected(e.target.checked)}
                className="h-4 w-4 rounded border-slate-300 text-indigo-600"
              />
              Show {selectedLabel.toLowerCase()} only
            </label>
          </div>
          <ol className="max-h-[32rem] space-y-1 overflow-y-auto rounded-xl border border-slate-200 p-2">
            {visible.map((s) => (
              <li
                key={s.index}
                className={`flex gap-3 rounded-lg px-3 py-2 text-sm leading-relaxed ${
                  s.selected ? 'border-l-4 border-indigo-600 bg-indigo-50 text-slate-900' : 'text-slate-500'
                }`}
              >
                <span className="w-8 shrink-0 text-right text-xs tabular-nums text-slate-400">{s.index + 1}</span>
                <span className="flex-1">{s.text}</span>
                <span className="flex shrink-0 flex-col items-end gap-1 text-xs tabular-nums text-slate-500">
                  {s.score != null && <span title="Importance score">{s.score.toFixed(3)}</span>}
                  {s.page != null && <span>p. {s.page}</span>}
                </span>
              </li>
            ))}
          </ol>
          {rows.length > INITIAL_ROWS && (
            <button
              type="button"
              onClick={() => setShowAll((v) => !v)}
              className="mt-2 text-sm font-medium text-indigo-600 hover:text-indigo-800"
            >
              {showAll ? 'Show fewer' : `Show all ${formatNumber(rows.length)} sentences`}
            </button>
          )}
        </div>
      </div>
    </Card>
  )
}
