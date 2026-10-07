import { useState } from 'react'
import { Link } from 'react-router-dom'
import Icon from '../components/ui/Icon'
import { Badge, Button } from '../components/ui/primitives'
import { copyText, downloadText, summaryFilename } from '../utils/download'
import { METHOD_LABELS, formatDate, formatNumber, formatPercent, formatScore, formatSeconds } from '../utils/format'
import { clearHistory, loadHistory, removeFromHistory } from '../utils/history'

function HistoryItem({ entry, onDelete }) {
  const s = entry.stats
  return (
    <li className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="flex flex-wrap items-center gap-2 text-sm">
            <Badge tone="indigo">{METHOD_LABELS[entry.method]}</Badge>
            <Badge>{entry.length}</Badge>
            <span className="font-medium text-slate-800">{entry.source}</span>
          </p>
          <p className="mt-1 text-xs text-slate-500">{formatDate(entry.date)}</p>
        </div>
        <div className="flex gap-2">
          <Button size="sm" icon="copy" onClick={() => copyText(entry.summary)}>
            Copy
          </Button>
          <Button size="sm" icon="download" onClick={() => downloadText(entry.summary, summaryFilename(entry.source, entry.method))}>
            .txt
          </Button>
          <Button size="sm" variant="ghost" icon="trash" onClick={() => onDelete(entry.id)} aria-label="Delete from history" />
        </div>
      </div>
      <p className="mt-3 line-clamp-4 text-sm leading-relaxed text-slate-700">{entry.summary}</p>
      <dl className="mt-3 flex flex-wrap gap-x-5 gap-y-1 text-xs text-slate-500">
        <div>
          <dt className="inline">Words </dt>
          <dd className="inline font-medium text-slate-700">
            {formatNumber(s.original_word_count)} → {formatNumber(s.summary_word_count)}
          </dd>
        </div>
        <div>
          <dt className="inline">Compression </dt>
          <dd className="inline font-medium text-slate-700">{formatPercent(s.compression_ratio)}</dd>
        </div>
        <div>
          <dt className="inline">Time </dt>
          <dd className="inline font-medium text-slate-700">{formatSeconds(s.processing_time)}</dd>
        </div>
        {s.rouge1 != null && (
          <div>
            <dt className="inline">ROUGE-1/2/L </dt>
            <dd className="inline font-medium tabular-nums text-slate-700">
              {formatScore(s.rouge1)} / {formatScore(s.rouge2)} / {formatScore(s.rougeL)}
            </dd>
          </div>
        )}
      </dl>
    </li>
  )
}

export default function HistoryPage() {
  const [entries, setEntries] = useState(loadHistory)

  return (
    <div className="mx-auto max-w-4xl px-4 py-10 sm:px-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold text-slate-900">History</h1>
          <p className="mt-1 text-sm text-slate-500">Your last {entries.length || ''} summaries, stored only in this browser.</p>
        </div>
        {entries.length > 0 && (
          <Button variant="danger" icon="trash" onClick={() => window.confirm('Delete all history?') && setEntries(clearHistory())}>
            Clear history
          </Button>
        )}
      </div>

      {entries.length === 0 ? (
        <div className="mt-8 flex flex-col items-center rounded-2xl border border-dashed border-slate-300 bg-white/60 px-6 py-12 text-center">
          <Icon name="history" className="h-8 w-8 text-slate-400" />
          <p className="mt-3 font-medium text-slate-800">No summaries yet</p>
          <p className="mt-1 text-sm text-slate-500">Summaries you generate are saved here automatically.</p>
          <Link to="/" className="mt-4 text-sm font-medium text-indigo-600 hover:text-indigo-800">
            Create a summary →
          </Link>
        </div>
      ) : (
        <ul className="mt-6 space-y-4">
          {entries.map((entry) => (
            <HistoryItem key={entry.id} entry={entry} onDelete={(id) => setEntries(removeFromHistory(id))} />
          ))}
        </ul>
      )}
    </div>
  )
}
