import { formatNumber, formatPercent, formatSeconds } from '../../utils/format'
import { Stat } from '../ui/primitives'

/**
 * Original vs summary at a glance: word counts on either side, the compression
 * in the middle as a meter (the summary's share of the original), then key
 * figures as stat tiles.
 */
export default function ComparisonPanel({ result }) {
  const { metrics } = result
  const summaryShare = result.original_word_count
    ? Math.min(100, (result.summary_word_count / result.original_word_count) * 100)
    : 0
  const selected = metrics.num_selected_sentences
  const chunks = result.metadata?.chunks

  return (
    <div className="space-y-4">
      <div className="grid gap-4 rounded-2xl border border-slate-200 bg-white p-5 shadow-sm sm:grid-cols-[1fr_auto_1fr] sm:items-center">
        <div>
          <p className="text-sm font-medium text-slate-500">Original document</p>
          <p className="mt-1 text-3xl font-semibold tabular-nums text-slate-900">{formatNumber(result.original_word_count)}</p>
          <p className="text-sm text-slate-500">
            words · {formatNumber(metrics.num_sentences)} sentences
            {result.source?.pages ? ` · ${result.source.pages} pages` : ''}
          </p>
        </div>

        <div className="text-center sm:px-6">
          <p className="text-4xl font-bold tabular-nums text-indigo-700">{formatPercent(result.compression_ratio)}</p>
          <p className="text-sm font-medium text-slate-600">compression</p>
          <div
            className="mx-auto mt-2 h-2 w-40 overflow-hidden rounded-full bg-indigo-100"
            role="meter"
            aria-label="Summary length as a share of the original"
            aria-valuenow={Math.round(summaryShare)}
            aria-valuemin={0}
            aria-valuemax={100}
          >
            <div className="h-full rounded-full bg-indigo-600" style={{ width: `${Math.max(summaryShare, 2)}%` }} />
          </div>
          <p className="mt-1 text-xs text-slate-500">summary is {formatPercent(summaryShare)} of the original</p>
        </div>

        <div className="sm:text-right">
          <p className="text-sm font-medium text-slate-500">Summary</p>
          <p className="mt-1 text-3xl font-semibold tabular-nums text-slate-900">{formatNumber(result.summary_word_count)}</p>
          <p className="text-sm text-slate-500">words</p>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Stat label="Processing time" value={formatSeconds(result.processing_time)} />
        {selected != null ? (
          <Stat
            label={result.method === 'hybrid' ? 'Sentences passed to BART' : 'Sentences selected'}
            value={`${formatNumber(selected)} / ${formatNumber(metrics.num_sentences)}`}
          />
        ) : (
          <Stat label="Chunks processed" value={formatNumber(chunks ?? 1)} hint={chunks > 1 ? 'long document' : 'fits in one pass'} />
        )}
        <Stat
          label="ROUGE-1 F1"
          value={metrics.rouge1 != null ? (metrics.rouge1 * 100).toFixed(1) : '—'}
          hint={metrics.rouge1 == null ? 'needs a reference summary' : 'vs. reference'}
        />
        <Stat
          label="ROUGE-L F1"
          value={metrics.rougeL != null ? (metrics.rougeL * 100).toFixed(1) : '—'}
          hint={metrics.rougeL == null ? 'needs a reference summary' : 'vs. reference'}
        />
      </div>
    </div>
  )
}
