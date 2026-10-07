import { NODE_LABELS, formatNumber } from '../../utils/format'
import Icon from '../ui/Icon'
import { Card, Collapsible } from '../ui/primitives'
import { TimingsChart } from './charts'

function Detail({ label, value }) {
  if (value == null || value === '') return null
  return (
    <div className="flex justify-between gap-4 border-b border-slate-100 py-1.5 text-sm last:border-0">
      <dt className="text-slate-500">{label}</dt>
      <dd className="text-right font-medium text-slate-800">{value}</dd>
    </div>
  )
}

/** How the LangGraph workflow produced this result: route, timings, model details, chunks. */
export default function ProcessPanel({ result }) {
  const meta = result.metadata
  const abstractive = result.method === 'bart' || result.method === 'hybrid'
  const levels = meta.reduction_levels ?? []

  return (
    <Card title="How it was produced" subtitle="The route through the LangGraph workflow and where the time went">
      <div className="space-y-5">
        <ol className="flex flex-wrap items-center gap-1.5 text-xs" aria-label="Workflow path">
          {result.path.map((node, i) => (
            <li key={i} className="flex items-center gap-1.5">
              {i > 0 && <Icon name="chevron" className="h-3 w-3 text-slate-400" />}
              <span className="rounded-md bg-slate-100 px-2 py-1 font-medium text-slate-700">{NODE_LABELS[node] ?? node}</span>
            </li>
          ))}
        </ol>

        <TimingsChart path={result.path} timings={result.timings} />

        {abstractive && (
          <dl>
            <Detail label="Model" value={meta.model} />
            <Detail label="Device" value={meta.device?.toUpperCase()} />
            <Detail label="Input to BART" value={`${formatNumber(meta.input_tokens)} tokens (window ${formatNumber(meta.max_input_tokens)})`} />
            <Detail label="Target summary length" value={`${formatNumber(meta.target_words)} words`} />
            <Detail label="Chunks" value={meta.chunks} />
            <Detail label="Reduction rounds" value={meta.reduction_rounds} />
          </dl>
        )}

        {levels.length > 0 && (
          <Collapsible title={`Chunk summaries (${result.intermediate_summaries.length})`}>
            <ol className="space-y-3">
              {result.intermediate_summaries.map((summary, i) => {
                const pages = levels[0].chunk_pages?.[i]
                return (
                  <li key={i} className="text-sm">
                    <p className="text-xs font-medium text-slate-500">
                      Chunk {i + 1} · {formatNumber(levels[0].chunk_token_counts[i])} tokens
                      {pages?.length ? ` · pages ${pages[0]}–${pages[pages.length - 1]}` : ''}
                    </p>
                    <p className="mt-0.5 text-slate-700">{summary}</p>
                  </li>
                )
              })}
            </ol>
          </Collapsible>
        )}
      </div>
    </Card>
  )
}
