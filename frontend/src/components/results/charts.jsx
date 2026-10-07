import { Bar, BarChart, CartesianGrid, Cell, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import {
  BAR_MAX_SIZE,
  EMPHASIS,
  INK,
  SERIES,
  axisProps,
  gridProps,
  roundedRight,
  roundedTop,
} from '../../utils/chartTheme'
import { NODE_LABELS, ROUGE_NAMES, formatSeconds } from '../../utils/format'

function TooltipBox({ children }) {
  return (
    <div className="max-w-xs rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs shadow-lg" style={{ color: INK.primary }}>
      {children}
    </div>
  )
}

/** Legend entry: colored swatch beside text in ink color (text never wears the series color). */
function LegendItems({ items }) {
  return (
    <ul className="flex flex-wrap gap-4 text-xs" style={{ color: INK.secondary }}>
      {items.map((item) => (
        <li key={item.label} className="flex items-center gap-1.5">
          <span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: item.color }} />
          {item.label}
        </li>
      ))}
    </ul>
  )
}

// ------------------------------------------------------------------- ROUGE

const ROUGE_SERIES = [
  { key: 'precision', label: 'Precision', color: SERIES[0] },
  { key: 'recall', label: 'Recall', color: SERIES[1] },
  { key: 'f1', label: 'F1', color: SERIES[2] },
]

/** Grouped bars: precision / recall / F1 for each ROUGE variant (values 0–100). */
export function RougeChart({ rouge }) {
  const data = Object.entries(ROUGE_NAMES).map(([key, name]) => ({
    name,
    precision: +(rouge[key].precision * 100).toFixed(1),
    recall: +(rouge[key].recall * 100).toFixed(1),
    f1: +(rouge[key].f1 * 100).toFixed(1),
  }))
  return (
    <div className="h-64" role="img" aria-label="ROUGE precision, recall and F1 per variant (values listed in the table)">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} barGap={2} margin={{ top: 8, right: 8, bottom: 0, left: -16 }}>
          <CartesianGrid {...gridProps} />
          <XAxis dataKey="name" {...axisProps} />
          <YAxis domain={[0, 100]} ticks={[0, 25, 50, 75, 100]} {...axisProps} axisLine={false} />
          <Tooltip
            cursor={{ fill: INK.grid, opacity: 0.4 }}
            content={({ active, payload, label }) =>
              active && payload?.length ? (
                <TooltipBox>
                  <p className="mb-1 font-semibold">{label}</p>
                  {payload.map((p) => (
                    <p key={p.dataKey} className="flex items-center gap-1.5">
                      <span className="inline-block h-2 w-2 rounded-sm" style={{ background: p.color }} />
                      {p.name}: <span className="font-medium tabular-nums">{p.value.toFixed(1)}</span>
                    </p>
                  ))}
                </TooltipBox>
              ) : null
            }
          />
          <Legend content={() => <LegendItems items={ROUGE_SERIES} />} wrapperStyle={{ paddingTop: 8 }} />
          {ROUGE_SERIES.map((s) => (
            <Bar key={s.key} dataKey={s.key} name={s.label} fill={s.color} radius={roundedTop} maxBarSize={BAR_MAX_SIZE} />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

// ------------------------------------------------------- sentence importance

const MAX_SENTENCE_BARS = 400

/**
 * One column per sentence in document order; height = importance score.
 * Selected sentences use the strong blue step, the rest the light step of the
 * same ramp (emphasis, not a second identity).
 */
export function SentenceImportanceChart({ sentences, selectedLabel }) {
  if (sentences.length > MAX_SENTENCE_BARS) {
    return (
      <p className="text-sm text-slate-500">
        The chart is hidden for documents with more than {MAX_SENTENCE_BARS} sentences; scores are listed below.
      </p>
    )
  }
  const data = sentences.map((s) => ({ ...s, position: s.index + 1 }))
  return (
    <div>
      <LegendItems
        items={[
          { label: selectedLabel, color: EMPHASIS.strong },
          { label: 'Other sentences', color: EMPHASIS.soft },
        ]}
      />
      <div className="mt-2 h-44" role="img" aria-label="Importance score of every sentence in document order">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} barCategoryGap={1} margin={{ top: 4, right: 4, bottom: 0, left: -16 }}>
            <CartesianGrid {...gridProps} />
            <XAxis dataKey="position" {...axisProps} interval="preserveStartEnd" minTickGap={24} />
            <YAxis {...axisProps} axisLine={false} tickFormatter={(v) => String(Number(v.toPrecision(2)))} width={56} />
            <Tooltip
              cursor={{ fill: INK.grid, opacity: 0.4 }}
              content={({ active, payload }) => {
                const s = active && payload?.[0]?.payload
                if (!s) return null
                return (
                  <TooltipBox>
                    <p className="font-semibold">
                      Sentence {s.position}
                      {s.page != null && ` · page ${s.page}`} · {s.selected ? selectedLabel.toLowerCase() : 'not selected'}
                    </p>
                    <p className="tabular-nums" style={{ color: INK.secondary }}>
                      Score {s.score?.toFixed(4)}
                    </p>
                    <p className="mt-1 line-clamp-3">{s.text}</p>
                  </TooltipBox>
                )
              }}
            />
            <Bar dataKey="score" radius={[2, 2, 0, 0]} maxBarSize={BAR_MAX_SIZE}>
              {data.map((s) => (
                <Cell key={s.index} fill={s.selected ? EMPHASIS.strong : EMPHASIS.soft} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

// ------------------------------------------------------------------ timings

/** Horizontal bars: seconds spent in each workflow node (single series, no legend). */
export function TimingsChart({ path, timings }) {
  const nodes = [...new Set(path)]
  const data = nodes.map((node) => ({ node: NODE_LABELS[node] ?? node, seconds: timings[node] ?? 0 }))
  return (
    <div style={{ height: 28 * data.length + 40 }} role="img" aria-label="Seconds spent in each workflow step">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical" margin={{ top: 4, right: 16, bottom: 0, left: 8 }}>
          <CartesianGrid stroke={INK.grid} horizontal={false} />
          <XAxis type="number" {...axisProps} tickFormatter={(v) => formatSeconds(v)} />
          <YAxis type="category" dataKey="node" {...axisProps} axisLine={false} width={130} />
          <Tooltip
            cursor={{ fill: INK.grid, opacity: 0.4 }}
            content={({ active, payload }) =>
              active && payload?.[0] ? (
                <TooltipBox>
                  {payload[0].payload.node}: <span className="font-medium">{formatSeconds(payload[0].value)}</span>
                </TooltipBox>
              ) : null
            }
          />
          <Bar dataKey="seconds" fill={SERIES[0]} radius={roundedRight} maxBarSize={18} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
