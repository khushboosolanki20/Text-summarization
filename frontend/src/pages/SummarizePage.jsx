import { Suspense, lazy, useEffect, useRef } from 'react'
import ProgressPanel from '../components/ProgressPanel'
import SummarizeForm from '../components/input/SummarizeForm'
import Icon from '../components/ui/Icon'
import { Banner } from '../components/ui/primitives'
import { useConfig } from '../hooks/useConfig'
import { useSummarize } from '../hooks/useSummarize'

// The results view (with the charting library) is loaded only once there is
// a result, keeping the initial page light.
const ResultView = lazy(() => import('../components/results/ResultView'))

function EmptyState() {
  return (
    <div className="flex flex-col items-center rounded-2xl border border-dashed border-slate-300 bg-white/60 px-6 py-12 text-center">
      <div className="rounded-full bg-indigo-50 p-3 text-indigo-600">
        <Icon name="sparkles" className="h-6 w-6" />
      </div>
      <p className="mt-3 font-medium text-slate-800">Your summary will appear here</p>
      <p className="mt-1 max-w-md text-sm text-slate-500">
        Paste text or upload a document, pick a method and press <span className="font-medium">Generate summary</span>.
        Results include statistics, highlighted sentences and, with a reference summary, ROUGE scores.
      </p>
    </div>
  )
}

export default function SummarizePage() {
  const config = useConfig()
  const job = useSummarize()
  const resultsRef = useRef(null)

  // Bring progress / results into view when a run starts or finishes.
  useEffect(() => {
    if (job.status !== 'idle') resultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }, [job.status])

  return (
    <div className="mx-auto max-w-6xl space-y-8 px-4 py-10 sm:px-6">
      <header className="text-center">
        <h1 className="text-4xl font-bold tracking-tight text-slate-900 sm:text-5xl">IntelliSum</h1>
        <p className="mt-2 text-lg text-slate-600">Intelligent Text Summarization</p>
        <p className="mx-auto mt-2 max-w-2xl text-sm text-slate-500">
          Classical extractive NLP (TF-IDF, TextRank) and a Transformer (BART), running locally, with ROUGE
          evaluation.
        </p>
      </header>

      <SummarizeForm config={config} running={job.status === 'running'} onSubmit={job.run} />

      <div ref={resultsRef} className="scroll-mt-20">
        {job.status === 'idle' && <EmptyState />}
        {job.status === 'running' && (
          <ProgressPanel progress={job.progress} startedAt={job.startedAt} method={job.method} onCancel={job.cancel} />
        )}
        {job.status === 'error' && (
          <Banner kind="error" title="The summary could not be created" onClose={job.reset}>
            {job.error}
          </Banner>
        )}
        {job.status === 'done' && (
          <Suspense fallback={<p className="text-center text-sm text-slate-500">Loading results…</p>}>
            <ResultView key={job.startedAt} result={job.result} onNew={job.reset} />
          </Suspense>
        )}
      </div>
    </div>
  )
}
