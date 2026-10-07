import { Banner } from '../ui/primitives'
import ComparisonPanel from './ComparisonPanel'
import FaithfulnessPanel from './FaithfulnessPanel'
import ProcessPanel from './ProcessPanel'
import RougePanel from './RougePanel'
import SentenceInsights from './SentenceInsights'
import SummaryCard from './SummaryCard'

export default function ResultView({ result, onNew }) {
  const hasSelection = result.sentences.some((s) => s.score != null)

  return (
    <div className="space-y-6">
      {result.warnings.map((warning) => (
        <Banner key={warning} kind="warning" title="Note about your document">
          {warning}
        </Banner>
      ))}
      <ComparisonPanel result={result} />
      <SummaryCard result={result} onNew={onNew} />
      <FaithfulnessPanel result={result} />
      <div className="grid gap-6 xl:grid-cols-2">
        <RougePanel result={result} />
        <ProcessPanel result={result} />
      </div>
      {hasSelection && <SentenceInsights result={result} />}
    </div>
  )
}
