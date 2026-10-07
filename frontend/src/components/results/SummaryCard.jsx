import { useState } from 'react'
import { copyText, downloadText, summaryFilename } from '../../utils/download'
import { METHOD_LABELS, STRATEGY_LABELS } from '../../utils/format'
import Icon from '../ui/Icon'
import { Badge, Button, Card } from '../ui/primitives'

/**
 * The summary itself. For abstractive output, sentences the faithfulness
 * check flagged are marked (dotted underline + icon + reason on hover/focus),
 * so the reader knows which claims to verify.
 */
export default function SummaryCard({ result, onNew }) {
  const [copied, setCopied] = useState(false)
  const faithfulness = result.faithfulness
  const sentences = faithfulness?.applicable ? faithfulness.sentences : null

  const copy = async () => {
    if (await copyText(result.summary)) {
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    }
  }

  return (
    <Card
      title="Summary"
      subtitle={
        <span className="flex flex-wrap items-center gap-2">
          <Badge tone="indigo">{METHOD_LABELS[result.method]}</Badge>
          <Badge>{result.length}</Badge>
          <span>{STRATEGY_LABELS[result.strategy] ?? result.strategy}</span>
        </span>
      }
      actions={
        <>
          <Button size="sm" icon={copied ? 'check' : 'copy'} onClick={copy}>
            {copied ? 'Copied' : 'Copy'}
          </Button>
          <Button
            size="sm"
            icon="download"
            onClick={() => downloadText(result.summary, summaryFilename(result.source?.filename, result.method))}
          >
            Download .txt
          </Button>
          <Button size="sm" variant="ghost" icon="refresh" onClick={onNew}>
            New summary
          </Button>
        </>
      }
    >
      <div className="max-w-none text-[15px] leading-7 text-slate-800">
        {sentences ? (
          <p>
            {sentences.map((s, i) => (
              <span key={i}>
                {s.flagged ? (
                  <span
                    tabIndex={0}
                    title={`Potentially unsupported: ${s.reasons.join('; ')}`}
                    className="decoration-amber-500 underline decoration-dotted decoration-2 underline-offset-4"
                  >
                    {s.sentence}
                    <Icon name="alert" className="ml-0.5 inline h-4 w-4 align-text-top text-amber-600" />
                  </span>
                ) : (
                  s.sentence
                )}{' '}
              </span>
            ))}
          </p>
        ) : (
          <p className="whitespace-pre-line">{result.summary}</p>
        )}
      </div>
    </Card>
  )
}
