/**
 * Accessible single-choice group (radio semantics) rendered as cards or a
 * segmented control. Arrow keys move between options.
 */
export default function OptionGroup({ label, options, value, onChange, variant = 'segmented', disabled }) {
  const index = options.findIndex((o) => o.value === value)

  const onKeyDown = (e) => {
    const step = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[e.key]
    if (!step) return
    e.preventDefault()
    onChange(options[(index + step + options.length) % options.length].value)
  }

  if (variant === 'cards') {
    return (
      <fieldset disabled={disabled}>
        <legend className="mb-2 text-sm font-medium text-slate-700">{label}</legend>
        <div role="radiogroup" aria-label={label} onKeyDown={onKeyDown} className="grid grid-cols-2 gap-2">
          {options.map((option) => {
            const active = option.value === value
            return (
              <button
                key={option.value}
                type="button"
                role="radio"
                aria-checked={active}
                aria-label={option.tag ? `${option.label} (${option.tag})` : option.label}
                aria-description={option.description}
                tabIndex={active ? 0 : -1}
                onClick={() => onChange(option.value)}
                className={`rounded-xl border p-3 text-left transition ${
                  active
                    ? 'border-indigo-500 bg-indigo-50 ring-1 ring-indigo-500'
                    : 'border-slate-200 bg-white hover:border-slate-300'
                }`}
              >
                <span className="block font-semibold text-slate-900">{option.label}</span>
                {option.tag && (
                  <span className="block text-[11px] font-medium uppercase tracking-wide text-indigo-600">{option.tag}</span>
                )}
                {option.description && <span className="mt-1 block text-xs leading-snug text-slate-600">{option.description}</span>}
              </button>
            )
          })}
        </div>
      </fieldset>
    )
  }

  return (
    <fieldset disabled={disabled}>
      <legend className="mb-2 text-sm font-medium text-slate-700">{label}</legend>
      <div role="radiogroup" aria-label={label} onKeyDown={onKeyDown} className="inline-flex w-full rounded-xl bg-slate-100 p-1">
        {options.map((option) => {
          const active = option.value === value
          return (
            <button
              key={option.value}
              type="button"
              role="radio"
              aria-checked={active}
              tabIndex={active ? 0 : -1}
              onClick={() => onChange(option.value)}
              className={`flex-1 rounded-lg px-3 py-2 text-sm font-medium transition ${
                active ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              {option.label}
              {option.hint && <span className="ml-1 text-xs font-normal text-slate-500">{option.hint}</span>}
            </button>
          )
        })}
      </div>
    </fieldset>
  )
}
