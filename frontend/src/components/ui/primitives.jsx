import Icon from './Icon'

export function Card({ title, subtitle, actions, children, className = '' }) {
  return (
    <section className={`rounded-2xl border border-slate-200 bg-white shadow-sm ${className}`}>
      {(title || actions) && (
        <header className="flex flex-wrap items-start justify-between gap-3 border-b border-slate-100 px-5 py-4">
          <div>
            {title && <h2 className="text-base font-semibold text-slate-900">{title}</h2>}
            {subtitle && <p className="mt-0.5 text-sm text-slate-500">{subtitle}</p>}
          </div>
          {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
        </header>
      )}
      <div className="p-5">{children}</div>
    </section>
  )
}

const BUTTON_STYLES = {
  primary: 'bg-indigo-600 text-white hover:bg-indigo-700 disabled:bg-indigo-300',
  secondary: 'bg-white text-slate-700 ring-1 ring-slate-300 hover:bg-slate-50 disabled:text-slate-400',
  ghost: 'text-slate-600 hover:bg-slate-100 disabled:text-slate-400',
  danger: 'bg-white text-rose-700 ring-1 ring-rose-200 hover:bg-rose-50',
}

export function Button({ variant = 'secondary', icon, children, className = '', size = 'md', ...props }) {
  const sizes = size === 'lg' ? 'px-5 py-3 text-base' : size === 'sm' ? 'px-2.5 py-1.5 text-xs' : 'px-3.5 py-2 text-sm'
  return (
    <button
      type="button"
      className={`inline-flex items-center justify-center gap-2 rounded-lg font-medium transition disabled:cursor-not-allowed ${sizes} ${BUTTON_STYLES[variant]} ${className}`}
      {...props}
    >
      {icon && <Icon name={icon} className={size === 'sm' ? 'h-4 w-4' : 'h-5 w-5'} />}
      {children}
    </button>
  )
}

const BANNER_STYLES = {
  error: { box: 'bg-rose-50 text-rose-800 ring-rose-200', icon: 'alert', label: 'Error' },
  warning: { box: 'bg-amber-50 text-amber-900 ring-amber-200', icon: 'alert', label: 'Warning' },
  info: { box: 'bg-sky-50 text-sky-900 ring-sky-200', icon: 'info', label: 'Note' },
}

/** Status banner: always icon + label + text, never color alone. */
export function Banner({ kind = 'info', title, children, onClose }) {
  const style = BANNER_STYLES[kind]
  return (
    <div role={kind === 'error' ? 'alert' : 'status'} className={`flex gap-3 rounded-xl p-4 text-sm ring-1 ${style.box}`}>
      <Icon name={style.icon} className="mt-0.5 h-5 w-5 shrink-0" />
      <div className="flex-1">
        <p className="font-semibold">{title ?? style.label}</p>
        {children && <div className="mt-1 leading-relaxed">{children}</div>}
      </div>
      {onClose && (
        <button type="button" onClick={onClose} className="self-start rounded p-1 hover:bg-black/5" aria-label="Dismiss">
          <Icon name="x" className="h-4 w-4" />
        </button>
      )}
    </div>
  )
}

export function Badge({ children, tone = 'slate' }) {
  const tones = {
    slate: 'bg-slate-100 text-slate-700',
    indigo: 'bg-indigo-50 text-indigo-700 ring-1 ring-indigo-200',
    emerald: 'bg-emerald-50 text-emerald-800 ring-1 ring-emerald-200',
    amber: 'bg-amber-50 text-amber-900 ring-1 ring-amber-200',
  }
  return <span className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium ${tones[tone]}`}>{children}</span>
}

/** Stat tile: label, value, optional hint. */
export function Stat({ label, value, hint }) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4">
      <p className="text-xs font-medium text-slate-500">{label}</p>
      <p className="mt-1 text-2xl font-semibold tabular-nums text-slate-900">{value}</p>
      {hint && <p className="mt-1 text-xs text-slate-500">{hint}</p>}
    </div>
  )
}

/** Collapsible section using native <details> (keyboard accessible for free). */
export function Collapsible({ title, children, defaultOpen = false }) {
  return (
    <details open={defaultOpen} className="group rounded-xl border border-slate-200 bg-white">
      <summary className="flex cursor-pointer list-none items-center gap-2 px-4 py-3 text-sm font-medium text-slate-800">
        <Icon name="chevron" className="h-4 w-4 transition group-open:rotate-90" />
        {title}
      </summary>
      <div className="border-t border-slate-100 p-4">{children}</div>
    </details>
  )
}
