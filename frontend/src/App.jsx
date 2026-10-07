import { NavLink, Route, Routes } from 'react-router-dom'
import BackendStatus from './components/BackendStatus'
import AboutPage from './pages/AboutPage'
import HistoryPage from './pages/HistoryPage'
import SummarizePage from './pages/SummarizePage'

const NAV = [
  { to: '/', label: 'Summarize', end: true },
  { to: '/history', label: 'History' },
  { to: '/about', label: 'How it works' },
]

export default function App() {
  return (
    <div className="flex min-h-screen flex-col">
      <header className="sticky top-0 z-10 border-b border-slate-200 bg-white/90 backdrop-blur">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-3 px-4 py-3 sm:px-6">
          <NavLink to="/" className="flex items-center gap-2">
            <img src="/favicon.svg" alt="" className="h-7 w-7" />
            <span className="text-lg font-semibold text-slate-900">IntelliSum</span>
          </NavLink>
          <nav className="order-3 flex w-full gap-1 sm:order-none sm:w-auto" aria-label="Main">
            {NAV.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.end}
                className={({ isActive }) =>
                  `rounded-lg px-3 py-1.5 text-sm font-medium transition ${
                    isActive ? 'bg-indigo-50 text-indigo-700' : 'text-slate-600 hover:bg-slate-100 hover:text-slate-900'
                  }`
                }
              >
                {item.label}
              </NavLink>
            ))}
          </nav>
          <BackendStatus />
        </div>
      </header>

      <main className="flex-1">
        <Routes>
          <Route path="/" element={<SummarizePage />} />
          <Route path="/history" element={<HistoryPage />} />
          <Route path="/about" element={<AboutPage />} />
          <Route path="*" element={<SummarizePage />} />
        </Routes>
      </main>

      <footer className="border-t border-slate-200 py-4 text-center text-xs text-slate-500">
        IntelliSum · B.Tech CSE Minor Project · NLP &amp; Transformers
      </footer>
    </div>
  )
}
