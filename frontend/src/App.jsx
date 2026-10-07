import BackendStatus from './components/BackendStatus'
import HomePage from './pages/HomePage'

export default function App() {
  return (
    <div className="flex min-h-screen flex-col">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-5xl items-center justify-between px-4 py-3 sm:px-6">
          <div className="flex items-center gap-2">
            <img src="/favicon.svg" alt="" className="h-7 w-7" />
            <span className="text-lg font-semibold text-slate-900">IntelliSum</span>
          </div>
          <BackendStatus />
        </div>
      </header>

      <main className="flex-1">
        <HomePage />
      </main>

      <footer className="border-t border-slate-200 py-4 text-center text-xs text-slate-500">
        IntelliSum · B.Tech CSE Minor Project · NLP &amp; Transformers
      </footer>
    </div>
  )
}
