import React, { useEffect, useState } from 'react'
import ModuleCard from './components/ModuleCard'

const modules = [
  {
    id: 'autoheal-pipeline',
    title: 'Autoheal Pipeline',
    description: 'Production agentic pipeline module — access and monitor the pipeline.',
    url: 'http://localhost:8501'
  }
]

export default function App() {
  const [theme, setTheme] = useState(localStorage.getItem('mn-theme') || 'light')

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    localStorage.setItem('mn-theme', theme)
  }, [theme])

  const toggleTheme = () => setTheme((t) => (t === 'light' ? 'dark' : 'light'))

  return (
    <div className="app-root">
      <header className="header">
        <div>
          <h1 className="title">Agentic Nexus</h1>
          <p className="subtitle">Agentic Modules Hub — a unified site for your modules</p>
        </div>
        <div className="controls">
          <button className="theme-toggle" onClick={toggleTheme} aria-label="Toggle theme">
            {theme === 'light' ? '🌙 Dark' : '☀️ Light'}
          </button>
        </div>
      </header>

      <main className="main">
        <section className="modules-grid">
          {modules.map((m) => (
            <ModuleCard key={m.id} title={m.title} description={m.description} url={m.url} />
          ))}
        </section>
        <section className="info">
          <h2>About</h2>
          <p>
            This hub surfaces your agentic modules . Click any module
            to open it in a new tab. Start your module server (for example, the Streamlit module) before
            clicking the card.
          </p>
        </section>
      </main>

      <footer className="footer">Agentic Nexus — built for your agentic modules</footer>
    </div>
  )
}
