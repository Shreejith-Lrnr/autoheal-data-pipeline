import React from 'react'

export default function ModuleCard({ title, description, url }) {
  function openModule() {
    try {
      window.open(url, '_blank', 'noopener')
    } catch (e) {
      // fallback
      window.location.href = url
    }
  }

  return (
    <div className="module-card" onClick={openModule} role="button" tabIndex={0} onKeyDown={(e)=>{if(e.key==='Enter') openModule()}}>
      <div className="module-card__icon">🔌</div>
      <div className="module-card__body">
        <h3 className="module-card__title">{title}</h3>
        <p className="module-card__desc">{description}</p>
        <div className="module-card__cta">Open in new tab ↗</div>
      </div>
    </div>
  )
}
