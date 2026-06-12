export default function TabNav({ tabs, active, onChange }) {
  return (
    <nav className="tab-nav" role="tablist" aria-label="Dashboard sections">
      {tabs.map((tab) => (
        <button
          key={tab.id}
          type="button"
          role="tab"
          aria-selected={active === tab.id}
          className={`tab-nav__item ${active === tab.id ? "is-active" : ""}`}
          onClick={() => onChange(tab.id)}
        >
          {tab.label}
          {tab.badge != null && <span className="tab-nav__badge">{tab.badge}</span>}
        </button>
      ))}
    </nav>
  );
}
