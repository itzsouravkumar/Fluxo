export function Header() {
  return (
    <header className="bg-fluxo-card border-b border-fluxo-border px-4 py-3 flex items-center justify-between">
      <div className="flex items-center gap-3">
        <h1 className="text-xl font-bold tracking-tight">FLUXO</h1>
        <span className="px-2 py-0.5 bg-fluxo-green/20 text-fluxo-green text-xs rounded-full font-medium">
          LIVE
        </span>
      </div>
      <div className="text-sm text-gray-400">
        {new Date().toLocaleDateString('en-IN', {
          day: 'numeric',
          month: 'short',
          year: 'numeric',
        })}
        {' '}
        {new Date().toLocaleTimeString('en-IN')}
      </div>
    </header>
  )
}
