export function ViolationFeed({ violations = [] }: { violations?: any[] }) {
  const formatTime = (ts: number) => {
    if (!ts) return '--:--:--'
    const d = new Date(ts * 1000)
    return d.toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit', second: '2-digit' })
  }

  const typeColor = (type: string) => {
    switch (type) {
      case 'signal_jump': return 'text-fluxo-red'
      case 'wrong_way': return 'text-fluxo-red'
      case 'no_helmet': return 'text-fluxo-yellow'
      case 'triple_riding': return 'text-fluxo-yellow'
      default: return 'text-gray-400'
    }
  }

  const typeLabel = (type: string) => {
    return type.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ')
  }

  return (
    <div className="space-y-2 text-sm max-h-64 overflow-y-auto">
      {violations.length === 0 ? (
        <div className="text-gray-500 text-center py-4">No violations detected</div>
      ) : (
        violations.map((v, i) => (
          <div key={v.id || i} className="flex justify-between items-center py-2 border-b border-fluxo-border">
            <span className={typeColor(v.type)}>{typeLabel(v.type)}</span>
            <span className="text-gray-400 font-mono text-xs">{v.plate_number || 'N/A'}</span>
            <span className="text-gray-500 text-xs">{formatTime(v.timestamp)}</span>
            {v.clip_path && (
              <button className="text-fluxo-accent text-xs">CLIP</button>
            )}
          </div>
        ))
      )}
    </div>
  )
}
