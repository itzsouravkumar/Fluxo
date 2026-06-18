export function StatusBar({ connected = true, junctionCount = 4 }: { connected?: boolean, junctionCount?: number }) {
  return (
    <footer className="bg-fluxo-card border-t border-fluxo-border px-4 py-2 flex items-center justify-between text-xs text-gray-500">
      <div className="flex items-center gap-4">
        <span className={connected ? 'text-fluxo-green' : 'text-fluxo-red'}>
          Backend: {connected ? 'Connected' : 'Disconnected'}
        </span>
        <span>WebSocket: {connected ? 'Active' : 'Inactive'}</span>
        <span>Junctions: {junctionCount} monitored</span>
      </div>
      <div>
        FLUXO v0.1.0 | Gridlock Hackathon 2.0
      </div>
    </footer>
  )
}
