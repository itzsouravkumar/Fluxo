export function StatusBar() {
  return (
    <footer className="bg-fluxo-card border-t border-fluxo-border px-4 py-2 flex items-center justify-between text-xs text-gray-500">
      <div className="flex items-center gap-4">
        <span>Backend: Connected</span>
        <span>WebSocket: Active</span>
        <span>Junctions: 4 monitored</span>
      </div>
      <div>
        FLUXO v0.1.0 | Gridlock Hackathon 2.0
      </div>
    </footer>
  )
}
