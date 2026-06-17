export function SignalPanel() {
  return (
    <div className="space-y-3">
      <div className="text-sm text-gray-400">Veerannapalya Jn.</div>
      <div className="text-2xl font-mono text-fluxo-green">N-S GREEN 0:23</div>
      <div className="text-sm text-fluxo-accent">RL Rec: Extend 15s</div>
      <div className="text-xs text-gray-500">Queue N: 847m | S: 312m</div>
      <div className="flex gap-2 mt-4">
        <button className="px-4 py-2 bg-fluxo-green text-white rounded text-sm">Apply</button>
        <button className="px-4 py-2 bg-fluxo-red text-white rounded text-sm">Override</button>
      </div>
    </div>
  )
}
