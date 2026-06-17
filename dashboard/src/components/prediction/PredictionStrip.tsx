export function PredictionStrip() {
  return (
    <div className="flex gap-8">
      <div>
        <span className="text-gray-400">+15min:</span>
        <span className="ml-2 text-fluxo-critical font-semibold">CRITICAL (0.89)</span>
      </div>
      <div>
        <span className="text-gray-400">+30min:</span>
        <span className="ml-2 text-fluxo-critical font-semibold">CRITICAL (0.93)</span>
      </div>
    </div>
  )
}
