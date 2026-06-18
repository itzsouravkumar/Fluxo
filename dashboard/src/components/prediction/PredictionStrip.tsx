export function PredictionStrip({ junctions = {} }: { junctions?: Record<string, any> }) {
  const jList = Object.values(junctions)
  const avgDensity = jList.length > 0
    ? jList.reduce((sum, j) => sum + (j.density_score || 0), 0) / jList.length
    : 0

  const level = avgDensity > 0.7 ? 'CRITICAL' : avgDensity > 0.4 ? 'HIGH' : avgDensity > 0.2 ? 'MODERATE' : 'CLEAR'

  return (
    <div className="flex gap-8 flex-wrap">
      <div>
        <span className="text-gray-400">Current Avg:</span>
        <span className={`ml-2 font-semibold ${avgDensity > 0.7 ? 'text-fluxo-red' : avgDensity > 0.4 ? 'text-fluxo-yellow' : 'text-fluxo-green'}`}>
          {level} ({avgDensity.toFixed(2)})
        </span>
      </div>
      <div>
        <span className="text-gray-400">+15min:</span>
        <span className={`ml-2 font-semibold ${avgDensity > 0.6 ? 'text-fluxo-red' : 'text-fluxo-yellow'}`}>
          {avgDensity > 0.6 ? 'CRITICAL' : 'HIGH'} ({(avgDensity * 1.1).toFixed(2)})
        </span>
      </div>
      <div>
        <span className="text-gray-400">+30min:</span>
        <span className={`ml-2 font-semibold ${avgDensity > 0.5 ? 'text-fluxo-red' : 'text-fluxo-yellow'}`}>
          {avgDensity > 0.5 ? 'CRITICAL' : 'MODERATE'} ({(avgDensity * 0.95).toFixed(2)})
        </span>
      </div>
      <div>
        <span className="text-gray-400">Junctions Monitored:</span>
        <span className="ml-2 text-white">{jList.length}</span>
      </div>
    </div>
  )
}
