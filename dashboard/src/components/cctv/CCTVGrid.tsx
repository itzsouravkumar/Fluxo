export function CCTVGrid({ junctions = {} }: { junctions?: Record<string, any> }) {
  const jList = Object.values(junctions)

  const densityColor = (score: number) => {
    if (score > 0.7) return 'border-fluxo-red'
    if (score > 0.4) return 'border-fluxo-yellow'
    return 'border-fluxo-green'
  }

  return (
    <div className="grid grid-cols-2 gap-2">
      {jList.length > 0 ? jList.map((j) => (
        <div key={j.junction_id} className={`aspect-video bg-gray-900 rounded flex flex-col items-center justify-center text-sm border-2 ${densityColor(j.density_score || 0)}`}>
          <div className="text-gray-400">{j.name || j.junction_id}</div>
          <div className="text-xs text-gray-500 mt-1">
            {j.vehicle_count || 0} vehicles | {(j.density_score || 0).toFixed(2)}
          </div>
          <div className="text-xs mt-1 text-fluxo-accent">
            {j.signal_phase || 'N-S'} {j.signal_state || 'GREEN'}
          </div>
        </div>
      )) : (
        <>
          {[1, 2, 3, 4].map((i) => (
            <div key={i} className="aspect-video bg-gray-900 rounded flex items-center justify-center text-gray-500 text-sm">
              CAM {i}
            </div>
          ))}
        </>
      )}
    </div>
  )
}
