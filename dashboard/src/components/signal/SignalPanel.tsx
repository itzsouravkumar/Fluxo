export function SignalPanel({ junction }: { junction?: any }) {
  const j = junction || {
    name: 'Veerannapalya Jn.',
    signal_phase: 'N-S',
    signal_remaining: 23,
    signal_state: 'GREEN',
    rl_recommendation: { phase: 'N-S', duration_s: 45, reason: 'rl_agent_recommendation' },
    lane_states: {},
  }

  const lanes = j.lane_states || {}
  const nq = lanes.north?.queue_length || 0
  const sq = lanes.south?.queue_length || 0
  const eq = lanes.east?.queue_length || 0
  const wq = lanes.west?.queue_length || 0

  const rlRec = j.rl_recommendation || {}
  const recDuration = rlRec.duration_s || 0
  const recReason = rlRec.reason || 'none'

  return (
    <div className="space-y-3">
      <div className="text-sm text-gray-400">{j.name || 'Junction'}</div>
      <div className="text-2xl font-mono text-fluxo-green">
        {j.signal_phase || 'N-S'} {j.signal_state || 'GREEN'} {j.signal_remaining || 0}s
      </div>
      {recDuration > 0 && (
        <div className="text-sm text-fluxo-accent">
          RL Rec: {recDuration}s ({recReason})
        </div>
      )}
      <div className="grid grid-cols-2 gap-2 text-xs mt-4">
        <div className="bg-gray-800 rounded p-2">
          <div className="text-gray-500">North</div>
          <div className="text-white">{Math.round(nq * 100)}m</div>
        </div>
        <div className="bg-gray-800 rounded p-2">
          <div className="text-gray-500">South</div>
          <div className="text-white">{Math.round(sq * 100)}m</div>
        </div>
        <div className="bg-gray-800 rounded p-2">
          <div className="text-gray-500">East</div>
          <div className="text-white">{Math.round(eq * 100)}m</div>
        </div>
        <div className="bg-gray-800 rounded p-2">
          <div className="text-gray-500">West</div>
          <div className="text-white">{Math.round(wq * 100)}m</div>
        </div>
      </div>
      <div className="flex gap-2 mt-4">
        <button className="px-4 py-2 bg-fluxo-green text-white rounded text-sm">Apply</button>
        <button className="px-4 py-2 bg-fluxo-red text-white rounded text-sm">Override</button>
      </div>
    </div>
  )
}
