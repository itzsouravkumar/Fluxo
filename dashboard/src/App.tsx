import { Header } from './components/layout/Header'
import { StatusBar } from './components/layout/StatusBar'
import { SignalPanel } from './components/signal/SignalPanel'
import { CCTVGrid } from './components/cctv/CCTVGrid'
import { ViolationFeed } from './components/violations/ViolationFeed'
import { PredictionStrip } from './components/prediction/PredictionStrip'
import { useFluxoData } from './hooks/useFluxoData'

function App() {
  const { junctions, violations, connected } = useFluxoData()
  const jList = Object.values(junctions)
  const mainJunction = jList[0] || { name: 'Veerannapalya Jn.', signal_phase: 'N-S', signal_remaining: 23, signal_state: 'GREEN', rl_recommendation: {}, lane_states: {} }

  return (
    <div className="min-h-screen bg-fluxo-bg">
      <Header />
      <main className="p-4">
        <div className="grid grid-cols-12 gap-4">
          {/* Junction Overview */}
          <div className="col-span-8 bg-fluxo-card rounded-lg border border-fluxo-border p-4">
            <h2 className="text-lg font-semibold mb-4">Junction Overview</h2>
            <div className="grid grid-cols-2 gap-3">
              {jList.map((j) => (
                <div key={j.junction_id} className="bg-gray-900 rounded p-3">
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-sm font-medium">{j.name}</span>
                    <span className={`text-xs px-2 py-0.5 rounded-full ${
                      j.congestion_level === 'CRITICAL' ? 'bg-fluxo-red/20 text-fluxo-red' :
                      j.congestion_level === 'HIGH' ? 'bg-fluxo-yellow/20 text-fluxo-yellow' :
                      'bg-fluxo-green/20 text-fluxo-green'
                    }`}>
                      {j.congestion_level || 'CLEAR'}
                    </span>
                  </div>
                  <div className="text-2xl font-mono text-fluxo-green mb-1">
                    {j.signal_phase || 'N-S'} {j.signal_state || 'GREEN'} {j.signal_remaining || 0}s
                  </div>
                  <div className="flex justify-between text-xs text-gray-500">
                    <span>{j.vehicle_count || 0} vehicles</span>
                    <span>Density: {(j.density_score || 0).toFixed(2)}</span>
                  </div>
                  {j.rl_recommendation?.duration_s > 0 && (
                    <div className="text-xs text-fluxo-accent mt-1">
                      RL: {j.rl_recommendation.duration_s}s
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>

          {/* Signal Controller Panel */}
          <div className="col-span-4 bg-fluxo-card rounded-lg border border-fluxo-border p-4">
            <h2 className="text-lg font-semibold mb-4">Signal Controller</h2>
            <SignalPanel junction={mainJunction} />
          </div>

          {/* CCTV Grid */}
          <div className="col-span-6 bg-fluxo-card rounded-lg border border-fluxo-border p-4">
            <h2 className="text-lg font-semibold mb-4">CCTV Grid</h2>
            <CCTVGrid junctions={junctions} />
          </div>

          {/* Violation Feed */}
          <div className="col-span-6 bg-fluxo-card rounded-lg border border-fluxo-border p-4">
            <h2 className="text-lg font-semibold mb-4">Violation Feed</h2>
            <ViolationFeed violations={violations} />
          </div>

          {/* Prediction Strip */}
          <div className="col-span-12 bg-fluxo-card rounded-lg border border-fluxo-border p-4">
            <h2 className="text-lg font-semibold mb-2">Congestion Prediction</h2>
            <PredictionStrip junctions={junctions} />
          </div>
        </div>
      </main>
      <StatusBar connected={connected} junctionCount={jList.length} />
    </div>
  )
}

export default App
