import { Header } from './components/layout/Header'
import { StatusBar } from './components/layout/StatusBar'
import { SignalPanel } from './components/signal/SignalPanel'
import { CCTVGrid } from './components/cctv/CCTVGrid'
import { ViolationFeed } from './components/violations/ViolationFeed'
import { PredictionStrip } from './components/prediction/PredictionStrip'
import { FluxoMap } from './components/map/FluxoMap'
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
          {/* Map */}
          <div className="col-span-8 bg-fluxo-card rounded-lg border border-fluxo-border p-4">
            <h2 className="text-lg font-semibold mb-4">Live Junction Map</h2>
            <FluxoMap junctions={junctions} />
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
