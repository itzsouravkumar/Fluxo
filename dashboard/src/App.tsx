import { Header } from './components/layout/Header'
import { StatusBar } from './components/layout/StatusBar'

function App() {
  return (
    <div className="min-h-screen bg-fluxo-bg">
      <Header />
      <main className="p-4">
        <div className="grid grid-cols-12 gap-4">
          {/* Map Panel */}
          <div className="col-span-8 bg-fluxo-card rounded-lg border border-fluxo-border p-4">
            <h2 className="text-lg font-semibold mb-4">Live Junction Map</h2>
            <div className="h-96 bg-gray-900 rounded flex items-center justify-center text-gray-500">
              MapmyIndia Map: Coming Soon
            </div>
          </div>

          {/* Signal Controller Panel */}
          <div className="col-span-4 bg-fluxo-card rounded-lg border border-fluxo-border p-4">
            <h2 className="text-lg font-semibold mb-4">Signal Controller</h2>
            <div className="space-y-3">
              <div className="text-sm text-gray-400">Veerannapalya Jn.</div>
              <div className="text-2xl font-mono text-fluxo-green">N-S GREEN 0:23</div>
              <div className="text-sm text-fluxo-accent">RL Rec: Extend 15s</div>
              <div className="flex gap-2 mt-4">
                <button className="px-4 py-2 bg-fluxo-green text-white rounded text-sm">Apply</button>
                <button className="px-4 py-2 bg-fluxo-red text-white rounded text-sm">Override</button>
              </div>
            </div>
          </div>

          {/* CCTV Grid */}
          <div className="col-span-6 bg-fluxo-card rounded-lg border border-fluxo-border p-4">
            <h2 className="text-lg font-semibold mb-4">CCTV Grid</h2>
            <div className="grid grid-cols-2 gap-2">
              {[1, 2, 3, 4].map((i) => (
                <div key={i} className="aspect-video bg-gray-900 rounded flex items-center justify-center text-gray-500 text-sm">
                  CAM {i}
                </div>
              ))}
            </div>
          </div>

          {/* Violation Feed */}
          <div className="col-span-6 bg-fluxo-card rounded-lg border border-fluxo-border p-4">
            <h2 className="text-lg font-semibold mb-4">Violation Feed</h2>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between items-center py-2 border-b border-fluxo-border">
                <span className="text-fluxo-yellow">No Helmet</span>
                <span className="text-gray-400">KA-05-MJ-4421</span>
                <span className="text-gray-500">17:42:11</span>
              </div>
              <div className="flex justify-between items-center py-2 border-b border-fluxo-border">
                <span className="text-fluxo-red">Signal Jump</span>
                <span className="text-gray-400">KA-09-HB-7823</span>
                <span className="text-gray-500">17:41:58</span>
              </div>
            </div>
          </div>

          {/* Prediction Strip */}
          <div className="col-span-12 bg-fluxo-card rounded-lg border border-fluxo-border p-4">
            <h2 className="text-lg font-semibold mb-2">Congestion Prediction</h2>
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
          </div>
        </div>
      </main>
      <StatusBar />
    </div>
  )
}

export default App
