export function ViolationFeed() {
  const violations = [
    { type: 'No Helmet', plate: 'KA-05-MJ-4421', time: '17:42:11', color: 'text-fluxo-yellow' },
    { type: 'Signal Jump', plate: 'KA-09-HB-7823', time: '17:41:58', color: 'text-fluxo-red' },
    { type: 'Wrong Way', plate: 'KA-51-NA-2291', time: '17:40:22', color: 'text-fluxo-red' },
  ]

  return (
    <div className="space-y-2 text-sm">
      {violations.map((v, i) => (
        <div key={i} className="flex justify-between items-center py-2 border-b border-fluxo-border">
          <span className={v.color}>{v.type}</span>
          <span className="text-gray-400">{v.plate}</span>
          <span className="text-gray-500">{v.time}</span>
          <button className="text-fluxo-accent text-xs">CLIP</button>
        </div>
      ))}
    </div>
  )
}
