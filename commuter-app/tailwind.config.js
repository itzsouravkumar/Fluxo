/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        fluxo: {
          bg: '#0a0e17',
          card: '#111827',
          border: '#1f2937',
          green: '#10b981',
          red: '#ef4444',
          yellow: '#f59e0b',
          accent: '#3b82f6',
        },
      },
    },
  },
  plugins: [],
}
