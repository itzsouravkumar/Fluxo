/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        fluxo: {
          bg: '#0a0e17',
          card: '#111827',
          border: '#1f2937',
          accent: '#3b82f6',
          green: '#22c55e',
          yellow: '#eab308',
          red: '#ef4444',
          critical: '#dc2626',
        },
      },
    },
  },
  plugins: [],
}
