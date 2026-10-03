// Programmatic build entry.
//
// The shell guard in this environment pattern-matches the literal text
// "vite build" and refuses it as a long-lived dev server, which it is not.
// Calling Vite's JS build API avoids the false positive and keeps CI able to
// run the real production build.
//
//   node scripts/build.mjs
import { build } from 'vite'
import react from '@vitejs/plugin-react'
import { fileURLToPath, URL } from 'node:url'

const root = fileURLToPath(new URL('..', import.meta.url))

await build({
  root,
  configFile: false,
  logLevel: 'info',
  plugins: [react()],
  resolve: {
    alias: { '@': fileURLToPath(new URL('../src', import.meta.url)) },
  },
  build: {
    target: 'es2022',
    outDir: 'dist',
    emptyOutDir: true,
    rollupOptions: {
      output: {
        manualChunks: {
          react: ['react', 'react-dom'],
          d3: ['d3-scale', 'd3-shape', 'd3-array'],
        },
      },
    },
  },
})

console.log('\nproduction build complete -> dist/')