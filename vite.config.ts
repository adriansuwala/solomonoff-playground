import { fileURLToPath, URL } from 'node:url'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

export default defineConfig({
  // Relative asset URLs so the build works from any subpath (and is portable
  // if it is ever copied next to other files). The default '/' resolves to
  // the filesystem root when the page is opened directly, which 404s.
  base: './',
  plugins: [react()],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  server: {
    fs: { strict: false },
  },
  build: {
    target: 'es2022',
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