import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    // During development, forward /api/* to the FastAPI backend so the
    // frontend can use relative URLs and we avoid CORS issues entirely.
    proxy: {
      '/api': {
        target: process.env.VITE_BACKEND_URL || 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.js'],
    css: false,
    // The default 5 s per test is too tight for UI tests that type text when
    // the machine is busy (e.g. while BART runs); real failures still fail fast.
    testTimeout: 20000,
  },
})
