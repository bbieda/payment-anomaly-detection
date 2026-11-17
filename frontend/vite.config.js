import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// W trybie dev zapytania /api idą do FastAPI (uvicorn backend.app.main:app --reload).
export default defineConfig({
  plugins: [react()],
  build: { chunkSizeWarningLimit: 800 },
  server: { proxy: { '/api': 'http://127.0.0.1:8000' } },
  test: { environment: 'jsdom', setupFiles: ['./src/__tests__/setup.js'], globals: true },
});
