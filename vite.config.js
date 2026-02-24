import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(async () => {
  let version = 'dev';
  try {
    const res = await fetch('https://api.github.com/repos/yapici/assaycurvefit/releases/latest');
    if (res.ok) {
      const data = await res.json();
      version = data.tag_name || 'dev';
    }
  } catch (e) {
    // Offline or API error - fall back to 'dev'
  }

  return {
    plugins: [react()],
    base: '/',
    define: {
      '__APP_VERSION__': JSON.stringify(version),
    },
  };
});