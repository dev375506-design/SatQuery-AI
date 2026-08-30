/**
 * Writes frontend/config.js from VITE_API_URL (or API_URL).
 * Used by Vercel build. No extra npm packages.
 */
const fs = require('fs');
const path = require('path');

const raw = process.env.VITE_API_URL || process.env.API_URL || 'http://127.0.0.1:8000';
const url = String(raw).trim().replace(/\/$/, '');
const dest = path.join(__dirname, 'frontend', 'config.js');
const body =
  '/* Generated at build time from VITE_API_URL. Public API base URL only -- no secrets. */\n' +
  'window.VITE_API_URL = ' + JSON.stringify(url) + ';\n';

fs.writeFileSync(dest, body, 'utf8');
console.log('Wrote', dest, '→', url);
