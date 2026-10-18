// Tiny static file server for ./web (or $WEB_ROOT, e.g. studio/web). No deps.
import http from 'node:http';
import { readFile, stat } from 'node:fs/promises';
import { extname, join, normalize, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = process.env.WEB_ROOT
  ? resolve(process.env.WEB_ROOT)
  : join(fileURLToPath(new URL('.', import.meta.url)), 'web');
const PORT = parseInt(process.env.PORT || '8080', 10);

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js':   'text/javascript; charset=utf-8',
  '.wasm': 'application/wasm',
  '.json': 'application/json; charset=utf-8',
  '.css':  'text/css; charset=utf-8',
  '.svg':  'image/svg+xml',
  '.ico':  'image/x-icon',
};

http.createServer(async (req, res) => {
  try {
    let urlPath = decodeURIComponent((req.url || '/').split('?')[0]);
    if (urlPath === '/') urlPath = '/index.html';
    const safe = normalize(join(ROOT, urlPath));
    if (!safe.startsWith(ROOT)) { res.writeHead(403).end('forbidden'); return; }
    const s = await stat(safe);
    const path = s.isDirectory() ? join(safe, 'index.html') : safe;
    const body = await readFile(path);
    res.writeHead(200, {
      'content-type': MIME[extname(path).toLowerCase()] || 'application/octet-stream',
      'cache-control': 'no-store',
    });
    res.end(body);
  } catch (err) {
    if (err && err.code === 'ENOENT') res.writeHead(404).end('not found');
    else res.writeHead(500).end(String(err));
  }
}).listen(PORT, () => process.stdout.write(`QuPU dev server: http://localhost:${PORT}\n`));
