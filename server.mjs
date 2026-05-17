// Minimal static dev server that sets the COOP/COEP cross-origin isolation
// headers required for SharedArrayBuffer to be available in the browser.
// Serves files from ./web.

import http from 'node:http';
import { readFile, stat } from 'node:fs/promises';
import { extname, join, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = fileURLToPath(new URL('.', import.meta.url));
const ROOT = join(__dirname, 'web');
const PORT = process.env.PORT ? parseInt(process.env.PORT, 10) : 8080;

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js':   'text/javascript; charset=utf-8',
  '.mjs':  'text/javascript; charset=utf-8',
  '.css':  'text/css; charset=utf-8',
  '.wasm': 'application/wasm',
  '.json': 'application/json; charset=utf-8',
  '.map':  'application/json; charset=utf-8',
  '.svg':  'image/svg+xml',
  '.ico':  'image/x-icon',
};

const server = http.createServer(async (req, res) => {
  try {
    let urlPath = decodeURIComponent((req.url || '/').split('?')[0]);
    if (urlPath === '/') urlPath = '/index.html';
    const safePath = normalize(join(ROOT, urlPath));
    if (!safePath.startsWith(ROOT)) {
      res.writeHead(403); res.end('forbidden'); return;
    }

    const s = await stat(safePath);
    const filePath = s.isDirectory() ? join(safePath, 'index.html') : safePath;
    const data = await readFile(filePath);
    const mime = MIME[extname(filePath).toLowerCase()] || 'application/octet-stream';

    res.writeHead(200, {
      'content-type': mime,
      // Cross-origin isolation — required for SharedArrayBuffer in modern browsers.
      'cross-origin-opener-policy':   'same-origin',
      'cross-origin-embedder-policy': 'require-corp',
      'cross-origin-resource-policy': 'same-origin',
      // Hot iteration during dev.
      'cache-control': 'no-store',
    });
    res.end(data);
  } catch (err) {
    if (err && err.code === 'ENOENT') {
      res.writeHead(404, { 'content-type': 'text/plain' });
      res.end('404 not found');
    } else {
      res.writeHead(500, { 'content-type': 'text/plain' });
      res.end(String(err));
    }
  }
});

server.listen(PORT, () => {
  process.stdout.write(`QuPU dev server listening at http://localhost:${PORT}\n`);
});
