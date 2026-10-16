// Print a film's edit list (window.EDIT: [{id, t0, t1}]) as one JSON line.   usage: node tools/edit_list.mjs films/<name>
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import puppeteer from 'puppeteer-core';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const rel = path.relative(ROOT, path.resolve(process.argv[2])).split(path.sep).join('/');
const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.json': 'application/json', '.webp': 'image/webp', '.ttf': 'font/ttf', '.css': 'text/css' };
const srv = http.createServer((req, res) => {
  let p = path.join(ROOT, decodeURIComponent(new URL(req.url, 'http://x').pathname));
  if (fs.existsSync(p) && fs.statSync(p).isDirectory()) p = path.join(p, 'index.html');
  if (!fs.existsSync(p)) { res.writeHead(404); return res.end(); }
  res.writeHead(200, { 'Content-Type': MIME[path.extname(p)] || 'application/octet-stream' }); fs.createReadStream(p).pipe(res);
}).listen(0, '127.0.0.1');
await new Promise((r) => srv.once('listening', r));
const CHROME = process.env.MG_CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const br = await puppeteer.launch({ executablePath: CHROME, headless: true, args: ['--use-angle=d3d11', '--enable-gpu', '--ignore-gpu-blocklist'] });
const pg = await br.newPage();
pg.on('pageerror', (e) => console.error('[page error]', e.message));
await pg.goto(`http://127.0.0.1:${srv.address().port}/${rel}/index.html`, { waitUntil: 'load' });
await pg.waitForFunction('window.__ready === true', { timeout: 120000 });
console.log(JSON.stringify(await pg.evaluate(() => window.EDIT)));
await br.close(); srv.close();
