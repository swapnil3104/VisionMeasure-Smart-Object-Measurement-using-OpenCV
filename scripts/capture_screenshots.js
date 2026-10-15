const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

const EDGE_PATH = fs.existsSync('C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe')
  ? 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe'
  : 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
const PORT = 9222;
const BASE_URL = 'http://127.0.0.1:8000';
const OUT_DIR = path.resolve(__dirname, '../docs/screenshots');

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function sendCdp(ws, method, params = {}) {
  return new Promise((resolve, reject) => {
    const id = Math.floor(Math.random() * 1000000);
    const handler = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.id === id) {
          ws.removeEventListener('message', handler);
          if (data.error) {
            reject(new Error(data.error.message || JSON.stringify(data.error)));
          } else {
            resolve(data.result);
          }
        }
      } catch (err) {
        // ignore other messages
      }
    };
    ws.addEventListener('message', handler);
    ws.send(JSON.stringify({ id, method, params }));
  });
}

async function main() {
  if (!fs.existsSync(OUT_DIR)) {
    fs.mkdirSync(OUT_DIR, { recursive: true });
  }

  console.log('Launching headless Edge...');
  const edge = spawn(EDGE_PATH, [
    '--headless=new',
    '--disable-gpu',
    `--remote-debugging-port=${PORT}`,
    '--window-size=1600,1050',
    'about:blank',
  ]);

  edge.stderr.on('data', (d) => {});

  let wsUrl = null;
  for (let i = 0; i < 20; i++) {
    await sleep(500);
    try {
      const res = await fetch(`http://127.0.0.1:${PORT}/json/list`);
      const targets = await res.json();
      if (targets && targets.length > 0 && targets[0].webSocketDebuggerUrl) {
        wsUrl = targets[0].webSocketDebuggerUrl;
        break;
      }
    } catch (e) {
      // wait
    }
  }

  if (!wsUrl) {
    console.error('Could not connect to Edge DevTools');
    edge.kill();
    process.exit(1);
  }

  console.log('Connecting to CDP:', wsUrl);
  const ws = new WebSocket(wsUrl);

  await new Promise((resolve, reject) => {
    ws.onopen = resolve;
    ws.onerror = reject;
  });

  console.log('Connected! Enabling Page & Runtime...');
  await sendCdp(ws, 'Page.enable');
  await sendCdp(ws, 'Runtime.enable');
  await sendCdp(ws, 'Emulation.setDeviceMetricsOverride', {
    width: 1600,
    height: 1050,
    deviceScaleFactor: 1,
    mobile: false,
  });

  // 1. Capture Main App with Detection
  console.log('Navigating to', BASE_URL);
  await sendCdp(ws, 'Page.navigate', { url: BASE_URL });
  await sleep(2000);

  // Switch to samples tab and click the first sample card
  console.log('Switching to samples tab and loading benchmark image...');
  await sendCdp(ws, 'Runtime.evaluate', {
    expression: `
      (async () => {
        const samplesTab = document.getElementById('tabSamples');
        if (samplesTab) samplesTab.click();
        await new Promise(r => setTimeout(r, 600));
        const sampleCards = document.querySelectorAll('.sample-card');
        if (sampleCards.length > 0) {
          sampleCards[0].click();
        }
      })();
    `,
    awaitPromise: true,
  });

  // Wait for YOLO detection to run and render
  console.log('Waiting for YOLO inference and canvas rendering...');
  await sleep(4000);

  // Capture Screenshot 1: Web App Dashboard with Detections
  console.log('Capturing Screenshot 1: web_app_dashboard.png');
  const shot1 = await sendCdp(ws, 'Page.captureScreenshot', {
    format: 'png',
    captureBeyondViewport: false,
  });
  fs.writeFileSync(path.join(OUT_DIR, 'web_app_dashboard.png'), Buffer.from(shot1.data, 'base64'));

  // 2. Open 2-Point Calibration Ruler Modal
  console.log('Triggering 2-Point Calibration Ruler modal...');
  await sendCdp(ws, 'Runtime.evaluate', {
    expression: `
      (() => {
        const modal = document.getElementById('calibModal');
        const pixelInput = document.getElementById('modalPixelDist');
        const knownInput = document.getElementById('modalKnownLength');
        if (pixelInput) pixelInput.value = '324.5 px';
        if (knownInput) knownInput.value = '8.56';
        if (modal) modal.style.display = 'flex';
      })();
    `,
  });
  await sleep(1000);

  console.log('Capturing Screenshot 2: calibration_ruler_modal.png');
  const shot2 = await sendCdp(ws, 'Page.captureScreenshot', {
    format: 'png',
  });
  fs.writeFileSync(path.join(OUT_DIR, 'calibration_ruler_modal.png'), Buffer.from(shot2.data, 'base64'));

  // Close modal
  await sendCdp(ws, 'Runtime.evaluate', {
    expression: `
      (() => {
        const modal = document.getElementById('calibModal');
        if (modal) modal.style.display = 'none';
      })();
    `,
  });
  await sleep(500);

  // 3. Capture Measurement Table and Statistics (scrolled down slightly)
  console.log('Scrolling down to measurement records table...');
  await sendCdp(ws, 'Runtime.evaluate', {
    expression: `window.scrollBy({ top: 380, behavior: 'instant' });`,
  });
  await sleep(600);

  console.log('Capturing Screenshot 3: measurement_results_table.png');
  const shot3 = await sendCdp(ws, 'Page.captureScreenshot', {
    format: 'png',
  });
  fs.writeFileSync(path.join(OUT_DIR, 'measurement_results_table.png'), Buffer.from(shot3.data, 'base64'));

  // 4. Capture FastAPI Swagger API Docs
  console.log('Navigating to API Docs:', `${BASE_URL}/docs`);
  await sendCdp(ws, 'Page.navigate', { url: `${BASE_URL}/docs` });
  await sleep(2500);

  console.log('Capturing Screenshot 4: api_swagger_docs.png');
  const shot4 = await sendCdp(ws, 'Page.captureScreenshot', {
    format: 'png',
  });
  fs.writeFileSync(path.join(OUT_DIR, 'api_swagger_docs.png'), Buffer.from(shot4.data, 'base64'));

  console.log('Finished capturing screenshots! Cleaning up...');
  ws.close();
  edge.kill();
  console.log('All screenshots successfully saved to:', OUT_DIR);
}

main().catch((err) => {
  console.error('Error during screenshot capture:', err);
  process.exit(1);
});
