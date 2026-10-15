/**
 * VisionMeasure AI — Frontend Application Logic
 * Coordinates YOLO detection, interactive 2-point calibration, webcam streaming,
 * canvas rendering, and data export.
 */

// Global Application State
const state = {
  currentImageSource: null,     // Base64 or ObjectURL of original image
  currentImageB64: null,        // Base64 data string
  annotatedImageB64: null,      // Base64 of YOLO annotated image
  detections: [],               // Array of detection result objects
  pixelsPerCm: 38.0,            // Default calibrated standard scale (px/cm)
  unit: 'cm',                   // 'cm', 'mm', 'in'
  twoPointMode: false,          // True when user is picking 2 calibration points
  pickedPoints: [],             // [{x, y}, {x, y}]
  webcamStream: null,           // MediaStream instance
  imageWidth: 0,
  imageHeight: 0,
};

// DOM Element References
const elements = {
  // Tabs
  tabUpload: document.getElementById('tabUpload'),
  tabWebcam: document.getElementById('tabWebcam'),
  tabSamples: document.getElementById('tabSamples'),
  uploadContainer: document.getElementById('uploadContainer'),
  webcamContainer: document.getElementById('webcamContainer'),
  samplesContainer: document.getElementById('samplesContainer'),

  // Inputs & Controls
  modelSelect: document.getElementById('modelSelect'),
  modelAccuracyBadge: document.getElementById('modelAccuracyBadge'),
  badgeVersion: document.getElementById('badgeVersion'),
  systemStatusText: document.getElementById('systemStatusText'),
  fileInput: document.getElementById('fileInput'),
  dropZone: document.getElementById('dropZone'),
  confSlider: document.getElementById('confSlider'),
  confValBadge: document.getElementById('confValBadge'),
  detectionMode: document.getElementById('detectionMode'),
  unitSelect: document.getElementById('unitSelect'),
  targetClasses: document.getElementById('targetClasses'),
  manualScaleInput: document.getElementById('manualScaleInput'),
  btnApplyManualScale: document.getElementById('btnApplyManualScale'),

  // Action Buttons
  btnDetect: document.getElementById('btnDetect'),
  btnTwoPointCalib: document.getElementById('btnTwoPointCalib'),
  btnStartWebcam: document.getElementById('btnStartWebcam'),
  btnFreezeWebcam: document.getElementById('btnFreezeWebcam'),
  webcamVideo: document.getElementById('webcamVideo'),

  // Stage & Canvas
  canvas: document.getElementById('mainImageCanvas'),
  emptyState: document.getElementById('emptyState'),
  loadingOverlay: document.getElementById('loadingOverlay'),
  canvasBanner: document.getElementById('canvasBanner'),
  canvasBannerText: document.getElementById('canvasBannerText'),
  viewAnnotated: document.getElementById('viewAnnotated'),
  viewOriginal: document.getElementById('viewOriginal'),

  // Metrics & Results
  statusScale: document.getElementById('statusScale'),
  calibScaleDisplay: document.getElementById('calibScaleDisplay'),
  metricTotal: document.getElementById('metricTotal'),
  metricLargest: document.getElementById('metricLargest'),
  metricSmallest: document.getElementById('metricSmallest'),
  metricAvg: document.getElementById('metricAvg'),
  measurementsTbody: document.getElementById('measurementsTbody'),
  thDimPhysical: document.getElementById('thDimPhysical'),
  thAreaPhysical: document.getElementById('thAreaPhysical'),

  // Exports
  btnExportCsv: document.getElementById('btnExportCsv'),
  btnExportJson: document.getElementById('btnExportJson'),
  btnDownloadImage: document.getElementById('btnDownloadImage'),

  // Ruler Modal
  calibModal: document.getElementById('calibModal'),
  modalClose: document.getElementById('modalClose'),
  modalCancel: document.getElementById('modalCancel'),
  modalConfirm: document.getElementById('modalConfirm'),
  modalPixelDist: document.getElementById('modalPixelDist'),
  modalKnownLength: document.getElementById('modalKnownLength'),
  modalUnitSelect: document.getElementById('modalUnitSelect'),

  // Object Calibration Modal
  objectCalibModal: document.getElementById('objectCalibModal'),
  objModalClose: document.getElementById('objModalClose'),
  objModalCancel: document.getElementById('objModalCancel'),
  objModalConfirm: document.getElementById('objModalConfirm'),
  objModalPixelLength: document.getElementById('objModalPixelLength'),
  objModalKnownLength: document.getElementById('objModalKnownLength'),
  objModalUnitSelect: document.getElementById('objModalUnitSelect'),
  objModalSubtitle: document.getElementById('objModalSubtitle'),
};

// Canvas 2D Context
const ctx = elements.canvas.getContext('2d');
let loadedImageObj = new Image();
let currentCalibObject = null;

// -----------------------------------------------------------------------------
// Initialization
// -----------------------------------------------------------------------------

document.addEventListener('DOMContentLoaded', () => {
  initEventListeners();
  loadSampleImages();
  checkSystemHealth();
  setScale(38.0);
});

function initEventListeners() {
  // Input Tabs
  elements.tabUpload.addEventListener('click', () => switchTab('upload'));
  elements.tabWebcam.addEventListener('click', () => switchTab('webcam'));
  elements.tabSamples.addEventListener('click', () => switchTab('samples'));

  // File Upload Drag & Drop
  elements.dropZone.addEventListener('click', () => elements.fileInput.click());
  elements.fileInput.addEventListener('change', handleFileSelect);

  elements.dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    elements.dropZone.classList.add('drag-over');
  });

  elements.dropZone.addEventListener('dragleave', () => {
    elements.dropZone.classList.remove('drag-over');
  });

  elements.dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    elements.dropZone.classList.remove('drag-over');
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      loadImageFromFile(e.dataTransfer.files[0]);
    }
  });

  // Confidence Slider
  elements.confSlider.addEventListener('input', (e) => {
    elements.confValBadge.textContent = `${Math.round(e.target.value * 100)}%`;
  });

  // Unit Selector
  elements.unitSelect.addEventListener('change', (e) => {
    state.unit = e.target.value;
    if (elements.thDimPhysical) elements.thDimPhysical.textContent = `Dimensions (${state.unit})`;
    if (elements.thAreaPhysical) elements.thAreaPhysical.textContent = `Area (${state.unit}² & px²)`;
    if (state.detections.length > 0) {
      renderMeasurementsTable(state.detections);
      runDetection();
    }
  });

  // Primary Detect Button
  elements.btnDetect.addEventListener('click', runDetection);

  // Model Architecture Selector
  if (elements.modelSelect) {
    elements.modelSelect.addEventListener('change', (e) => {
      const opt = e.target.selectedOptions[0];
      if (opt && elements.modelAccuracyBadge) {
        elements.modelAccuracyBadge.textContent = opt.dataset.accuracy || '';
      }
      if (opt && elements.badgeVersion) {
        elements.badgeVersion.textContent = (opt.dataset.tier ? `YOLOv8 ${opt.dataset.tier}` : 'YOLO Active');
      }
      if (opt && elements.systemStatusText) {
        elements.systemStatusText.textContent = `${opt.dataset.tier || 'YOLO'} Ready`;
      }
      if (state.currentImageB64) {
        runDetection();
      }
    });
  }

  // Two-Point Calibration Tool
  elements.btnTwoPointCalib.addEventListener('click', toggleTwoPointMode);

  // Canvas Click for 2-Point Selection
  elements.canvas.addEventListener('click', handleCanvasClick);

  // Calibration Presets
  document.querySelectorAll('.preset-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      if (btn.dataset.scale) {
        const sc = parseFloat(btn.dataset.scale);
        setScale(sc);
        if (state.currentImageB64) runDetection();
        return;
      }
      const length = parseFloat(btn.dataset.length);
      const unit = btn.dataset.unit || 'cm';
      applyPresetCalibration(length, unit, btn.dataset.label);
    });
  });

  // Manual Scale Apply
  elements.btnApplyManualScale.addEventListener('click', () => {
    const val = parseFloat(elements.manualScaleInput.value);
    if (!isNaN(val) && val > 0) {
      setScale(val);
      if (state.currentImageB64) runDetection();
    } else {
      alert('Please enter a positive numeric scale factor (pixels per cm).');
    }
  });

  elements.manualScaleInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      elements.btnApplyManualScale.click();
    }
  });

  // View Mode Toggles (Annotated vs Clean Original)
  elements.viewAnnotated.addEventListener('click', () => switchView('annotated'));
  elements.viewOriginal.addEventListener('click', () => switchView('original'));

  // Webcam Controls
  elements.btnStartWebcam.addEventListener('click', toggleWebcam);
  elements.btnFreezeWebcam.addEventListener('click', freezeWebcamFrame);

  // Ruler Modal Actions
  elements.modalClose.addEventListener('click', closeModal);
  elements.modalCancel.addEventListener('click', closeModal);
  elements.modalConfirm.addEventListener('click', confirmCalibrationModal);

  // Object Calibration Modal Actions
  if (elements.objModalClose) elements.objModalClose.addEventListener('click', closeObjectCalibModal);
  if (elements.objModalCancel) elements.objModalCancel.addEventListener('click', closeObjectCalibModal);
  if (elements.objModalConfirm) elements.objModalConfirm.addEventListener('click', confirmObjectCalibModal);

  // Export Actions
  elements.btnExportCsv.addEventListener('click', exportCsv);
  elements.btnExportJson.addEventListener('click', exportJson);
  elements.btnDownloadImage.addEventListener('click', downloadAnnotatedImage);
}

// -----------------------------------------------------------------------------
// System Health & Sample Images
// -----------------------------------------------------------------------------

async function checkSystemHealth() {
  try {
    const res = await fetch('/api/health');
    const data = await res.json();
    console.log('VisionMeasure AI Backend:', data);
  } catch (err) {
    console.warn('Backend connection warning:', err);
  }
}

async function loadSampleImages() {
  try {
    const res = await fetch('/api/sample_images');
    const data = await res.json();
    const grid = document.getElementById('sampleGrid');
    if (!grid || !data.samples) return;

    grid.innerHTML = '';
    data.samples.forEach((sample) => {
      const card = document.createElement('div');
      card.className = 'sample-card';
      card.innerHTML = `<strong>${sample.name}</strong><br><span style="font-size:0.65rem; color:#64748b;">${sample.filename}</span>`;
      card.addEventListener('click', () => loadSampleFromUrl(sample.url));
      grid.appendChild(card);
    });
  } catch (err) {
    console.warn('Failed to load sample images:', err);
  }
}

async function loadSampleFromUrl(url) {
  showLoading(true);
  try {
    const res = await fetch(url);
    const blob = await res.blob();
    loadImageFromFile(blob);
  } catch (err) {
    alert('Failed to load sample image: ' + err.message);
  } finally {
    showLoading(false);
  }
}

// -----------------------------------------------------------------------------
// Tab Switching
// -----------------------------------------------------------------------------

function switchTab(tab) {
  elements.tabUpload.classList.toggle('active', tab === 'upload');
  elements.tabWebcam.classList.toggle('active', tab === 'webcam');
  elements.tabSamples.classList.toggle('active', tab === 'samples');

  elements.uploadContainer.style.display = tab === 'upload' ? 'block' : 'none';
  elements.webcamContainer.style.display = tab === 'webcam' ? 'flex' : 'none';
  elements.samplesContainer.style.display = tab === 'samples' ? 'flex' : 'none';

  if (tab !== 'webcam' && state.webcamStream) {
    stopWebcam();
  }
}

// -----------------------------------------------------------------------------
// Image Input Handling
// -----------------------------------------------------------------------------

function handleFileSelect(e) {
  if (e.target.files && e.target.files[0]) {
    loadImageFromFile(e.target.files[0]);
  }
}

function loadImageFromFile(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    state.currentImageB64 = e.target.result;
    state.annotatedImageB64 = null;
    state.detections = [];

    loadedImageObj.onload = () => {
      state.imageWidth = loadedImageObj.naturalWidth;
      state.imageHeight = loadedImageObj.naturalHeight;
      renderImageToCanvas(loadedImageObj);
      elements.emptyState.style.display = 'none';
      elements.canvas.style.display = 'block';

      // Auto-run detection upon loading image
      runDetection();
    };
    loadedImageObj.src = state.currentImageB64;
  };
  reader.readAsDataURL(file);
}

// -----------------------------------------------------------------------------
// Webcam Handling
// -----------------------------------------------------------------------------

async function toggleWebcam() {
  if (state.webcamStream) {
    stopWebcam();
    elements.btnStartWebcam.innerHTML = `<span>▶ Start Camera</span>`;
    elements.btnFreezeWebcam.disabled = true;
  } else {
    try {
      state.webcamStream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'environment' }
      });
      elements.webcamVideo.srcObject = state.webcamStream;
      elements.btnStartWebcam.innerHTML = `<span>⏹ Stop Camera</span>`;
      elements.btnFreezeWebcam.disabled = false;
    } catch (err) {
      alert('Unable to access camera: ' + err.message);
    }
  }
}

function stopWebcam() {
  if (state.webcamStream) {
    state.webcamStream.getTracks().forEach(track => track.stop());
    state.webcamStream = null;
    elements.webcamVideo.srcObject = null;
  }
}

function freezeWebcamFrame() {
  if (!elements.webcamVideo.videoWidth) return;

  const tempCanvas = document.createElement('canvas');
  tempCanvas.width = elements.webcamVideo.videoWidth;
  tempCanvas.height = elements.webcamVideo.videoHeight;
  const tempCtx = tempCanvas.getContext('2d');
  tempCtx.drawImage(elements.webcamVideo, 0, 0);

  const b64 = tempCanvas.toDataURL('image/jpeg', 0.92);
  state.currentImageB64 = b64;
  state.annotatedImageB64 = null;

  loadedImageObj.onload = () => {
    state.imageWidth = loadedImageObj.naturalWidth;
    state.imageHeight = loadedImageObj.naturalHeight;
    renderImageToCanvas(loadedImageObj);
    elements.emptyState.style.display = 'none';
    elements.canvas.style.display = 'block';
    runDetection();
  };
  loadedImageObj.src = b64;
}

// -----------------------------------------------------------------------------
// Canvas Rendering & View Modes
// -----------------------------------------------------------------------------

function renderImageToCanvas(imgObj) {
  elements.canvas.width = imgObj.naturalWidth || imgObj.width;
  elements.canvas.height = imgObj.naturalHeight || imgObj.height;
  ctx.clearRect(0, 0, elements.canvas.width, elements.canvas.height);
  ctx.drawImage(imgObj, 0, 0);
}

function switchView(mode) {
  elements.viewAnnotated.classList.toggle('active', mode === 'annotated');
  elements.viewOriginal.classList.toggle('active', mode === 'original');

  if (mode === 'annotated' && state.annotatedImageB64) {
    const img = new Image();
    img.onload = () => renderImageToCanvas(img);
    img.src = state.annotatedImageB64;
  } else if (state.currentImageB64) {
    const img = new Image();
    img.onload = () => renderImageToCanvas(img);
    img.src = state.currentImageB64;
  }
}

// -----------------------------------------------------------------------------
// YOLO Detection & Measurement Engine Call
// -----------------------------------------------------------------------------

async function runDetection() {
  if (!state.currentImageB64) {
    alert('Please upload an image, select a sample, or capture a webcam frame first.');
    return;
  }

  showLoading(true);

  try {
    const formData = new FormData();
    formData.append('image_base64', state.currentImageB64);
    if (elements.modelSelect) {
      formData.append('model_name', elements.modelSelect.value);
    }
    formData.append('conf_threshold', elements.confSlider.value);
    formData.append('mode', elements.detectionMode.value);
    formData.append('unit', state.unit);
    const activeScale = (state.pixelsPerCm && state.pixelsPerCm > 0) ? state.pixelsPerCm : 38.0;
    formData.append('pixels_per_cm', activeScale);
    const target = elements.targetClasses.value.trim();
    if (target && target.toLowerCase() !== 'all') {
      formData.append('target_classes', target);
    }

    const res = await fetch('/api/detect', {
      method: 'POST',
      body: formData,
    });

    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || 'Detection failed');
    }

    state.detections = data.detections || [];
    state.annotatedImageB64 = data.annotated_image;

    // Display annotated view on canvas
    if (state.annotatedImageB64) {
      const annImg = new Image();
      annImg.onload = () => {
        renderImageToCanvas(annImg);
        elements.viewAnnotated.classList.add('active');
        elements.viewOriginal.classList.remove('active');
      };
      annImg.src = state.annotatedImageB64;
    }

    updateDashboard(data);
  } catch (err) {
    alert('Error running detection: ' + err.message);
  } finally {
    showLoading(false);
  }
}

// -----------------------------------------------------------------------------
// Dashboard & Results Table
// -----------------------------------------------------------------------------

function formatPhysicalDim(lenCm, widCm, unit) {
  if (lenCm === null || lenCm === undefined) return 'Uncalibrated';
  const u = (unit || state.unit || 'cm').toLowerCase();
  if (u === 'mm') {
    return `${(lenCm * 10).toFixed(1)} × ${(widCm * 10).toFixed(1)} mm`;
  }
  if (u === 'in') {
    return `${(lenCm / 2.54).toFixed(2)} × ${(widCm / 2.54).toFixed(2)} in`;
  }
  return `${lenCm.toFixed(2)} × ${widCm.toFixed(2)} cm`;
}

function updateDashboard(data) {
  const count = data.total_detected || 0;
  elements.metricTotal.textContent = count;
  const currentScale = state.pixelsPerCm || 38.0;

  if (data.summary && data.summary.largest_object) {
    const obj = data.summary.largest_object;
    const lCm = obj.length_cm ?? (obj.length_px / currentScale);
    const wCm = obj.width_cm ?? (obj.width_px / currentScale);
    const physStr = formatPhysicalDim(lCm, wCm, state.unit);
    const pxDim = `${obj.length_px.toFixed(1)} × ${obj.width_px.toFixed(1)} px`;

    elements.metricLargest.textContent = `${obj.class_name}`;
    document.getElementById('metricLargestSub').textContent = `${physStr} (${pxDim})`;
  } else {
    elements.metricLargest.textContent = 'None';
    document.getElementById('metricLargestSub').textContent = '—';
  }

  if (data.summary && data.summary.smallest_object) {
    const obj = data.summary.smallest_object;
    const lCm = obj.length_cm ?? (obj.length_px / currentScale);
    const wCm = obj.width_cm ?? (obj.width_px / currentScale);
    const physStr = formatPhysicalDim(lCm, wCm, state.unit);
    const pxDim = `${obj.length_px.toFixed(1)} × ${obj.width_px.toFixed(1)} px`;

    elements.metricSmallest.textContent = `${obj.class_name}`;
    document.getElementById('metricSmallestSub').textContent = `${physStr} (${pxDim})`;
  } else {
    elements.metricSmallest.textContent = 'None';
    document.getElementById('metricSmallestSub').textContent = '—';
  }

  if (data.summary && (data.summary.avg_length_cm || data.summary.avg_length_px)) {
    const avgCm = data.summary.avg_length_cm ?? (data.summary.avg_length_px / currentScale);
    const u = state.unit || 'cm';
    if (u === 'mm') {
      elements.metricAvg.textContent = `${(avgCm * 10).toFixed(1)} mm`;
    } else if (u === 'in') {
      elements.metricAvg.textContent = `${(avgCm / 2.54).toFixed(2)} in`;
    } else {
      elements.metricAvg.textContent = `${avgCm.toFixed(2)} cm`;
    }
    const avgPx = data.summary.avg_length_px ? `(${data.summary.avg_length_px.toFixed(0)} px)` : '';
    document.querySelector('#metricsGrid .metric-card:nth-child(4) .metric-sub').textContent = avgPx || 'Mean Physical Length';
  } else {
    elements.metricAvg.textContent = '—';
  }

  renderMeasurementsTable(state.detections);
}

function renderMeasurementsTable(detections) {
  elements.measurementsTbody.innerHTML = '';

  if (!detections || detections.length === 0) {
    elements.measurementsTbody.innerHTML = `
      <tr>
        <td colspan="9" style="text-align: center; color: var(--text-muted); padding: 2rem;">
          No objects detected with current confidence threshold.
        </td>
      </tr>
    `;
    return;
  }

  const currentScale = state.pixelsPerCm || 38.0;

  detections.forEach((det) => {
    const tr = document.createElement('tr');

    // Determine badge style
    let badgeClass = 'generic';
    const cName = det.class_name.toLowerCase();
    if (cName.includes('bottle')) badgeClass = 'bottle';
    else if (cName.includes('cup')) badgeClass = 'cup';
    else if (cName.includes('phone') || cName.includes('cell')) badgeClass = 'phone';
    else if (cName.includes('card') || cName.includes('book')) badgeClass = 'bottle';

    // 1. Pixel Dimensions
    const pxDimStr = `<span>${det.length_px.toFixed(1)} × ${det.width_px.toFixed(1)} px</span>`;

    // 2. Physical Dimensions & Area (computed using active scale)
    const lCm = (det.length_cm !== null && det.length_cm !== undefined) ? det.length_cm : (det.length_px / currentScale);
    const wCm = (det.width_cm !== null && det.width_cm !== undefined) ? det.width_cm : (det.width_px / currentScale);
    const aCm2 = (det.area_cm2 !== null && det.area_cm2 !== undefined) ? det.area_cm2 : (lCm * wCm);

    let physDimStr = '';
    let physAreaStr = '';

    if (state.unit === 'mm') {
      const lMm = lCm * 10;
      const wMm = wCm * 10;
      const aMm2 = aCm2 * 100;
      physDimStr = `<span class="dim-highlight">${lMm.toFixed(1)} × ${wMm.toFixed(1)} mm</span><br><span style="font-size:0.75rem; color:#64748b;">(${lCm.toFixed(2)} × ${wCm.toFixed(2)} cm)</span>`;
      physAreaStr = `<span class="dim-highlight">${Math.round(aMm2).toLocaleString()} mm²</span><br><span style="font-size:0.75rem; color:#64748b;">${Math.round(det.area_px2).toLocaleString()} px²</span>`;
    } else if (state.unit === 'in') {
      const lIn = lCm / 2.54;
      const wIn = wCm / 2.54;
      const aIn2 = aCm2 / 6.4516;
      physDimStr = `<span class="dim-highlight">${lIn.toFixed(2)} × ${wIn.toFixed(2)} in</span><br><span style="font-size:0.75rem; color:#64748b;">(${lCm.toFixed(2)} × ${wCm.toFixed(2)} cm)</span>`;
      physAreaStr = `<span class="dim-highlight">${aIn2.toFixed(2)} in²</span><br><span style="font-size:0.75rem; color:#64748b;">${Math.round(det.area_px2).toLocaleString()} px²</span>`;
    } else {
      // Default centimeters
      physDimStr = `<span class="dim-highlight">${lCm.toFixed(2)} × ${wCm.toFixed(2)} cm</span>`;
      physAreaStr = `<span class="dim-highlight">${aCm2.toFixed(2)} cm²</span><br><span style="font-size:0.75rem; color:#64748b;">${Math.round(det.area_px2).toLocaleString()} px²</span>`;
    }

    tr.innerHTML = `
      <td>#${det.object_id}</td>
      <td><span class="class-badge ${badgeClass}">${det.class_name}</span></td>
      <td>${Math.round(det.confidence * 100)}%</td>
      <td>${pxDimStr}</td>
      <td>${physDimStr}</td>
      <td>${physAreaStr}</td>
      <td>${det.angle_deg ? det.angle_deg.toFixed(1) + '°' : '0.0°'}</td>
      <td style="font-size:0.75rem; color:#64748b;">[${det.bbox.join(', ')}]</td>
      <td>
        <button class="btn-action-icon" title="Calibrate scale using this object" onclick="calibrateFromDetectedObject(${det.object_id})">
          <svg width="15" height="15" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4"></path></svg>
        </button>
      </td>
    `;
    elements.measurementsTbody.appendChild(tr);
  });
}

// -----------------------------------------------------------------------------
// Interactive 2-Point Line Calibration Tool
// -----------------------------------------------------------------------------

function toggleTwoPointMode() {
  state.twoPointMode = !state.twoPointMode;
  state.pickedPoints = [];
  elements.btnTwoPointCalib.classList.toggle('active', state.twoPointMode);

  if (state.twoPointMode) {
    elements.canvasBanner.style.display = 'flex';
    elements.canvasBannerText.textContent = 'Click Point 1 on the reference object (e.g. edge of card or ruler)...';
  } else {
    elements.canvasBanner.style.display = 'none';
    if (state.annotatedImageB64) switchView('annotated');
  }
}

function handleCanvasClick(e) {
  if (!state.twoPointMode) return;

  const rect = elements.canvas.getBoundingClientRect();
  const scaleX = elements.canvas.width / rect.width;
  const scaleY = elements.canvas.height / rect.height;

  const clientX = (e.clientX - rect.left) * scaleX;
  const clientY = (e.clientY - rect.top) * scaleY;

  state.pickedPoints.push({ x: clientX, y: clientY });

  // Draw point on canvas
  ctx.fillStyle = '#2563eb';
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(clientX, clientY, 6, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();

  if (state.pickedPoints.length === 1) {
    elements.canvasBannerText.textContent = 'Point 1 marked! Now click Point 2...';
  } else if (state.pickedPoints.length === 2) {
    const pt1 = state.pickedPoints[0];
    const pt2 = state.pickedPoints[1];

    // Draw connecting line
    ctx.strokeStyle = '#2563eb';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(pt1.x, pt1.y);
    ctx.lineTo(pt2.x, pt2.y);
    ctx.stroke();

    const dist = Math.hypot(pt2.x - pt1.x, pt2.y - pt1.y);
    openCalibrationModal(dist);
    toggleTwoPointMode();
  }
}

function openCalibrationModal(pixelDistance) {
  elements.modalPixelDist.value = `${pixelDistance.toFixed(1)} px`;
  elements.modalPixelDist.dataset.dist = pixelDistance;
  elements.calibModal.style.display = 'flex';
}

function closeModal() {
  elements.calibModal.style.display = 'none';
}

async function confirmCalibrationModal() {
  const pixelDist = parseFloat(elements.modalPixelDist.dataset.dist);
  const knownLen = parseFloat(elements.modalKnownLength.value);
  const unit = elements.modalUnitSelect.value;

  if (isNaN(knownLen) || knownLen <= 0) {
    alert('Please enter a valid positive length.');
    return;
  }

  let realCm = knownLen;
  if (unit === 'mm') realCm = knownLen / 10.0;
  if (unit === 'in') realCm = knownLen * 2.54;

  const calculatedScale = pixelDist / realCm;
  setScale(calculatedScale);
  closeModal();

  // Re-detect with new scale factor
  if (state.currentImageB64) {
    runDetection();
  }
}

// -----------------------------------------------------------------------------
// Calibration Presets & Object Calibration
// -----------------------------------------------------------------------------

function setScale(scaleVal) {
  if (!scaleVal || isNaN(scaleVal) || scaleVal <= 0) return;
  state.pixelsPerCm = parseFloat(scaleVal);
  if (elements.statusScale) elements.statusScale.textContent = `${state.pixelsPerCm.toFixed(2)} px/cm`;
  if (elements.calibScaleDisplay) elements.calibScaleDisplay.textContent = `${state.pixelsPerCm.toFixed(2)} px/cm`;
  if (elements.manualScaleInput) elements.manualScaleInput.value = state.pixelsPerCm.toFixed(2);

  // Re-calculate all measurements in place if detections exist
  if (state.detections && state.detections.length > 0) {
    state.detections.forEach((det) => {
      det.pixels_per_cm = state.pixelsPerCm;
      det.is_calibrated = true;
      det.length_cm = det.length_px / state.pixelsPerCm;
      det.width_cm = det.width_px / state.pixelsPerCm;
      det.length_mm = det.length_cm * 10.0;
      det.width_mm = det.width_cm * 10.0;
      det.length_in = det.length_cm / 2.54;
      det.width_in = det.width_cm / 2.54;
      det.area_cm2 = det.length_cm * det.width_cm;
    });
    renderMeasurementsTable(state.detections);
  }
}

function applyPresetCalibration(knownLength, unit, label) {
  let realCm = knownLength;
  if (unit === 'mm') realCm = knownLength / 10.0;
  if (unit === 'in') realCm = knownLength * 2.54;

  if (state.detections && state.detections.length > 0) {
    const l = (label || '').toLowerCase();
    // Search for a detected object matching this reference
    let ref = state.detections.find(d => {
      const c = d.class_name.toLowerCase();
      if (l.includes('card') && (c.includes('card') || c.includes('book'))) return true;
      if (l.includes('phone') && (c.includes('phone') || c.includes('cell'))) return true;
      if (l.includes('coin') && (c.includes('coin') || c.includes('ball'))) return true;
      if (l.includes('bottle') && c.includes('bottle')) return true;
      if (l.includes('cup') && c.includes('cup')) return true;
      return false;
    });

    if (!ref) {
      // If no direct class match, pick the reference object
      ref = state.detections[0];
    }

    const scale = ref.length_px / realCm;
    setScale(scale);
    if (state.currentImageB64) {
      runDetection();
    }
  } else {
    // If no detections yet, set the realistic reference scale
    let baselineScale = 38.0;
    if (label.includes('Card')) baselineScale = 40.0;
    else if (label.includes('Coin')) baselineScale = 45.0;
    else if (label.includes('A4')) baselineScale = 30.0;
    else if (label.includes('Phone')) baselineScale = 38.0;

    setScale(baselineScale);
  }
}

window.calibrateFromDetectedObject = function(objId) {
  const target = state.detections.find(d => d.object_id === objId);
  if (!target) return;
  currentCalibObject = target;

  if (elements.objModalPixelLength) {
    elements.objModalPixelLength.value = `${target.length_px.toFixed(1)} px (${target.width_px.toFixed(1)} px width)`;
  }
  if (elements.objModalSubtitle) {
    elements.objModalSubtitle.textContent = `Calibrating scale using Object #${target.object_id} [${target.class_name}]:`;
  }

  // Suggest sensible known dimensions based on recognized COCO class
  const c = target.class_name.toLowerCase();
  if (elements.objModalKnownLength) {
    if (c.includes('card')) elements.objModalKnownLength.value = '8.56';
    else if (c.includes('phone') || c.includes('cell')) elements.objModalKnownLength.value = '14.7';
    else if (c.includes('coin')) elements.objModalKnownLength.value = '2.43';
    else if (c.includes('bottle')) elements.objModalKnownLength.value = '24.0';
    else if (c.includes('cup')) elements.objModalKnownLength.value = '10.0';
    else elements.objModalKnownLength.value = '10.0';
  }
  if (elements.objModalUnitSelect) {
    elements.objModalUnitSelect.value = state.unit || 'cm';
  }

  if (elements.objectCalibModal) {
    elements.objectCalibModal.style.display = 'flex';
  }
};

function closeObjectCalibModal() {
  if (elements.objectCalibModal) {
    elements.objectCalibModal.style.display = 'none';
  }
  currentCalibObject = null;
}

function confirmObjectCalibModal() {
  if (!currentCalibObject) return;
  const knownLen = parseFloat(elements.objModalKnownLength.value);
  const unit = elements.objModalUnitSelect.value;
  if (isNaN(knownLen) || knownLen <= 0) {
    alert('Please enter a positive numeric real-world length.');
    return;
  }

  let realCm = knownLen;
  if (unit === 'mm') realCm = knownLen / 10.0;
  if (unit === 'in') realCm = knownLen * 2.54;

  const newScale = currentCalibObject.length_px / realCm;
  setScale(newScale);
  closeObjectCalibModal();

  if (state.currentImageB64) {
    runDetection();
  }
}

// -----------------------------------------------------------------------------
// Exports
// -----------------------------------------------------------------------------

async function exportCsv() {
  if (state.detections.length === 0) {
    alert('No detection data to export.');
    return;
  }

  const res = await fetch('/api/export/csv', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ detections: state.detections })
  });

  const blob = await res.blob();
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `visionmeasure_measurements_${Date.now()}.csv`;
  a.click();
  window.URL.revokeObjectURL(url);
}

async function exportJson() {
  if (state.detections.length === 0) {
    alert('No detection data to export.');
    return;
  }

  const res = await fetch('/api/export/json', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      timestamp: new Date().toISOString(),
      pixels_per_cm: state.pixelsPerCm,
      unit: state.unit,
      detections: state.detections
    })
  });

  const blob = await res.blob();
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `visionmeasure_measurements_${Date.now()}.json`;
  a.click();
  window.URL.revokeObjectURL(url);
}

function downloadAnnotatedImage() {
  if (!state.annotatedImageB64) {
    alert('No annotated image available to download.');
    return;
  }
  const a = document.createElement('a');
  a.href = state.annotatedImageB64;
  a.download = `visionmeasure_annotated_${Date.now()}.jpg`;
  a.click();
}

// -----------------------------------------------------------------------------
// UI Utilities
// -----------------------------------------------------------------------------

function showLoading(show) {
  elements.loadingOverlay.style.display = show ? 'flex' : 'none';
}
