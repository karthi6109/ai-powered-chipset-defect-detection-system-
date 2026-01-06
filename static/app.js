async function initCamera() {
  const video = document.getElementById('video');
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    video.srcObject = stream;
  } catch (e) {
    console.error('Camera init error', e);
  }
}

function dataURLtoBlob(dataurl) {
  const arr = dataurl.split(',');
  const mime = arr[0].match(/:(.*?);/)[1];
  const bstr = atob(arr[1]);
  let n = bstr.length;
  const u8arr = new Uint8Array(n);
  while (n--) {
    u8arr[n] = bstr.charCodeAt(n);
  }
  return new Blob([u8arr], { type: mime });
}

async function captureAndAnalyze() {
  const video = document.getElementById('video');
  const canvas = document.createElement('canvas');
  canvas.width = 224; canvas.height = 224;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  const dataUrl = canvas.toDataURL('image/png');
  const blob = dataURLtoBlob(dataUrl);
  const form = new FormData();
  form.append('image', blob, 'capture.png');

  const res = await fetch('/analyze', { method: 'POST', body: form });
  const j = await res.json();
  showResult(j);
}

async function uploadAndAnalyze() {
  const input = document.getElementById('fileInput');
  if (!input.files || input.files.length === 0) return alert('Select a file');
  const form = new FormData();
  form.append('image', input.files[0]);
  const res = await fetch('/analyze', { method: 'POST', body: form });
  const j = await res.json();
  showResult(j);
}

function showResult(j) {
  const resultText = document.getElementById('resultText');
  resultText.innerHTML = `<strong>${j.result}</strong> (probs: ${j.probs.map(p=>p.toFixed(3)).join(', ')})`;
  const heat = document.getElementById('heatmapContainer');
  heat.innerHTML = '';
  const img = document.createElement('img');
  img.src = 'data:image/png;base64,' + j.heatmap_b64;
  img.style.maxWidth = '480px';
  heat.appendChild(img);

  const report = document.getElementById('reportContainer');
  report.innerHTML = j.report_html + '\n<button id="downloadReport">Download Report</button>';
  document.getElementById('downloadReport').addEventListener('click', async () => {
    const r = await fetch('/report/download', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({ report_html: j.report_html }) });
    const blob = await r.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = 'report.html'; a.click();
    URL.revokeObjectURL(url);
  });
}

window.addEventListener('load', ()=>{
  initCamera();
  document.getElementById('captureBtn').addEventListener('click', captureAndAnalyze);
  document.getElementById('uploadBtn').addEventListener('click', uploadAndAnalyze);
});