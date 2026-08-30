
/* ===================== GEOLOCATION / SCENE SELECTOR (Leaflet) ===================== */
let sceneMap = null, sceneMarker = null, sceneTileOk = false;
let selectedScene = { lat: 23.0225, lon: 72.5714, source: 'DEFAULT', status: 'pending', seed: 230225 };
let geoMode = 'SCENE';
let previewMode = 'RGB';
let currentAnalysisMode = 'single'; // 'single' or 'bitemporal'
let coordFormat = localStorage.getItem('satquery_coord_format') || 'DD';
let footprintLayer = null, analysisGridLayer = null, simAnalysisLayer = null, overlayControl = null;
let baseOsmLayer = null, baseSatLayer = null;
let aoiPoints = [], aoiPolygon = null, aoiMarkers = [], aoiDrawing = false, aoiLocked = false, aoiAreaKm2 = 0;
let compareA = null, compareB = null, compareMarkerA = null, compareMarkerB = null, compareLine = null, comparePick = 'A';
let terrainOptsFromSeed = function (seed) { return { waterX: 0.6 + ((seed % 10) / 40), waterY: 0.55, builtCount: 3 + (seed % 5) }; };

const IN_CITIES = {
  ahmedabad: [23.0225, 72.5714], delhi: [28.6139, 77.2090], 'new delhi': [28.6139, 77.2090],
  mumbai: [19.0760, 72.8777], bangalore: [12.9716, 77.5946], bengaluru: [12.9716, 77.5946],
  chennai: [13.0827, 80.2707], kolkata: [22.5726, 88.3639], hyderabad: [17.3850, 78.4867],
  pune: [18.5204, 73.8567], jaipur: [26.9124, 75.7873], lucknow: [26.8467, 80.9462],
  kanpur: [26.4499, 80.3319], nagpur: [21.1458, 79.0882], indore: [22.7196, 75.8577],
  bhopal: [23.2599, 77.4126], visakhapatnam: [17.6868, 83.2185], patna: [25.5941, 85.1376],
  vadodara: [22.3072, 73.1812], surat: [21.1702, 72.8311], rajkot: [22.3039, 70.8022],
  gandhinagar: [23.2156, 72.6369], chandigarh: [30.7333, 76.7794], kochi: [9.9312, 76.2673],
  thiruvananthapuram: [8.5241, 76.9366], coimbatore: [11.0168, 76.9558], madurai: [9.9252, 78.1198],
  varanasi: [25.3176, 83.0059], agra: [27.1767, 78.0081], amritsar: [31.6340, 74.8723],
  srinagar: [34.0837, 74.7973], guwahati: [26.1445, 91.7362], bhubaneswar: [20.2961, 85.8245],
  ranchi: [23.3441, 85.3096], dehradun: [30.3165, 78.0322], shimla: [31.1048, 77.1734],
  goa: [15.2993, 74.1240], panaji: [15.4909, 73.8278], noida: [28.5355, 77.3910],
  gurgaon: [28.4595, 77.0266], gurugram: [28.4595, 77.0266], faridabad: [28.4089, 77.3178],
  ghaziabad: [28.6692, 77.4538], mysore: [12.2958, 76.6394], mysuru: [12.2958, 76.6394],
  jodhpur: [26.2389, 73.0243], udaipur: [24.5854, 73.7125], allahabad: [25.4358, 81.8463],
  prayagraj: [25.4358, 81.8463]
};

function coordToSeed(lat, lon) {
  // deterministic integer seed from coordinates so the same location always renders the same "scene"
  return Math.abs(Math.round((lat * 10000) + (lon * 7919) + 100000)) % 2147483647;
}
function getSceneId(lat, lon) {
  const seed = coordToSeed(lat ?? selectedScene.lat, lon ?? selectedScene.lon);
  return 'S2-L2A-DEMO-' + String(seed).padStart(8, '0').slice(0, 8);
}
function demoCloudCover(seed) { return 3 + (seed % 37); }
function demoLandCover(seed) {
  const built = 12 + (seed % 28), water = 5 + (seed % 15), veg = Math.max(8, 92 - built - water);
  return { built, veg, water };
}
function toDMS(v, type) {
  const dir = type === 'lat' ? (v >= 0 ? 'N' : 'S') : (v >= 0 ? 'E' : 'W');
  const abs = Math.abs(v);
  const d = Math.floor(abs);
  const mFloat = (abs - d) * 60;
  const m = Math.floor(mFloat);
  const s = ((mFloat - m) * 60).toFixed(1);
  return d + 'Â° ' + m + 'â€² ' + s + 'â€³ ' + dir;
}
function haversineKm(lat1, lon1, lat2, lon2) {
  const R = 6371, toR = Math.PI / 180;
  const dLat = (lat2 - lat1) * toR, dLon = (lon2 - lon1) * toR;
  const a = Math.sin(dLat / 2) ** 2 + Math.cos(lat1 * toR) * Math.cos(lat2 * toR) * Math.sin(dLon / 2) ** 2;
  return 2 * R * Math.asin(Math.min(1, Math.sqrt(a)));
}
function polygonAreaKm2(latlngs) {
  if (!latlngs || latlngs.length < 3) return 0;
  const R = 6371, toR = Math.PI / 180;
  let sum = 0;
  for (let i = 0; i < latlngs.length; i++) {
    const j = (i + 1) % latlngs.length;
    const lat1 = latlngs[i].lat * toR, lon1 = latlngs[i].lng * toR;
    const lat2 = latlngs[j].lat * toR, lon2 = latlngs[j].lng * toR;
    sum += (lon2 - lon1) * (2 + Math.sin(lat1) + Math.sin(lat2));
  }
  return Math.abs(sum) * R * R / 2;
}
function showSearchMsg(t) {
  const el = document.getElementById('searchMsg');
  if (!el) return;
  if (!t) { el.classList.remove('show'); el.textContent = ''; return; }
  el.textContent = t; el.classList.add('show');
}
function ensureMapSize() {
  if (sceneMap) setTimeout(() => sceneMap.invalidateSize(), 80);
}

function initSceneMap() {
  if (sceneMap) return;
  if (typeof L === 'undefined') {
    // Leaflet failed to load (e.g. no network) â€” show graceful fallback, don't break the rest of the app
    const err = document.getElementById('mapLoadError');
    if (err) err.classList.remove('hidden');
    return;
  }
  const mapEl = document.getElementById('sceneMap');
  if (!mapEl) return;
  sceneMap = L.map('sceneMap', { zoomControl: true, attributionControl: true }).setView([selectedScene.lat, selectedScene.lon], 7);
  window.satQueryMap = sceneMap;

  baseOsmLayer = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 18, attribution: 'Â© OpenStreetMap contributors'
  }).addTo(sceneMap);

  baseSatLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
    maxZoom: 18, attribution: 'Tiles Â© Esri â€” Source: Esri, Maxar, Earthstar Geographics'
  });
  window.baseOsmLayer = baseOsmLayer;
  window.baseSatLayer = baseSatLayer;

  const AnalysisGrid = L.GridLayer.extend({
    createTile: function () {
      const tile = L.DomUtil.create('canvas', 'leaflet-tile');
      const size = this.getTileSize();
      tile.width = size.x; tile.height = size.y;
      const ctx = tile.getContext('2d');
      ctx.strokeStyle = 'rgba(63,216,255,0.22)';
      ctx.lineWidth = 1;
      const step = 32;
      for (let x = 0; x <= size.x; x += step) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, size.y); ctx.stroke(); }
      for (let y = 0; y <= size.y; y += step) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(size.x, y); ctx.stroke(); }
      return tile;
    }
  });
  analysisGridLayer = new AnalysisGrid({ opacity: 0.55, className: 'analysis-grid-layer' });
  footprintLayer = L.rectangle(footprintBounds(selectedScene.lat, selectedScene.lon), {
    color: '#3fd8ff', weight: 1.5, fillColor: '#3fd8ff', fillOpacity: 0.12, dashArray: '4 3'
  });
  simAnalysisLayer = L.layerGroup();

  overlayControl = L.control.layers(
    { 'Street (OSM)': baseOsmLayer, 'Satellite Imagery': baseSatLayer },
    { 'Scene Footprint': footprintLayer, 'Analysis Grid': analysisGridLayer, 'Simulated Analysis': simAnalysisLayer },
    { collapsed: true }
  ).addTo(sceneMap);
  footprintLayer.addTo(sceneMap);
  L.control.scale({ imperial: false, metric: true, position: 'bottomleft' }).addTo(sceneMap);

  baseOsmLayer.on('tileload', () => { sceneTileOk = true; const err = document.getElementById('mapLoadError'); if (err) err.classList.add('hidden'); });

  sceneMarker = L.marker([selectedScene.lat, selectedScene.lon]).addTo(sceneMap)
    .bindPopup('Default scene â€” Ahmedabad, Gujarat');

  sceneMap.on('click', function (e) {
    const lat = e.latlng.lat, lon = e.latlng.lng;
    if (geoMode === 'AOI' && aoiDrawing) { addAoiVertex(e.latlng); return; }
    if (geoMode === 'COMPARE') { placeComparePoint(lat, lon); return; }
    placeMarker(lat, lon);
  });
  sceneMap.on('zoomend moveend', updateMapHud);

  // fix sizing since container may have been hidden/animated at init
  setTimeout(() => sceneMap.invalidateSize(), 200);
  updateMapHud();
  // if no tiles have loaded within a few seconds, assume tile access is blocked and show the fallback note
  setTimeout(() => { if (!sceneTileOk) { const err = document.getElementById('mapLoadError'); if (err) err.classList.remove('hidden'); } }, 4000);
}

function footprintBounds(lat, lon) {
  const d = 0.045;
  return [[lat - d, lon - d], [lat + d, lon + d]];
}
function updateFootprint() {
  if (!footprintLayer) return;
  footprintLayer.setBounds(footprintBounds(selectedScene.lat, selectedScene.lon));
}
function clearFootprint() {
  if (footprintLayer && sceneMap && sceneMap.hasLayer(footprintLayer)) sceneMap.removeLayer(footprintLayer);
  if (footprintLayer) footprintLayer.setBounds(footprintBounds(selectedScene.lat, selectedScene.lon));
}
function clearAnalysisOverlay() {
  if (simAnalysisLayer) simAnalysisLayer.clearLayers();
  if (simAnalysisLayer && sceneMap && sceneMap.hasLayer(simAnalysisLayer)) sceneMap.removeLayer(simAnalysisLayer);
}
function buildSimulatedAnalysisOverlay() {
  if (!sceneMap || !simAnalysisLayer) return;
  simAnalysisLayer.clearLayers();
  const lat = selectedScene.lat, lon = selectedScene.lon;
  const cover = demoLandCover(selectedScene.seed);
  const label = L.divIcon({ className: '', html: '<div class="sim-analysis-label">SIMULATED ANALYSIS</div>', iconSize: [140, 18], iconAnchor: [70, 9] });
  simAnalysisLayer.addLayer(L.marker([lat, lon], { icon: label, interactive: false }));
  simAnalysisLayer.addLayer(L.circle([lat + 0.018, lon - 0.02], { radius: 1800, color: '#ff6a6a', fillColor: '#ff6a6a', fillOpacity: 0.22, weight: 1 }).bindTooltip('Built-up ' + cover.built + '% Â· DEMO ESTIMATE'));
  simAnalysisLayer.addLayer(L.circle([lat - 0.012, lon + 0.016], { radius: 2200, color: '#3ee089', fillColor: '#3ee089', fillOpacity: 0.2, weight: 1 }).bindTooltip('Vegetation ' + cover.veg + '% Â· DEMO ESTIMATE'));
  simAnalysisLayer.addLayer(L.circle([lat + 0.006, lon + 0.028], { radius: 1400, color: '#4f7cff', fillColor: '#4f7cff', fillOpacity: 0.22, weight: 1 }).bindTooltip('Water ' + cover.water + '% Â· DEMO ESTIMATE'));
  if (!sceneMap.hasLayer(simAnalysisLayer)) simAnalysisLayer.addTo(sceneMap);
}

function placeMarker(lat, lon) {
  selectedScene.lat = lat; selectedScene.lon = lon; selectedScene.status = 'pending';
  selectedScene.seed = coordToSeed(lat, lon);
  if (sceneMarker) { sceneMarker.setLatLng([lat, lon]); }
  else if (sceneMap) { sceneMarker = L.marker([lat, lon]).addTo(sceneMap); }
  document.getElementById('readLat').textContent = formatCoord(lat, 'lat');
  document.getElementById('readLon').textContent = formatCoord(lon, 'lon');
  document.getElementById('mapTag').textContent = 'location picked â€” click USE LOCATION';
  const st = document.getElementById('sceneStatus');
  st.classList.remove('ready');
  document.getElementById('sceneStatusText').textContent = 'SCENE NOT CONFIRMED';
  updateFootprint();
  if (footprintLayer && sceneMap && !sceneMap.hasLayer(footprintLayer)) footprintLayer.addTo(sceneMap);
  captureScenePreview(selectedScene.seed);
  updateMapHud();
  updateLandingHud();
  window.selectedScene = selectedScene;
}

function captureScenePreview(seed) {
  const el = document.getElementById('scenePreview');
  if (!el) return;
  el.innerHTML = '';
  el.style.display = 'block';
  const opts = terrainOptsFromSeed(seed);
  canvasTo(el, 260, 90, ctx => {
    if (previewMode === 'SAR') drawSAR(ctx, 260, 90, seed);
    else if (previewMode === 'NIR') drawNIR(ctx, 260, 90, seed, opts);
    else drawTerrain(ctx, 260, 90, seed, opts);
  });
}
function setPreviewMode(mode) {
  previewMode = mode;
  ['RGB', 'NIR', 'SAR'].forEach(m => {
    const b = document.getElementById('pv' + m);
    if (b) b.classList.toggle('active', m === mode);
  });
  captureScenePreview(selectedScene.seed);
}

function formatCoord(v, type) {
  if (coordFormat === 'DMS') return toDMS(v, type);
  const dir = type === 'lat' ? (v >= 0 ? 'N' : 'S') : (v >= 0 ? 'E' : 'W');
  return Math.abs(v).toFixed(4) + 'Â° ' + dir;
}
function saveCoordFormat() {
  const sel = document.getElementById('coordFormatSelect');
  if (sel) coordFormat = sel.value;
  localStorage.setItem('satquery_coord_format', coordFormat);
  document.getElementById('readLat').textContent = formatCoord(selectedScene.lat, 'lat');
  document.getElementById('readLon').textContent = formatCoord(selectedScene.lon, 'lon');
  updateMapHud();
  updateLandingHud();
}
function updateMapHud() {
  const sid = document.getElementById('hudSceneId');
  const hc = document.getElementById('hudCoords');
  const hz = document.getElementById('hudZoom');
  const ha = document.getElementById('hudAcq');
  const hcl = document.getElementById('hudCloud');
  const seed = selectedScene.seed;
  const cloud = demoCloudCover(seed);
  if (sid) sid.textContent = 'SCENE ' + getSceneId();
  if (hc) hc.textContent = formatCoord(selectedScene.lat, 'lat') + ' Â· ' + formatCoord(selectedScene.lon, 'lon');
  if (hz) hz.textContent = 'Z ' + (sceneMap ? sceneMap.getZoom() : 7);
  if (ha) ha.textContent = 'ACQ DEMO/SIMULATED';
  if (hcl) hcl.textContent = 'CLOUD ' + cloud + '% DEMO';
  const mini = document.getElementById('sceneMetaMini');
  if (mini) {
    mini.innerHTML = '<b>SENSOR</b> Sentinel-2<br><b>PRODUCT</b> L2A / DEMO<br><b>GSD</b> 10m<br><b>ACQUISITION</b> DEMO/SIMULATED<br><b>CLOUD</b> ' + cloud + '% DEMO<br><b>SCENE ID</b> ' + getSceneId();
  }
}
function updateLandingHud() {
  const el = document.getElementById('heroLiveCoords');
  const sid = document.getElementById('heroSceneId');
  if (el) el.textContent = formatCoord(selectedScene.lat, 'lat') + ' Â· ' + formatCoord(selectedScene.lon, 'lon');
  if (sid) sid.textContent = 'SCENE_ID: ' + getSceneId();
}
function hudZoom(dir) {
  if (!sceneMap) return;
  if (dir > 0) sceneMap.zoomIn(); else sceneMap.zoomOut();
}
function centerScene() {
  if (!sceneMap) return;
  if (geoMode === 'AOI' && aoiPolygon) { sceneMap.fitBounds(aoiPolygon.getBounds(), { padding: [28, 28] }); }
  else if (geoMode === 'COMPARE' && compareA && compareB) { sceneMap.fitBounds(L.latLngBounds([compareA, compareB]), { padding: [40, 40] }); }
  else sceneMap.setView([selectedScene.lat, selectedScene.lon], Math.max(sceneMap.getZoom(), 11));
  ensureMapSize();
}
function requestBrowserLocation() {
  if (!navigator.geolocation) { showSearchMsg('Geolocation is not supported in this browser.'); return; }
  showSearchMsg('Requesting current locationâ€¦');
  navigator.geolocation.getCurrentPosition(function (pos) {
    showSearchMsg('');
    const lat = pos.coords.latitude, lon = pos.coords.longitude;
    placeMarker(lat, lon);
    if (sceneMap) sceneMap.setView([lat, lon], 13);
  }, function (err) {
    if (err && err.code === 1) showSearchMsg('Location permission denied. Pick a point on the map or search.');
    else if (err && err.code === 2) showSearchMsg('Position unavailable. Try search or map click.');
    else if (err && err.code === 3) showSearchMsg('Location request timed out.');
    else showSearchMsg('Could not read current location.');
  }, { enableHighAccuracy: true, timeout: 8000, maximumAge: 0 });
}
async function searchLocation() {
  const input = document.getElementById('locSearch');
  const q = (input && input.value || '').trim();
  if (!q) { showSearchMsg('Enter a city name or lat, lon.'); return; }
  const pair = q.match(/^\s*(-?\d+\.?\d*)\s*[, ]\s*(-?\d+\.?\d*)\s*$/);
  if (pair) {
    const lat = parseFloat(pair[1]), lon = parseFloat(pair[2]);
    if (lat >= -90 && lat <= 90 && lon >= -180 && lon <= 180) {
      showSearchMsg('');
      placeMarker(lat, lon);
      if (sceneMap) sceneMap.setView([lat, lon], 12);
      return;
    }
    showSearchMsg('Latitude must be âˆ’90â€¦90 and longitude âˆ’180â€¦180.');
    return;
  }
  const key = q.toLowerCase();
  if (IN_CITIES[key]) {
    const [lat, lon] = IN_CITIES[key];
    showSearchMsg('Offline city match: ' + q);
    placeMarker(lat, lon);
    if (sceneMap) sceneMap.setView([lat, lon], 12);
    setTimeout(() => showSearchMsg(''), 1800);
    return;
  }
  try {
    const resp = await fetch('https://nominatim.openstreetmap.org/search?format=json&limit=1&q=' + encodeURIComponent(q), { headers: { 'Accept': 'application/json' } });
    if (!resp.ok) throw new Error('nominatim');
    const data = await resp.json();
    if (data && data[0]) {
      const lat = parseFloat(data[0].lat), lon = parseFloat(data[0].lon);
      showSearchMsg('');
      placeMarker(lat, lon);
      if (sceneMap) sceneMap.setView([lat, lon], 12);
    } else showSearchMsg('No match for "' + q + '". Try lat, lon or a major city.');
  } catch (e) {
    showSearchMsg('Search needs internet for unknown places. Try an Indian city name or lat, lon (offline).');
  }
}
function setGeoMode(mode) {
  geoMode = mode;
  ['SCENE', 'AOI', 'COMPARE'].forEach(m => {
    const b = document.getElementById('mode' + m);
    if (b) b.classList.toggle('active', m === mode);
  });
  const aoiBar = document.getElementById('aoiBar');
  const cmpBar = document.getElementById('cmpBar');
  if (aoiBar) aoiBar.classList.toggle('show', mode === 'AOI');
  if (cmpBar) cmpBar.classList.toggle('show', mode === 'COMPARE');
  refreshQueryChips();
  ensureMapSize();
}
function refreshQueryChips() {
  const el = document.getElementById('queryChips');
  if (!el) return;
  let chips;
  if (geoMode === 'AOI') {
    chips = [
      ['grounding', 'Highlight water inside this AOI'],
      ['vqa', 'What land cover is in this AOI?'],
      ['change', 'What changed inside this AOI?'],
      ['fusion', 'Fuse optical + SAR for this AOI']
    ];
  } else if (geoMode === 'COMPARE') {
    chips = [
      ['change', 'What changed between Scene A and Scene B?'],
      ['vqa', 'Compare land use between the two scenes'],
      ['fusion', 'Fuse optical + SAR across both dates']
    ];
  } else {
    chips = [
      ['vqa', "What's visible in this image?"],
      ['grounding', 'Highlight the water body'],
      ['change', 'What changed between these images?'],
      ['fusion', 'Fuse optical + SAR for built-up areas']
    ];
  }
  el.innerHTML = chips.map(([k, t]) => '<button class="chip" onclick="pickDemo(\'' + k + '\')">' + t + '</button>').join('');
}
function startAoiDraw() {
  aoiDrawing = true; aoiLocked = false;
  clearAoiGeometry(false);
  const h = document.getElementById('aoiHint');
  if (h) h.textContent = 'Click map to add vertices Â· FINISH when done';
}
function addAoiVertex(ll) {
  aoiPoints.push(ll);
  if (sceneMap) {
    const m = L.circleMarker(ll, { radius: 5, color: '#3fd8ff', fillColor: '#3fd8ff', fillOpacity: 0.9, weight: 1 }).addTo(sceneMap);
    aoiMarkers.push(m);
    if (aoiPolygon) { aoiPolygon.setLatLngs(aoiPoints); }
    else if (aoiPoints.length >= 2) {
      aoiPolygon = L.polygon(aoiPoints, { color: '#3fd8ff', weight: 1.5, fillColor: '#3fd8ff', fillOpacity: 0.12 }).addTo(sceneMap);
    }
  }
  const h = document.getElementById('aoiHint');
  if (h) h.textContent = aoiPoints.length + ' vertices â€” FINISH to close polygon';
}
function finishAoi() {
  aoiDrawing = false;
  if (aoiPoints.length < 3) { showSearchMsg('AOI needs at least 3 points.'); return; }
  aoiAreaKm2 = polygonAreaKm2(aoiPoints);
  const h = document.getElementById('aoiHint');
  if (h) h.textContent = 'AOI closed Â· ' + aoiAreaKm2.toFixed(2) + ' kmÂ² (approx) Â· USE AOI to lock';
  showAoiDemoStats();
}
function showAoiDemoStats() {
  const box = document.getElementById('aoiStatsBox');
  const txt = document.getElementById('aoiStatsText');
  if (!box || !txt) return;
  const c = centroidAoi();
  const seed = coordToSeed(c.lat, c.lng);
  const cover = demoLandCover(seed);
  txt.innerHTML = 'Area ~ ' + aoiAreaKm2.toFixed(2) + ' kmÂ²<br>Built-up ' + cover.built + '%<br>Vegetation ' + cover.veg + '%<br>Water ' + cover.water + '%';
  box.classList.remove('hidden');
}
function centroidAoi() {
  if (!aoiPoints.length) return { lat: selectedScene.lat, lng: selectedScene.lon };
  let lat = 0, lng = 0;
  aoiPoints.forEach(p => { lat += p.lat; lng += p.lng; });
  return { lat: lat / aoiPoints.length, lng: lng / aoiPoints.length };
}
function clearAoiGeometry(hideStats) {
  aoiPoints = []; aoiAreaKm2 = 0; aoiLocked = false;
  aoiMarkers.forEach(m => { if (sceneMap) sceneMap.removeLayer(m); });
  aoiMarkers = [];
  if (aoiPolygon && sceneMap) sceneMap.removeLayer(aoiPolygon);
  aoiPolygon = null;
  if (hideStats !== false) {
    const box = document.getElementById('aoiStatsBox');
    if (box) box.classList.add('hidden');
  }
}
function clearAoi() {
  aoiDrawing = false;
  clearAoiGeometry(true);
  const h = document.getElementById('aoiHint');
  if (h) h.textContent = 'DRAW AOI â€” click the map to add vertices';
}
function useAoi() {
  if (aoiPoints.length < 3) { showSearchMsg('Draw and finish an AOI first.'); return; }
  if (aoiDrawing) finishAoi();
  aoiLocked = true; aoiDrawing = false;
  const c = centroidAoi();
  placeMarker(c.lat, c.lng);
  useLocation();
  mapSceneActive = true;
  selectedScene.source = 'AOI';
  document.getElementById('mapTag').textContent = 'AOI locked (DEMO)';
  document.getElementById('sceneStatusText').textContent = 'AOI LOCKED Â· ' + aoiAreaKm2.toFixed(2) + ' kmÂ² Â· DEMO ESTIMATE';
  showAoiDemoStats();
}
function setComparePick(which) {
  comparePick = which;
  const h = document.getElementById('cmpHint');
  if (h) h.textContent = 'Click map to set Scene ' + which;
}
function placeComparePoint(lat, lon) {
  const ll = [lat, lon];
  if (comparePick === 'A') {
    compareA = ll;
    if (compareMarkerA) compareMarkerA.setLatLng(ll);
    else if (sceneMap) compareMarkerA = L.marker(ll).addTo(sceneMap).bindPopup('Scene A');
    if (compareMarkerA) compareMarkerA.bindPopup('Scene A').openPopup();
    comparePick = 'B';
  } else {
    compareB = ll;
    if (compareMarkerB) compareMarkerB.setLatLng(ll);
    else if (sceneMap) compareMarkerB = L.marker(ll).addTo(sceneMap).bindPopup('Scene B');
    if (compareMarkerB) compareMarkerB.bindPopup('Scene B').openPopup();
  }
  updateCompareLine();
}
function updateCompareLine() {
  if (compareLine && sceneMap) { sceneMap.removeLayer(compareLine); compareLine = null; }
  const h = document.getElementById('cmpHint');
  if (compareA && compareB && sceneMap) {
    compareLine = L.polyline([compareA, compareB], { color: '#3fd8ff', weight: 2, dashArray: '6 4' }).addTo(sceneMap);
    const d = haversineKm(compareA[0], compareA[1], compareB[0], compareB[1]);
    if (h) h.textContent = 'Distance ' + d.toFixed(2) + ' km Â· RUN COMPARISON for bi-temporal demo';
    placeMarker(compareA[0], compareA[1]);
  } else if (h) {
    h.textContent = compareA ? 'Scene A set â€” click Scene B' : 'Click map: Scene A, then Scene B';
  }
}
function clearCompare() {
  if (compareMarkerA && sceneMap) sceneMap.removeLayer(compareMarkerA);
  if (compareMarkerB && sceneMap) sceneMap.removeLayer(compareMarkerB);
  if (compareLine && sceneMap) sceneMap.removeLayer(compareLine);
  compareA = compareB = compareMarkerA = compareMarkerB = compareLine = null;
  comparePick = 'A';
  const h = document.getElementById('cmpHint');
  if (h) h.textContent = 'Click map: Scene A, then Scene B';
}
function runComparison() {
  if (!compareA || !compareB) { showSearchMsg('Set Scene A and Scene B first.'); return; }
  placeMarker(compareA[0], compareA[1]);
  selectedDemo = 'change';
  pickDemo('change');
  const seedA = coordToSeed(compareA[0], compareA[1]);
  const seedB = coordToSeed(compareB[0], compareB[1]);
  fillUploadCard('img1', seedA, terrainOptsFromSeed(seedA), 'SCENE A Â· DEMO');
  fillUploadCard('img2', seedB, terrainOptsFromSeed(seedB), 'SCENE B Â· DEMO');
  document.getElementById('queryInput').value = 'What changed between Scene A and Scene B?';
  runAnalysis();
}
function analyzeThisScene() {
  const qi = document.getElementById('queryInput');
  if (qi && !qi.value.trim()) {
    if (geoMode === 'AOI') qi.value = 'What land cover is visible inside this AOI?';
    else if (geoMode === 'COMPARE') qi.value = 'What changed between Scene A and Scene B?';
    else qi.value = 'What is visible in this scene?';
  }
  if (selectedScene.status !== 'ready') useLocation();
  const grid = document.querySelector('.ws-grid');
  if (grid) grid.scrollIntoView({ behavior: 'smooth' });
  if (qi) qi.focus();
  document.getElementById('analyzeBtn').disabled = false;
  // Prototype: never auto-run analysis from this button
}

function useLocation() {
  selectedScene.source = 'SATELLITE SCENE';
  selectedScene.status = 'ready';
  document.getElementById('mapTag').textContent = 'scene confirmed';
  const st = document.getElementById('sceneStatus');
  st.classList.add('ready');
  document.getElementById('sceneStatusText').textContent =
    'SELECTED SCENE Â· LAT ' + formatCoord(selectedScene.lat, 'lat') + ' Â· LON ' + formatCoord(selectedScene.lon, 'lon') + ' Â· READY';
  if (sceneMarker) {
    sceneMarker.bindPopup('Selected Scene<br>LAT ' + formatCoord(selectedScene.lat, 'lat') + '<br>LON ' + formatCoord(selectedScene.lon, 'lon') + '<br>STATUS: READY').openPopup();
  }
  // make the confirmed coordinates available to the existing analysis workflow
  window.selectedScene = selectedScene;

  // pull the captured scene straight into the analysis workspace, ready for a free-form question
  selectedDemo = null;
  if (!document.getElementById('slot-img1')) buildUploadCards('default');
  if (!uploadedAssets.img1) {
    fillUploadCard('img1', selectedScene.seed, terrainOptsFromSeed(selectedScene.seed), 'MAP SCENE');
  }
  recountUploaded();
  document.getElementById('sceneSourceNote').classList.remove('hidden');
  showValidation();
  document.getElementById('analyzeBtn').disabled = false;
  document.getElementById('resultsPanel').classList.add('hidden');
  resetTrace();
  mapSceneActive = true;
  const qi = document.getElementById('queryInput');
  if (!qi.value.trim()) {
    qi.value = '';
    qi.placeholder = 'Ask anything about this scene, e.g. What land use is visible here?';
  }
  qi.focus();
  document.querySelector('.ws-grid').scrollIntoView({ behavior: 'smooth' });
}

/* ===================== GEMINI (LIVE AI ANSWER ENGINE) ===================== */
let geminiApiKey = localStorage.getItem('satquery_gemini_key') || '';
let geminiEnabled = localStorage.getItem('satquery_gemini_enabled') === '1';

function refreshGeminiUI() {
  const keyInput = document.getElementById('geminiKeyInput');
  const toggle = document.getElementById('geminiEnableToggle');
  if (keyInput) keyInput.value = geminiApiKey;
  if (toggle) toggle.checked = geminiEnabled;
  const tag = document.getElementById('geminiStatusTag');
  const live = geminiEnabled && !!geminiApiKey;
  if (tag) { tag.textContent = live ? 'live' : 'simulated'; }
  const engineLabel = document.getElementById('aiEngineLabel');
  if (engineLabel) { engineLabel.textContent = live ? 'Gemini (live)' : ('SATQUERY API (' + getSatqueryApiBase() + ')'); }
}

function saveGeminiSettings() {
  geminiApiKey = document.getElementById('geminiKeyInput').value.trim();
  geminiEnabled = document.getElementById('geminiEnableToggle').checked;
  localStorage.setItem('satquery_gemini_key', geminiApiKey);
  localStorage.setItem('satquery_gemini_enabled', geminiEnabled ? '1' : '0');
  refreshGeminiUI();
}
function clearGeminiSettings() {
  geminiApiKey = ''; geminiEnabled = false;
  localStorage.removeItem('satquery_gemini_key');
  localStorage.removeItem('satquery_gemini_enabled');
  refreshGeminiUI();
}

function getSatqueryApiBase() {
  const raw = (typeof window !== 'undefined' && window.VITE_API_URL) ? String(window.VITE_API_URL).trim() : '';
  return (raw || 'http://127.0.0.1:8000').replace(/\/$/, '');
}
const SATQUERY_API_BASE = getSatqueryApiBase();
const SATQUERY_API = SATQUERY_API_BASE + '/api/analyze';

function assetAsFile(asset, fallbackName) {
  if (!asset) return null;
  if (asset.file instanceof File) return asset.file;
  if (asset.file instanceof Blob) {
    return new File([asset.file], asset.name || fallbackName || 'image.png', { type: asset.mime || asset.file.type || 'image/png' });
  }
  return null;
}

async function callSatqueryAnalyze(query) {
  const fd = new FormData();
  fd.append('query', query);
  const f1 = assetAsFile(uploadedAssets.img1, 'image1.png') || assetAsFile(uploadedAssets.opt, 'image1.png');
  // In single image mode, do NOT send image2 even if residual data exists
  const f2 = (currentAnalysisMode === 'single') ? null : (assetAsFile(uploadedAssets.img2, 'image2.png') || assetAsFile(uploadedAssets.sar, 'image2.png'));
  if (f1) fd.append('image1', f1, f1.name || 'image1.png');
  if (f2) fd.append('image2', f2, f2.name || 'image2.png');
  const resp = await fetch(SATQUERY_API, { method: 'POST', body: fd });
  let data = null;
  try { data = await resp.json(); } catch (e) { data = null; }
  if (!resp.ok) {
    const detail = data && (data.detail || data.error || data.message);
    const msg = typeof detail === 'string' ? detail : (Array.isArray(detail) ? detail.map(x => x.msg || JSON.stringify(x)).join('; ') : ('HTTP ' + resp.status));
    throw new Error(msg || ('Analyze failed (' + resp.status + ')'));
  }
  return data;
}

function setAnalyzeLoading(on) {
  const btn = document.getElementById('analyzeBtn');
  if (!btn) return;
  if (on) {
    btn.disabled = true;
    btn.textContent = 'ANALYZINGâ€¦';
    const st = document.getElementById('traceStatus');
    if (st) st.textContent = 'waiting for model';
  } else {
    btn.disabled = false;
    btn.textContent = 'ANALYZE';
  }
}

async function askGemini(promptText, images) {
  if (!geminiEnabled || !geminiApiKey) return null;
  try {
    const parts = [{ text: promptText || '' }];
    (images || []).forEach(im => {
      if (im && im.base64) {
        parts.push({ inlineData: { mimeType: im.mime || 'image/png', data: im.base64 } });
      }
    });
    const resp = await fetch('https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=' + encodeURIComponent(geminiApiKey), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ contents: [{ parts: parts }] })
    });
    if (!resp.ok) return null;
    const data = await resp.json();
    const text = data && data.candidates && data.candidates[0] && data.candidates[0].content
      && data.candidates[0].content.parts && data.candidates[0].content.parts[0]
      && data.candidates[0].content.parts[0].text;
    return text ? text.trim() : null;
  } catch (e) {
    console.error('Gemini request failed', e);
    return null;
  }
}

const ALLOWED_UPLOAD_MIME = {
  'image/png': 1, 'image/jpeg': 1, 'image/jpg': 1, 'image/webp': 1, 'image/gif': 1,
  'image/tiff': 1, 'image/tif': 1, 'image/geotiff': 1
};
const MAX_UPLOAD_BYTES = 12 * 1024 * 1024;

function detectUploadMime(file) {
  if (file && file.type && ALLOWED_UPLOAD_MIME[file.type.toLowerCase()]) return file.type.toLowerCase() === 'image/jpg' ? 'image/jpeg' : file.type.toLowerCase();
  const n = (file && file.name || '').toLowerCase();
  if (n.endsWith('.png')) return 'image/png';
  if (n.endsWith('.jpg') || n.endsWith('.jpeg')) return 'image/jpeg';
  if (n.endsWith('.webp')) return 'image/webp';
  if (n.endsWith('.gif')) return 'image/gif';
  if (n.endsWith('.tif') || n.endsWith('.tiff')) return 'image/tiff';
  return '';
}
function sourceTagInfo(source) {
  if (source === 'MAP_CAPTURE') return { cls: 'tag-capture', text: 'MAP CAPTURE Â· DEMO SCENE' };
  if (source === 'SIMULATED_DEMO') return { cls: 'tag-demo', text: 'SIMULATED DEMO' };
  return { cls: 'tag-upload', text: 'USER UPLOAD' };
}
function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => {
      const dataUrl = String(r.result || '');
      const comma = dataUrl.indexOf(',');
      resolve({ dataUrl, base64: comma >= 0 ? dataUrl.slice(comma + 1) : dataUrl, mime: detectUploadMime(file) || 'image/png' });
    };
    r.onerror = () => reject(new Error('read failed'));
    r.readAsDataURL(file);
  });
}
function collectGeminiImages() {
  return ['img1', 'img2', 'opt', 'sar'].map(k => uploadedAssets[k]).filter(a => a && a.base64);
}
function recountUploaded() {
  uploaded = ['img1', 'img2', 'opt', 'sar'].filter(k => uploadedAssets[k]).length;
  const tag = document.getElementById('uploadTag');
  if (tag) {
    if (currentAnalysisMode === 'single') {
      const singleCount = uploadedAssets.img1 ? 1 : 0;
      tag.textContent = singleCount + ' / 1 loaded';
    } else {
      tag.textContent = Math.min(uploaded, 2) + ' / 2 loaded';
    }
  }
  const note = document.getElementById('sceneSourceNote');
  if (note) {
    const cap = uploadedAssets.img1 && uploadedAssets.img1.source === 'MAP_CAPTURE';
    if (cap) { note.classList.remove('hidden'); note.textContent = 'â—Ž Image from MAP CAPTURE Â· DEMO SCENE (not Sentinel-2). Gemini uses the same upload pipeline.'; }
  }
}
function renderUploadSlot(slotId) {
  const el = document.getElementById('slot-' + slotId);
  if (!el) return;
  const asset = uploadedAssets[slotId];
  const labels = (currentAnalysisMode === 'single') ? { img1: 'Satellite Image', opt: 'Optical', sar: 'SAR' } : { img1: 'Image 1', img2: 'Image 2 (optional)', opt: 'Optical', sar: 'SAR' };
  const lbl = labels[slotId] || slotId;
  if (!asset) {
    el.classList.remove('filled');
    el.innerHTML = '<div class="icn">â‡ª</div><div class="lbl">' + lbl + '</div><div class="fmt">Click to upload Â· or capture from map</div>' +
      '<input class="slot-file" type="file" accept="image/png,image/jpeg,image/webp,image/tiff,.png,.jpg,.jpeg,.tif,.tiff,.webp" onclick="event.stopPropagation()">';
    const inp = el.querySelector('.slot-file');
    el.onclick = function (e) { if (e.target === inp) return; inp.click(); };
    inp.onchange = function () { if (inp.files && inp.files[0]) ingestUploadedImage(slotId, inp.files[0], 'USER_UPLOAD'); };
    return;
  }
  const tag = sourceTagInfo(asset.source);
  el.classList.add('filled');
  el.onclick = null;
  el.innerHTML = '<div class="tagline ' + tag.cls + '">' + tag.text + '</div>' +
    '<img alt="' + lbl + '" src="' + asset.dataUrl + '">' +
    '<div class="slot-ready">READY</div>' +
    '<input class="slot-file" type="file" accept="image/png,image/jpeg,image/webp,image/tiff,.png,.jpg,.jpeg,.tif,.tiff,.webp" onclick="event.stopPropagation()">';
  const inp = el.querySelector('.slot-file');
  el.onclick = function (e) { if (e.target === inp) return; inp.click(); };
  inp.onchange = function () { if (inp.files && inp.files[0]) ingestUploadedImage(slotId, inp.files[0], 'USER_UPLOAD'); };
}
async function ingestUploadedImage(slotId, file, source) {
  if (!file) { showSearchMsg('No image to ingest.'); return { ok: false }; }
  const mime = detectUploadMime(file);
  if (!mime) {
    showSearchMsg('Unsupported format. Use PNG, JPEG, WebP, or TIFF.');
    showValidation(false, 'format');
    return { ok: false };
  }
  if (file.size > MAX_UPLOAD_BYTES) {
    showSearchMsg('Image too large (max 12 MB).');
    showValidation(false, 'size');
    return { ok: false };
  }
  try {
    const parsed = await fileToBase64(file);
    parsed.mime = mime === 'image/tiff' ? 'image/png' : (parsed.mime || mime);
    // Gemini accepts png/jpeg/webp/gif; keep TIFF as data URL preview but send as png if we can
    uploadedAssets[slotId] = { file, dataUrl: parsed.dataUrl, base64: parsed.base64, mime: parsed.mime === 'image/tiff' ? 'image/png' : parsed.mime, source: source || 'USER_UPLOAD', name: file.name };
    lastCaptureSlot = slotId;
    renderUploadSlot(slotId);
    recountUploaded();
    showValidation(true);
    document.getElementById('analyzeBtn').disabled = false;
    document.getElementById('resultsPanel').classList.add('hidden');
    mapSceneActive = true;
    return { ok: true };
  } catch (e) {
    console.error('ingestUploadedImage', e);
    showSearchMsg('Could not read image.');
    return { ok: false };
  }
}

function preferredCaptureLayer() {
  if (!sceneMap) return null;
  if (baseSatLayer && sceneMap.hasLayer(baseSatLayer)) return baseSatLayer;
  if (baseOsmLayer && sceneMap.hasLayer(baseOsmLayer)) return baseOsmLayer;
  return baseSatLayer || baseOsmLayer;
}
function loadCorsTile(url) {
  return new Promise(resolve => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    const t = setTimeout(() => resolve(null), 5000);
    img.onload = () => { clearTimeout(t); resolve(img); };
    img.onerror = () => { clearTimeout(t); resolve(null); };
    img.src = url;
  });
}
function drawCaptureOverlays(ctx, nw, zoom, w, h) {
  function toXY(latlng) {
    const p = sceneMap.project(L.latLng(latlng), zoom);
    return { x: p.x - nw.x, y: p.y - nw.y };
  }
  ctx.save();
  if (aoiPolygon && aoiPoints.length >= 3) {
    ctx.beginPath();
    aoiPoints.forEach((pt, i) => { const xy = toXY(pt); if (i === 0) ctx.moveTo(xy.x, xy.y); else ctx.lineTo(xy.x, xy.y); });
    ctx.closePath();
    ctx.fillStyle = 'rgba(63,216,255,0.12)';
    ctx.strokeStyle = '#3fd8ff';
    ctx.lineWidth = 2;
    ctx.fill(); ctx.stroke();
  } else if (footprintLayer && sceneMap.hasLayer(footprintLayer)) {
    const b = footprintLayer.getBounds();
    const a = toXY(b.getNorthWest()), c = toXY(b.getSouthEast());
    ctx.fillStyle = 'rgba(63,216,255,0.12)';
    ctx.strokeStyle = '#3fd8ff';
    ctx.setLineDash([5, 3]);
    ctx.lineWidth = 1.5;
    ctx.fillRect(a.x, a.y, c.x - a.x, c.y - a.y);
    ctx.strokeRect(a.x, a.y, c.x - a.x, c.y - a.y);
    ctx.setLineDash([]);
  }
  ctx.fillStyle = 'rgba(6,10,18,0.82)';
  ctx.fillRect(8, 8, 248, 34);
  ctx.strokeStyle = 'rgba(255,180,84,0.55)';
  ctx.strokeRect(8, 8, 248, 34);
  ctx.fillStyle = '#ffb454';
  ctx.font = '11px JetBrains Mono, monospace';
  ctx.fillText('MAP CAPTURE Â· DEMO SCENE', 16, 22);
  ctx.fillStyle = '#9fb0c8';
  ctx.font = '9px JetBrains Mono, monospace';
  ctx.fillText('Not a Sentinel-2 product', 16, 35);
  ctx.restore();
}
function fallbackCaptureCanvas(w, h, reason) {
  const canvas = document.createElement('canvas');
  canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext('2d');
  const opts = terrainOptsFromSeed(selectedScene.seed);
  drawTerrain(ctx, w, h, selectedScene.seed, opts);
  ctx.fillStyle = 'rgba(6,10,18,0.72)';
  ctx.fillRect(0, 0, w, 54);
  ctx.fillStyle = '#ffb454';
  ctx.font = '13px JetBrains Mono, monospace';
  ctx.fillText('MAP CAPTURE Â· DEMO SCENE', 12, 20);
  ctx.fillStyle = '#e7eef9';
  ctx.font = '11px JetBrains Mono, monospace';
  ctx.fillText(formatCoord(selectedScene.lat, 'lat') + '  ' + formatCoord(selectedScene.lon, 'lon') + '  Z ' + (sceneMap ? sceneMap.getZoom() : ''), 12, 38);
  ctx.fillStyle = '#9fb0c8';
  ctx.font = '10px JetBrains Mono, monospace';
  ctx.fillText('Not Sentinel-2 Â· ' + reason, 12, h - 14);
  return canvas;
}
async function compositeVisibleTiles(layer) {
  const size = sceneMap.getSize();
  const zoom = sceneMap.getZoom();
  const z = Math.round(zoom);
  const bounds = sceneMap.getBounds();
  const nw = sceneMap.project(bounds.getNorthWest(), z);
  const se = sceneMap.project(bounds.getSouthEast(), z);
  const tileSize = (layer.getTileSize && layer.getTileSize().x) || 256;
  const minX = Math.floor(nw.x / tileSize), maxX = Math.floor((se.x - 1) / tileSize);
  const minY = Math.floor(nw.y / tileSize), maxY = Math.floor((se.y - 1) / tileSize);
  const canvas = document.createElement('canvas');
  canvas.width = Math.max(1, Math.round(size.x));
  canvas.height = Math.max(1, Math.round(size.y));
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#0a111c';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  let ok = 0, fail = 0;
  const jobs = [];
  for (let x = minX; x <= maxX; x++) {
    for (let y = minY; y <= maxY; y++) {
      jobs.push({ x, y, z });
    }
  }
  if (jobs.length > 64) jobs.length = 64;
  const loaded = await Promise.all(jobs.map(async c => {
    let url = '';
    try { url = layer.getTileUrl(c); } catch (e) { return { c, img: null }; }
    const img = await loadCorsTile(url);
    return { c, img };
  }));
  for (const { c, img } of loaded) {
    if (!img) { fail++; continue; }
    const dx = c.x * tileSize - nw.x;
    const dy = c.y * tileSize - nw.y;
    try { ctx.drawImage(img, dx, dy, tileSize, tileSize); ok++; }
    catch (e) { fail++; }
  }
  return { canvas, ctx, nw, z, ok, fail, w: canvas.width, h: canvas.height };
}
function canvasToFile(canvas, name) {
  return new Promise((resolve, reject) => {
    try {
      canvas.toBlob(function (blob) {
        if (!blob) { reject(new Error('toBlob empty')); return; }
        resolve(new File([blob], name, { type: 'image/png' }));
      }, 'image/png');
    } catch (e) { reject(e); }
  });
}
async function captureMapAsImage(targetSlot) {
  // In single image mode, always capture to img1 regardless of targetSlot
  const slot = (currentAnalysisMode === 'single') ? 'img1' : (targetSlot === 'img2' ? 'img2' : 'img1');
  lastCaptureSlot = slot;
  if (!sceneMap) {
    showSearchMsg('Map is not ready. Launch Analysis first.');
    return;
  }
  const bar = document.getElementById('captureBar');
  if (bar) bar.classList.add('capture-busy');
  showSearchMsg('Capturing visible map viewâ€¦');
  try {
    let layer = preferredCaptureLayer();
    let composed = null;
    if (layer) composed = await compositeVisibleTiles(layer);
    if ((!composed || composed.ok === 0) && baseSatLayer && layer !== baseSatLayer) {
      composed = await compositeVisibleTiles(baseSatLayer);
      if (composed && composed.ok > 0) showSearchMsg('Street tiles blocked CORS. Captured Satellite Imagery instead (still not Sentinel-2).');
    }
    let file;
    if (composed && composed.ok > 0) {
      drawCaptureOverlays(composed.ctx, composed.nw, composed.z, composed.w, composed.h);
      file = await canvasToFile(composed.canvas, 'satquery-map-' + slot + '.png');
      showSearchMsg('');
    } else {
      showSearchMsg('Tile layer blocked canvas capture (CORS). Using labeled DEMO composite of the current geographic state â€” not live satellite pixels.');
      const size = sceneMap.getSize();
      const fb = fallbackCaptureCanvas(Math.max(480, size.x), Math.max(280, size.y), 'CORS fallback');
      file = await canvasToFile(fb, 'satquery-map-' + slot + '-demo.png');
    }
    const res = await ingestUploadedImage(slot, file, 'MAP_CAPTURE');
    if (res.ok) {
      if (!document.getElementById('queryInput').value.trim() && slot === 'img2' && uploadedAssets.img1) {
        document.getElementById('queryInput').placeholder = 'e.g. What changed between these two images?';
      }
      const grid = document.querySelector('.ws-grid');
      if (grid) grid.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  } catch (e) {
    console.error('captureMapAsImage', e);
    showSearchMsg('Could not capture this tile layer (often CORS). Try Satellite Imagery, or upload a file.');
  } finally {
    if (bar) bar.classList.remove('capture-busy');
  }
}
function clearCapture(slot) {
  const s = slot || lastCaptureSlot || 'img1';
  if (uploadedAssets[s] && uploadedAssets[s].source === 'MAP_CAPTURE') {
    uploadedAssets[s] = null;
  } else if (!slot) {
    ['img1', 'img2'].forEach(k => { if (uploadedAssets[k] && uploadedAssets[k].source === 'MAP_CAPTURE') uploadedAssets[k] = null; });
  } else {
    uploadedAssets[s] = null;
  }
  ['img1', 'img2'].forEach(k => { if (document.getElementById('slot-' + k)) renderUploadSlot(k); });
  recountUploaded();
  if (uploaded === 0) {
    document.getElementById('analyzeBtn').disabled = true;
    document.getElementById('validationBlock').classList.add('hidden');
    document.getElementById('sceneSourceNote').classList.add('hidden');
  }
  showSearchMsg('');
}

/* ===================== PROCEDURAL SATELLITE IMAGERY ===================== */
function mulberry32(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }

function drawTerrain(ctx, w, h, seed, opts) {
  opts = opts || {};
  const rnd = mulberry32(seed);
  // base soil/vegetation gradient
  const g = ctx.createLinearGradient(0, 0, w, h);
  g.addColorStop(0, '#3a4a2e'); g.addColorStop(0.5, '#4d5e35'); g.addColorStop(1, '#39492a');
  ctx.fillStyle = g; ctx.fillRect(0, 0, w, h);
  // patchwork agricultural fields
  const cols = 6 + Math.floor(rnd() * 3), rows = 5 + Math.floor(rnd() * 3);
  const palette = ['#5b6d34', '#6f8341', '#46592b', '#7a8f4a', '#3f4f28', '#8a9a55'];
  for (let i = 0; i < cols; i++) {
    for (let j = 0; j < rows; j++) {
      const x = i * w / cols + rnd() * 6, y = j * h / rows + rnd() * 6, cw = w / cols - rnd() * 8, ch = h / rows - rnd() * 8;
      ctx.fillStyle = palette[Math.floor(rnd() * palette.length)];
      ctx.globalAlpha = 0.55 + rnd() * 0.3;
      ctx.fillRect(x, y, cw, ch);
    }
  }
  ctx.globalAlpha = 1;
  // roads
  ctx.strokeStyle = 'rgba(180,175,160,0.55)'; ctx.lineWidth = Math.max(1.2, w * 0.004);
  for (let r = 0; r < 3; r++) {
    ctx.beginPath();
    let y = (0.15 + r * 0.32) * h + rnd() * 20;
    ctx.moveTo(0, y);
    for (let x = 0; x <= w; x += w / 8) { y += (rnd() - 0.5) * 22; ctx.lineTo(x, y); }
    ctx.stroke();
  }
  ctx.beginPath(); ctx.moveTo(w * (0.3 + rnd() * 0.1), 0); ctx.lineTo(w * (0.42 + rnd() * 0.1), h); ctx.stroke();
  // water body
  if (!opts.noWater) {
    ctx.fillStyle = '#1c4f66';
    ctx.beginPath();
    const cx = opts.waterX !== undefined ? opts.waterX * w : w * 0.68, cy = opts.waterY !== undefined ? opts.waterY * h : h * 0.62;
    const rx = w * 0.14, ry = h * 0.11;
    ctx.ellipse(cx, cy, rx, ry * 1.1, 0.4, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = 'rgba(60,140,170,0.5)';
    ctx.beginPath(); ctx.ellipse(cx - rx * 0.3, cy - ry * 0.2, rx * 0.5, ry * 0.4, 0.3, 0, Math.PI * 2); ctx.fill();
    if (opts.highlightWater) {
      ctx.strokeStyle = '#3fd8ff'; ctx.lineWidth = 2.5; ctx.setLineDash([]);
      ctx.beginPath(); ctx.ellipse(cx, cy, rx * 1.25, ry * 1.35, 0.4, 0, Math.PI * 2); ctx.stroke();
      ctx.fillStyle = 'rgba(63,216,255,0.12)'; ctx.beginPath(); ctx.ellipse(cx, cy, rx * 1.25, ry * 1.35, 0.4, 0, Math.PI * 2); ctx.fill();
    }
  }
  // built-up clusters
  const buCount = opts.builtCount !== undefined ? opts.builtCount : 5;
  ctx.fillStyle = '#8a8578';
  for (let k = 0; k < buCount; k++) {
    let bx, by;
    if (opts.buCluster) { bx = opts.buCluster.x * w + (rnd() - 0.5) * w * 0.14; by = opts.buCluster.y * h + (rnd() - 0.5) * h * 0.14; }
    else { bx = rnd() * w; by = rnd() * h; }
    const s = 6 + rnd() * 10;
    ctx.fillStyle = rnd() > 0.5 ? '#9a9486' : '#7d7a6c';
    ctx.fillRect(bx, by, s, s * 0.7);
  }
  // subtle noise
  const id = ctx.getImageData(0, 0, w, h);
  for (let i = 0; i < id.data.length; i += 4) {
    const n = (rnd() - 0.5) * 14;
    id.data[i] += n; id.data[i + 1] += n; id.data[i + 2] += n;
  }
  ctx.putImageData(id, 0, 0);
  return { waterCx: (opts.waterX !== undefined ? opts.waterX : 0.68), waterCy: (opts.waterY !== undefined ? opts.waterY : 0.62) };
}

function drawNIR(ctx, w, h, seed, opts) {
  drawTerrain(ctx, w, h, seed, opts || {});
  const id = ctx.getImageData(0, 0, w, h);
  for (let i = 0; i < id.data.length; i += 4) {
    const r = id.data[i], g = id.data[i + 1], b = id.data[i + 2];
    // Simulated false-color IR: vegetation (green) â†’ red/magenta, water stays dark blue
    id.data[i] = Math.min(255, g * 1.15);
    id.data[i + 1] = Math.min(255, r * 0.45 + b * 0.25);
    id.data[i + 2] = Math.min(255, b * 0.85);
  }
  ctx.putImageData(id, 0, 0);
}

function drawSAR(ctx, w, h, seed) {
  const rnd = mulberry32(seed);
  ctx.fillStyle = '#1a1e22'; ctx.fillRect(0, 0, w, h);
  const id = ctx.getImageData(0, 0, w, h);
  for (let i = 0; i < id.data.length; i += 4) {
    const speckle = rnd() * rnd() * 255;
    const v = Math.min(255, 60 + speckle * 0.55);
    id.data[i] = v; id.data[i + 1] = v; id.data[i + 2] = v * 1.02;
  }
  ctx.putImageData(id, 0, 0);
  // bright structure returns
  ctx.fillStyle = 'rgba(230,230,235,0.85)';
  for (let k = 0; k < 70; k++) {
    const x = rnd() * w, y = rnd() * h;
    ctx.fillRect(x, y, 1.5 + rnd() * 3, 1.5 + rnd() * 3);
  }
  // dark water patch
  ctx.fillStyle = 'rgba(5,8,10,0.75)';
  ctx.beginPath(); ctx.ellipse(w * 0.68, h * 0.62, w * 0.14, h * 0.12, 0.4, 0, Math.PI * 2); ctx.fill();
}

function drawChangeMap(ctx, w, h, seed, cluster) {
  const rnd = mulberry32(seed);
  ctx.fillStyle = '#050708'; ctx.fillRect(0, 0, w, h);
  // scattered white "no significant change" texture is background; red = change cluster
  ctx.fillStyle = 'rgba(255,255,255,0.06)';
  for (let k = 0; k < 400; k++) { ctx.fillRect(rnd() * w, rnd() * h, 1, 1); }
  ctx.fillStyle = '#ff5252';
  const cx = cluster.x * w, cy = cluster.y * h;
  for (let k = 0; k < 26; k++) {
    const a = rnd() * Math.PI * 2, r = rnd() * w * 0.11;
    const bx = cx + Math.cos(a) * r * 1.4, by = cy + Math.sin(a) * r;
    ctx.beginPath(); ctx.arc(bx, by, 4 + rnd() * 10, 0, Math.PI * 2); ctx.fill();
  }
}

function drawFusion(ctx, w, h, seed, cluster) {
  const rnd = mulberry32(seed);
  drawTerrain(ctx, w, h, seed, { builtCount: 3, waterX: 0.68, waterY: 0.62 });
  ctx.globalCompositeOperation = 'overlay';
  ctx.fillStyle = 'rgba(80,80,90,0.5)';
  const id = ctx.getImageData(0, 0, w, h);
  ctx.globalCompositeOperation = 'source-over';
  // red = built-up overlay near cluster
  ctx.fillStyle = 'rgba(255,70,70,0.55)';
  for (let k = 0; k < 36; k++) {
    const a = rnd() * Math.PI * 2, r = rnd() * w * 0.1;
    ctx.fillRect(cluster.x * w + Math.cos(a) * r * 1.3 - 3, cluster.y * h + Math.sin(a) * r - 3, 5 + rnd() * 5, 5 + rnd() * 5);
  }
  // blue = water overlay
  ctx.fillStyle = 'rgba(70,140,255,0.5)';
  ctx.beginPath(); ctx.ellipse(w * 0.68, h * 0.62, w * 0.145, h * 0.125, 0.4, 0, Math.PI * 2); ctx.fill();
}

function canvasTo(el, w, h, fn) {
  const c = document.createElement('canvas'); c.width = w; c.height = h;
  const ctx = c.getContext('2d'); fn(ctx);
  el.innerHTML = ''; el.appendChild(c);
}

/* Hero canvas */
(function () {
  const c = document.getElementById('heroCanvas'); const ctx = c.getContext('2d');
  drawTerrain(ctx, 640, 440, 42, { waterX: 0.72, waterY: 0.6, builtCount: 8 });
})();

/* ===================== APP DATA ===================== */
const BUILT_CLUSTER = { x: 0.30, y: 0.32 };
function sceneRenderSeed() { return (selectedScene && selectedScene.seed) ? selectedScene.seed : 101; }

const DEMOS = {
  vqa: {
    task: 'Single-Image VQA', model: 'Remote Sensing VQA Model',
    query: "What is visible in this image?",
    trace: ['Query classified', 'Input validated', 'Task identified: Visual Question Answering', 'Specialist model selected', 'Processing satellite imagery', 'Generating answer', 'Result generated'],
    confidence: 94,
    render(body) {
      body.innerHTML = `
        <div class="ws-grid" style="grid-template-columns:1fr 1fr;">
          <div class="img-box"><div id="vqaImg"></div><div class="cap">OPTICAL SCENE</div></div>
          <div>
            <div class="answer-box"><div class="lbl2">QUESTION</div><p>What is visible in this image?</p></div>
            <div class="answer-box" style="margin-top:12px;"><div class="lbl2">ANSWER</div><p>Several agricultural fields, roads, built-up structures and a water body are visible.</p></div>
            <div class="conf-row"><div class="conf-bar"><div class="conf-fill" style="width:94%;"></div></div><div class="conf-val">94%</div></div>
          </div>
        </div>`;
      canvasTo(document.getElementById('vqaImg'), 480, 320, ctx => drawTerrain(ctx, 480, 320, sceneRenderSeed(), terrainOptsFromSeed(sceneRenderSeed())));
    }
  },
  grounding: {
    task: 'Region Grounding', model: 'Region Grounding Model',
    query: "Highlight the water body.",
    trace: ['Query classified', 'Region Grounding selected', 'Grounding model loaded', 'Water region detected', 'Mask generated', 'Result displayed'],
    confidence: 95,
    render(body) {
      body.innerHTML = `
        <div class="ws-grid" style="grid-template-columns:1fr 1fr;">
          <div class="img-box"><div id="grImg"></div><div class="cap">GROUNDED REGION</div></div>
          <div>
            <div class="answer-box"><div class="lbl2">INSTRUCTION</div><p>Highlight the water body.</p></div>
            <div class="answer-box" style="margin-top:12px;"><div class="lbl2">RESULT</div><p><b style="color:var(--cyan);">Water body detected.</b> A single contiguous water region was localized in the eastern portion of the scene.</p></div>
            <div class="conf-row"><div class="conf-bar"><div class="conf-fill" style="width:95%;"></div></div><div class="conf-val">95%</div></div>
          </div>
        </div>`;
      canvasTo(document.getElementById('grImg'), 480, 320, ctx => drawTerrain(ctx, 480, 320, sceneRenderSeed(), Object.assign({}, terrainOptsFromSeed(sceneRenderSeed()), { highlightWater: true })));
    }
  },
  change: {
    task: 'Bi-Temporal Change Analysis', model: 'Change Understanding Model',
    query: "What changed between these two images?",
    trace: ['Query classified', 'Input validation: 2 images Â· optical', 'Task identified: Bi-temporal Change Analysis', 'Specialist model selected', 'Processing satellite imagery', 'Generating change map', 'Generating explanation', 'Result generated'],
    confidence: 91,
    render(body) {
      body.innerHTML = `
        <div class="compare-row">
          <div class="img-box"><div id="chImg1"></div><div class="cap">2022</div></div>
          <div class="arrow-mid">â†’</div>
          <div class="img-box"><div id="chImg2"></div><div class="cap">2026</div></div>
        </div>
        <div class="img-box" style="max-width:340px;margin:0 auto 6px;"><div id="chMap"></div><div class="cap">CHANGE MAP</div></div>
        <div class="answer-box"><div class="lbl2">AI ANSWER</div><p>Built-up area increased in the northeastern portion of the scene. New structures and roads are visible compared with 2022.</p></div>
        <div class="conf-row"><div class="conf-bar"><div class="conf-fill" style="width:91%;"></div></div><div class="conf-val">91%</div></div>`;
      const s = sceneRenderSeed(); const o = terrainOptsFromSeed(s);
      canvasTo(document.getElementById('chImg1'), 300, 220, ctx => drawTerrain(ctx, 300, 220, s, o));
      canvasTo(document.getElementById('chImg2'), 300, 220, ctx => drawTerrain(ctx, 300, 220, s, Object.assign({}, o, { buCluster: BUILT_CLUSTER })));
      // add extra built cluster manually for change image
      const c2 = document.getElementById('chImg2').querySelector('canvas');
      const ctx2 = c2.getContext('2d');
      const rnd = mulberry32(555);
      ctx2.fillStyle = '#a8a294';
      for (let k = 0; k < 14; k++) { const bx = BUILT_CLUSTER.x * 300 + (rnd() - 0.5) * 40, by = BUILT_CLUSTER.y * 220 + (rnd() - 0.5) * 40; ctx2.fillRect(bx, by, 6 + rnd() * 5, 5 + rnd() * 4); }
      canvasTo(document.getElementById('chMap'), 300, 220, ctx => drawChangeMap(ctx, 300, 220, 301, BUILT_CLUSTER));
    }
  },
  fusion: {
    task: 'Optical + SAR Fusion', model: 'Optical + SAR Fusion Model',
    query: "Fuse optical and SAR imagery to identify built-up areas.",
    trace: ['Query classified', 'Input validation: optical + SAR', 'Task identified: Optical + SAR Fusion', 'Specialist model selected', 'Cross-modal alignment', 'Generating fusion result', 'Generating explanation', 'Result generated'],
    confidence: 93,
    render(body) {
      body.innerHTML = `
        <div class="triple">
          <div class="img-box"><div id="fuOpt"></div><div class="cap">OPTICAL</div></div>
          <div class="img-box"><div id="fuSar"></div><div class="cap">SAR</div></div>
          <div class="img-box"><div id="fuRes"></div><div class="cap">FUSION RESULT</div></div>
        </div>
        <div class="legend" style="flex-direction:row;gap:20px;margin:12px 0 4px;">
          <span><span class="sw" style="background:#ff4646;"></span>Built-up</span>
          <span><span class="sw" style="background:#4686ff;"></span>Water</span>
        </div>
        <div class="answer-box"><div class="lbl2">ANSWER</div><p>Built-up areas are identified using complementary optical and SAR information. SAR improves detection of structures under varying illumination conditions.</p></div>
        <div class="conf-row"><div class="conf-bar"><div class="conf-fill" style="width:93%;"></div></div><div class="conf-val">93%</div></div>`;
      canvasTo(document.getElementById('fuOpt'), 260, 190, ctx => drawTerrain(ctx, 260, 190, sceneRenderSeed(), terrainOptsFromSeed(sceneRenderSeed())));
      canvasTo(document.getElementById('fuSar'), 260, 190, ctx => drawSAR(ctx, 260, 190, sceneRenderSeed()));
      canvasTo(document.getElementById('fuRes'), 260, 190, ctx => drawFusion(ctx, 260, 190, sceneRenderSeed(), BUILT_CLUSTER));
    }
  }
};

const MODEL_REGISTRY = [
  { name: 'Remote Sensing VQA', task: 'Visual Question Answering', desc: 'Answers free-form natural-language questions about a single satellite scene.' },
  { name: 'Region Grounding', task: 'Text-guided region localization', desc: 'Localizes the image region referred to by a natural-language instruction.' },
  { name: 'Change Understanding', task: 'Bi-temporal analysis', desc: 'Compares two dates of imagery and generates a pixel-level change map.' },
  { name: 'Optical + SAR Fusion', task: 'Cross-modal satellite analysis', desc: 'Combines optical and radar data for robust, all-weather detection.' },
  { name: 'Captioning', task: 'Scene description', desc: 'Generates a natural-language caption summarizing the contents of a scene.' },
];

let uploaded = 0, selectedDemo = null, historyLog = [], mapSceneActive = false;
let uploadedAssets = { img1: null, img2: null, opt: null, sar: null };
let lastCaptureSlot = 'img1';
window.uploadedAssets = uploadedAssets;
window.selectedScene = selectedScene;

/* ===================== NAV / VIEW LOGIC ===================== */
function showLanding() { document.getElementById('landing').style.display = 'block'; document.getElementById('app').classList.remove('show'); updateLandingHud(); }
function launchApp(view) {
  document.getElementById('landing').style.display = 'none';
  document.getElementById('app').classList.add('show');
  switchView(view || 'workspace');
  setTimeout(() => { initSceneMap(); ensureMapSize(); }, 60);
  setTimeout(() => ensureMapSize(), 280);
}
function scrollToCaps() { document.getElementById('caps').scrollIntoView({ behavior: 'smooth' }); }
function scrollToTrace() { document.querySelector('.ws-grid').scrollIntoView({ behavior: 'smooth' }); }

function switchView(v) {
  document.querySelectorAll('.sb-item[data-view]').forEach(el => el.classList.toggle('active', el.dataset.view === v));
  ['workspace', 'models', 'history', 'datasets', 'settings'].forEach(id => {
    document.getElementById('view-' + id).classList.toggle('hidden', id !== v);
  });
  if (v === 'models') renderModels();
  if (v === 'history') renderHistory();
  if (v === 'settings') {
    refreshGeminiUI();
    const sel = document.getElementById('coordFormatSelect');
    if (sel) sel.value = coordFormat;
  }
  if (v === 'workspace') {
    if (!sceneMap) setTimeout(() => { initSceneMap(); ensureMapSize(); }, 60);
    else { ensureMapSize(); setTimeout(() => ensureMapSize(), 200); }
  }
}

function renderModels() {
  const grid = document.getElementById('modelGrid');
  grid.innerHTML = MODEL_REGISTRY.map(m => `
    <div class="model-card">
      <h3>${m.name}</h3>
      <div class="mtask">${m.task}</div>
      <p>${m.desc}</p>
      <div class="status"><span style="width:6px;height:6px;border-radius:50%;background:var(--green);display:inline-block;"></span>READY</div>
    </div>`).join('');
}
function renderHistory() {
  const p = document.getElementById('historyPanel');
  if (!historyLog.length) { p.innerHTML = '<div style="font-family:var(--mono);font-size:12px;color:var(--tx3);padding:10px 4px;">No analyses run yet this session.</div>'; return; }
  p.innerHTML = historyLog.slice().reverse().map((h, revI) => {
    const idx = historyLog.length - 1 - revI;
    const geo = [h.sceneId, h.lat != null ? formatCoord(h.lat, 'lat') : '', h.lon != null ? formatCoord(h.lon, 'lon') : ''].filter(Boolean).join(' Â· ');
    return `
    <div class="hist-item" onclick="restoreHistory(${idx})">
      <div><div class="hl">${h.query}</div><div class="ht">${h.task} Â· ${h.time}${geo ? ' Â· ' + geo : ''}</div></div>
      <span class="badge">${h.confidence}%</span>
    </div>`;
  }).join('');
}
function restoreHistory(idx) {
  const h = historyLog[idx];
  if (!h || h.lat == null) return;
  switchView('workspace');
  setTimeout(() => {
    initSceneMap();
    placeMarker(h.lat, h.lon);
    if (sceneMap) sceneMap.setView([h.lat, h.lon], 12);
    ensureMapSize();
  }, 80);
}

/* ===================== UPLOAD CARDS ===================== */
function buildUploadCards(kind) {
  const grid = document.getElementById('uploadGrid');
  const heading = document.getElementById('uploadHeading');
  let slots;
  if (kind === 'change') {
    slots = [{ k: 'img1', l: 'Image 1 (2022)' }, { k: 'img2', l: 'Image 2 (2026)' }];
    if (heading) heading.textContent = 'â–£ Upload Satellite Images';
  } else if (kind === 'fusion') {
    slots = [{ k: 'opt', l: 'Optical' }, { k: 'sar', l: 'SAR' }];
    if (heading) heading.textContent = 'â–£ Upload Satellite Images';
  } else if (currentAnalysisMode === 'single') {
    // Single Image mode: only one upload slot
    slots = [{ k: 'img1', l: 'Satellite Image' }];
    if (heading) heading.textContent = 'â–£ Upload Satellite Image';
  } else {
    slots = [{ k: 'img1', l: 'Image 1' }, { k: 'img2', l: 'Image 2 (optional)' }];
    if (heading) heading.textContent = 'â–£ Upload Satellite Images';
  }
  grid.innerHTML = '';
  slots.forEach((s, i) => {
    const div = document.createElement('div');
    div.className = 'upload-card';
    div.id = 'slot-' + s.k;
    grid.appendChild(div);
    renderUploadSlot(s.k);
  });
  // In Single Image mode, add an OR divider and Capture from Map button
  if (currentAnalysisMode === 'single' && kind !== 'change' && kind !== 'fusion') {
    const orDiv = document.createElement('div');
    orDiv.className = 'single-capture-section';
    orDiv.innerHTML =
      '<div class="single-or-divider"><span class="single-or-line"></span><span class="single-or-text">OR</span><span class="single-or-line"></span></div>' +
      '<button class="btn-capture-map" type="button" onclick="captureMapAsImage(\'img1\')"><span class="capture-icon">â—Ž</span> Capture Current Map</button>';
    grid.appendChild(orDiv);
  }
}
buildUploadCards('default');
refreshGeminiUI();
refreshQueryChips();
updateLandingHud();
updateMapHud();
selectedScene.seed = coordToSeed(selectedScene.lat, selectedScene.lon);
window.addEventListener('resize', ensureMapSize);

function fillUploadCard(slotId, seed, opts, label) {
  const w = 260, h = 150;
  const c = document.createElement('canvas'); c.width = w; c.height = h;
  const ctx = c.getContext('2d');
  if (opts && opts.sar) drawSAR(ctx, w, h, seed); else drawTerrain(ctx, w, h, seed, opts);
  c.toBlob(function (blob) {
    if (!blob) return;
    ingestUploadedImage(slotId, new File([blob], (label || slotId) + '-demo.png', { type: 'image/png' }), 'SIMULATED_DEMO');
  }, 'image/png');
}

/* ===================== VALIDATION ===================== */
function showValidation(ok, which) {
  document.getElementById('validationBlock').classList.remove('hidden');
  const list = document.getElementById('validationList');
  const good = ok !== false;
  const fmtOk = which !== 'format';
  const sizeOk = which !== 'size';
  list.innerHTML = `
    <div class="vitem ${good ? '' : 'bad'}"><span class="chk">${good ? 'âœ“' : '!'}</span>Valid satellite image</div>
    <div class="vitem ${fmtOk ? '' : 'bad'}"><span class="chk">${fmtOk ? 'âœ“' : '!'}</span>Image format supported (PNG / JPEG / WebP / TIFF)</div>
    <div class="vitem ${sizeOk ? '' : 'bad'}"><span class="chk">${sizeOk ? 'âœ“' : '!'}</span>Size within 12 MB</div>
    <div class="vitem"><span class="chk">âœ“</span>Sources: USER UPLOAD Â· MAP CAPTURE Â· SIMULATED DEMO</div>
    <div class="vitem"><span class="chk">âœ“</span>Image count valid</div>`;
}

/* ===================== DEMO SELECTION ===================== */
function pickDemo(key) {
  selectedDemo = key;
  document.getElementById('queryInput').value = DEMOS[key].query;
  const kind = key === 'change' ? 'change' : (key === 'fusion' ? 'fusion' : 'default');
  uploadedAssets.img1 = uploadedAssets.img2 = uploadedAssets.opt = uploadedAssets.sar = null;
  buildUploadCards(kind);
  const seed = selectedScene.seed;
  const opts = terrainOptsFromSeed(seed);
  if (mapSceneActive) {
    if (key === 'vqa' || key === 'grounding') {
      fillUploadCard('img1', seed, Object.assign({}, opts, { highlightWater: key === 'grounding' }), 'MAP SCENE Â· DEMO');
    } else if (key === 'change') {
      fillUploadCard('img1', seed, opts, 'T0 Â· DEMO');
      fillUploadCard('img2', seed, Object.assign({}, opts, { buCluster: BUILT_CLUSTER }), 'T1 Â· DEMO');
    } else if (key === 'fusion') {
      fillUploadCard('opt', seed, opts, 'OPTICAL Â· DEMO');
      fillUploadCard('sar', seed, { sar: true }, 'SAR Â· DEMO');
    }
  } else if (key === 'vqa' || key === 'grounding') {
    fillUploadCard('img1', 101 + (key === 'grounding' ? 1000 : 0), { waterX: 0.7, waterY: 0.6, builtCount: 6 }, 'OPTICAL');
  } else if (key === 'change') {
    fillUploadCard('img1', 201, { waterX: 0.68, waterY: 0.62, builtCount: 3 }, '2022');
    fillUploadCard('img2', 201, { waterX: 0.68, waterY: 0.62, builtCount: 3, buCluster: BUILT_CLUSTER }, '2026');
  } else if (key === 'fusion') {
    fillUploadCard('opt', 401, { waterX: 0.68, waterY: 0.62, builtCount: 3 }, 'OPTICAL');
    fillUploadCard('sar', 402, { sar: true }, 'SAR');
  }
  const isSingleDemo = (key === 'vqa' || key === 'grounding');
  if (currentAnalysisMode === 'single' && isSingleDemo) {
    uploaded = 1;
    document.getElementById('uploadTag').textContent = '1 / 1 loaded';
  } else {
    uploaded = 2;
    document.getElementById('uploadTag').textContent = '2 / 2 loaded';
  }
  showValidation();
  document.getElementById('analyzeBtn').disabled = false;
  document.getElementById('resultsPanel').classList.add('hidden');
  resetTrace();
}

function loadDemo() { pickDemo('change'); }

function newAnalysis() {
  selectedDemo = null; uploaded = 0; mapSceneActive = false;
  uploadedAssets.img1 = uploadedAssets.img2 = uploadedAssets.opt = uploadedAssets.sar = null;
  lastCaptureSlot = 'img1';
  document.getElementById('queryInput').value = '';
  document.getElementById('uploadTag').textContent = (currentAnalysisMode === 'single') ? '0 / 1 loaded' : '0 / 2 loaded';
  document.getElementById('validationBlock').classList.add('hidden');
  document.getElementById('analyzeBtn').disabled = true;
  document.getElementById('analyzeBtn').textContent = 'ANALYZE';
  document.getElementById('resultsPanel').classList.add('hidden');
  document.getElementById('sceneSourceNote').classList.add('hidden');
  buildUploadCards('default');
  resetTrace();
  clearAoi();
  clearCompare();
  clearAnalysisOverlay();
  clearFootprint();
  if (analysisGridLayer && sceneMap && sceneMap.hasLayer(analysisGridLayer)) sceneMap.removeLayer(analysisGridLayer);
  setGeoMode('SCENE');
  aoiLocked = false;
  previewMode = 'RGB';
  ['RGB', 'NIR', 'SAR'].forEach(m => { const b = document.getElementById('pv' + m); if (b) b.classList.toggle('active', m === 'RGB'); });
  const prev = document.getElementById('scenePreview');
  if (prev) { prev.innerHTML = 'click map to capture'; prev.style.display = 'flex'; }
  selectedScene.status = 'pending';
  selectedScene.source = 'DEFAULT';
  const st = document.getElementById('sceneStatus');
  if (st) st.classList.remove('ready');
  const sst = document.getElementById('sceneStatusText');
  if (sst) sst.textContent = 'NO SCENE SELECTED';
  document.getElementById('mapTag').textContent = 'click map to pick scene';
  showSearchMsg('');
  window.scrollTo({ top: 0, behavior: 'smooth' });
  ensureMapSize();
}

function resetTrace() {
  document.getElementById('traceEmpty').classList.remove('hidden');
  document.getElementById('traceList').classList.add('hidden');
  document.getElementById('traceMeta').classList.add('hidden');
  document.getElementById('traceStatus').textContent = 'idle';
}

/* ===================== ANALYSIS RUN ===================== */
function runAnalysis() {
  if (!selectedDemo) {
    // infer from typed query
    const q = (document.getElementById('queryInput').value || '').toLowerCase();
    if (q.includes('chang')) selectedDemo = 'change';
    else if (q.includes('highlight') || q.includes('where')) selectedDemo = 'grounding';
    else if (q.includes('sar') || q.includes('fus')) selectedDemo = 'fusion';
    else selectedDemo = 'vqa';
  }
  const demo = DEMOS[selectedDemo];
  setAnalyzeLoading(true);
  document.getElementById('resultsPanel').classList.add('hidden');
  document.getElementById('traceEmpty').classList.add('hidden');
  document.getElementById('traceStatus').textContent = 'running';

  const geoSteps = [];
  geoSteps.push('Scene georeference Â· ' + getSceneId() + ' Â· ' + formatCoord(selectedScene.lat, 'lat') + ' ' + formatCoord(selectedScene.lon, 'lon'));
  geoSteps.push('Metadata lock Â· Sentinel-2 L2A/DEMO Â· 10m Â· cloud ' + demoCloudCover(selectedScene.seed) + '% DEMO');
  if (aoiLocked) geoSteps.push('AOI locked Â· ~' + aoiAreaKm2.toFixed(2) + ' kmÂ² (approx) Â· DEMO ESTIMATE');
  if (geoMode === 'COMPARE' && compareA && compareB) geoSteps.push('Compare pair Â· ' + haversineKm(compareA[0], compareA[1], compareB[0], compareB[1]).toFixed(2) + ' km baseline');
  const imgs = collectGeminiImages();
  if (uploadedAssets.img1) geoSteps.push((currentAnalysisMode === 'single' ? 'Image' : 'Image 1') + ' ready Â· ' + sourceTagInfo(uploadedAssets.img1.source).text);
  if (uploadedAssets.img2 && currentAnalysisMode !== 'single') geoSteps.push('Image 2 ready Â· ' + sourceTagInfo(uploadedAssets.img2.source).text);
  geoSteps.push('POST ' + SATQUERY_API + ' Â· query + ' + (currentAnalysisMode === 'single' ? 'image' : 'image1/image2') + ' via multipart FormData');
  const steps = geoSteps.concat(demo.trace);

  const traceList = document.getElementById('traceList');
  traceList.classList.remove('hidden');
  traceList.innerHTML = steps.map((t, i) => `
    <div class="trace-step" id="tstep-${i}">
      <div>
        <div class="t-title">${t}</div>
        <div class="t-sub">step ${i + 1} of ${steps.length}</div>
      </div>
    </div>`).join('');

  let i = 0;
  function stepThrough() {
    if (i > 0) { const prev = document.getElementById('tstep-' + (i - 1)); if (prev) { prev.classList.remove('active'); prev.classList.add('done'); } }
    if (i < steps.length) {
      document.getElementById('tstep-' + i).classList.add('active');
      i++;
      setTimeout(stepThrough, 420);
    } else {
      finishAnalysisRun(demo);
    }
  }
  stepThrough();
}

function buildGeminiPrompt(demo) {
  const qEl = document.getElementById('queryInput');
  const q = (qEl && qEl.value.trim()) || demo.query;
  const src1 = uploadedAssets.img1 ? sourceTagInfo(uploadedAssets.img1.source).text : 'none';
  const src2 = uploadedAssets.img2 ? sourceTagInfo(uploadedAssets.img2.source).text : 'none';
  return 'You are SATQUERY AI, an SIH prototype for satellite-image understanding. ' +
    'Images may be USER UPLOAD, MAP CAPTURE (browser map view, NOT a Sentinel-2 product), or SIMULATED DEMO. ' +
    'Image 1 source: ' + src1 + '. Image 2 source: ' + src2 + '. ' +
    'Scene ID ' + getSceneId() + ' at ' + formatCoord(selectedScene.lat, 'lat') + ' ' + formatCoord(selectedScene.lon, 'lon') + '. ' +
    'User query: ' + q;
}

async function finishAnalysisRun(demo) {
  const qEl = document.getElementById('queryInput');
  const query = (qEl && qEl.value.trim()) || demo.query;
  document.getElementById('traceStatus').textContent = 'waiting for model';
  let api = null, apiError = null;
  try {
    api = await callSatqueryAnalyze(query);
  } catch (e) {
    apiError = (e && e.message) ? e.message : ('Could not reach SATQUERY API at ' + SATQUERY_API_BASE);
    console.error('SATQUERY API', e);
  }
  let geminiText = null;
  if (!api && geminiEnabled && geminiApiKey) {
    document.getElementById('traceStatus').textContent = 'gemini';
    geminiText = await askGemini(buildGeminiPrompt(demo), collectGeminiImages());
  }
  if (api && Array.isArray(api.trace) && api.trace.length) {
    const extra = api.trace;
    const list = document.getElementById('traceList');
    extra.forEach((t, idx) => {
      const div = document.createElement('div');
      div.className = 'trace-step done';
      div.innerHTML = '<div><div class="t-title"></div><div class="t-sub">backend step ' + (idx + 1) + '</div></div>';
      div.querySelector('.t-title').textContent = t;
      list.appendChild(div);
    });
  }
  document.getElementById('traceStatus').textContent = apiError ? 'error' : 'complete';
  document.getElementById('traceMeta').classList.remove('hidden');
  document.getElementById('metaModel').textContent = (api && api.model) ? api.model : ((geminiText ? 'Gemini + ' : '') + demo.model);
  const conf = (api && api.confidence != null) ? api.confidence : demo.confidence;
  document.getElementById('metaConf').textContent = conf + '%';
  showResults(demo, geminiText, api, apiError);
  setAnalyzeLoading(false);
  historyLog.push({
    query: query,
    task: (api && api.task) || demo.task,
    confidence: conf,
    latitude: selectedScene.lat,
    longitude: selectedScene.lon,
    lat: selectedScene.lat,
    lon: selectedScene.lon,
    sceneId: getSceneId(),
    time: new Date().toLocaleTimeString()
  });
  buildSimulatedAnalysisOverlay();
}

function attachCapturedToResults() {
  const map = {
    vqaImg: uploadedAssets.img1,
    grImg: uploadedAssets.img1,
    chImg1: uploadedAssets.img1,
    chImg2: uploadedAssets.img2 || uploadedAssets.img1,
    fuOpt: uploadedAssets.opt || uploadedAssets.img1,
    fuSar: uploadedAssets.sar || uploadedAssets.img2
  };
  Object.keys(map).forEach(id => {
    const host = document.getElementById(id);
    const asset = map[id];
    if (host && asset && asset.dataUrl) {
      host.innerHTML = '<img src="' + asset.dataUrl + '" alt="" style="width:100%;display:block;">';
    }
  });
}

function showResults(demo, geminiText, api, apiError) {
  const panel = document.getElementById('resultsPanel');
  panel.classList.remove('hidden');
  panel.classList.add('fadein');
  document.getElementById('resultTaskTag').textContent = (api && api.task) || demo.task;
  const body = document.getElementById('resultsBody');
  demo.render(body);
  attachCapturedToResults();
  const conf = (api && api.confidence != null) ? api.confidence : demo.confidence;
  const fill = body.querySelector('.conf-fill');
  const val = body.querySelector('.conf-val');
  if (fill) fill.style.width = conf + '%';
  if (val) val.textContent = conf + '%';
  if (api && api.answer) {
    let p = null;
    body.querySelectorAll('.answer-box').forEach(box => {
      const lbl = box.querySelector('.lbl2');
      if (lbl && /ANSWER|AI ANSWER|RESULT/i.test(lbl.textContent || '')) p = box.querySelector('p');
    });
    if (p) p.textContent = api.answer;
    else {
      const box = document.createElement('div');
      box.className = 'answer-box';
      box.style.marginTop = '14px';
      box.innerHTML = '<div class="lbl2">ANSWER</div><p></p>';
      box.querySelector('p').textContent = api.answer;
      body.insertBefore(box, body.firstChild);
    }
  }
  if (apiError) {
    const box = document.createElement('div');
    box.className = 'answer-box';
    box.style.marginTop = '14px';
    box.style.borderLeftColor = 'var(--red)';
    box.innerHTML = '<div class="lbl2">API ERROR</div><p></p>';
    box.querySelector('p').textContent = apiError + ' â€” showing local demo visuals. Check that the FastAPI backend is running and VITE_API_URL is set (' + SATQUERY_API_BASE + ').';
    body.insertBefore(box, body.firstChild);
  }
  if (geminiText) {
    const box = document.createElement('div');
    box.className = 'answer-box';
    box.style.marginTop = '14px';
    box.innerHTML = '<div class="lbl2">GEMINI (LIVE)</div><p></p>';
    box.querySelector('p').textContent = geminiText;
    body.appendChild(box);
  }
  setTimeout(() => panel.scrollIntoView({ behavior: 'smooth', block: 'nearest' }), 300);
}

/* ===================== BI-TEMPORAL CHANGE ANALYSIS ===================== */

let btBeforeFile = null;
let btAfterFile = null;
let btBeforeDataUrl = null;
let btAfterDataUrl = null;
// currentAnalysisMode declared at top of file

const BT_CHANGE_API = getSatqueryApiBase() + '/api/analyze/change';
const FUSION_API = getSatqueryApiBase() + '/api/analyze/fusion';

function switchAnalysisMode(mode) {
  currentAnalysisMode = mode;
  const tabSingle = document.getElementById('tabSingleImage');
  const tabBT = document.getElementById('tabBiTemporal');
  const tabFusion = document.getElementById('tabFusion');
  const singleContent = document.getElementById('singleImageContent');
  const btContent = document.getElementById('biTemporalContent');
  const fusionContent = document.getElementById('fusionContent');

  // Hide/show the "CAPTURE AS IMAGE 2" button based on mode
  const captureImg2Btn = document.querySelector('#captureBar .btn-ghost[onclick*="img2"]');
  if (captureImg2Btn) captureImg2Btn.style.display = (mode === 'single') ? 'none' : '';

  // Reset tab active classes
  if (tabSingle) tabSingle.classList.remove('active');
  if (tabBT) tabBT.classList.remove('active');
  if (tabFusion) tabFusion.classList.remove('active');

  // Hide all mode containers
  if (singleContent) singleContent.classList.add('hidden');
  if (btContent) btContent.classList.add('hidden');
  if (fusionContent) fusionContent.classList.add('hidden');

  if (mode === 'bitemporal') {
    if (tabBT) tabBT.classList.add('active');
    if (btContent) btContent.classList.remove('hidden');
  } else if (mode === 'fusion') {
    if (tabFusion) tabFusion.classList.add('active');
    if (fusionContent) fusionContent.classList.remove('hidden');
  } else {
    if (tabSingle) tabSingle.classList.add('active');
    if (singleContent) singleContent.classList.remove('hidden');
    // Rebuild upload cards for single-image mode (one slot only)
    buildUploadCards('default');
    recountUploaded();
    // Re-init map if needed
    if (sceneMap) { ensureMapSize(); }
    else { setTimeout(() => { initSceneMap(); ensureMapSize(); }, 60); }
  }
}


function btTriggerUpload(which) {
  const input = document.getElementById(which === 'before' ? 'btFileBefore' : 'btFileAfter');
  if (input) input.click();
}

function btHandleUpload(which, input) {
  if (!input.files || !input.files[0]) return;
  const file = input.files[0];

  // Validate
  const mime = detectUploadMime(file);
  if (!mime) {
    btShowError('Unsupported image format. Use PNG, JPEG, or TIFF.');
    return;
  }
  if (file.size > MAX_UPLOAD_BYTES) {
    btShowError('Image too large (max 12 MB).');
    return;
  }

  const reader = new FileReader();
  reader.onload = function () {
    const dataUrl = reader.result;
    if (which === 'before') {
      btBeforeFile = file;
      btBeforeDataUrl = dataUrl;
      btRenderSlot('before', dataUrl, file.name);
    } else {
      btAfterFile = file;
      btAfterDataUrl = dataUrl;
      btRenderSlot('after', dataUrl, file.name);
    }
    btUpdateAnalyzeBtn();
    btHideError();
  };
  reader.readAsDataURL(file);
}

function btRenderSlot(which, dataUrl, filename) {
  const slot = document.getElementById(which === 'before' ? 'btSlotBefore' : 'btSlotAfter');
  if (!slot) return;
  const label = which === 'before' ? 'BEFORE IMAGE' : 'AFTER IMAGE';
  slot.classList.add('filled');
  slot.innerHTML =
    '<div class="bt-tag">' + label + '</div>' +
    '<img src="' + dataUrl + '" alt="' + label + '">' +
    '<div class="bt-ready">READY</div>' +
    '<input class="bt-file" type="file" accept="image/png,image/jpeg,image/tiff,.png,.jpg,.jpeg,.tif,.tiff" onchange="btHandleUpload(\'' + which + '\', this)">';
}

function btUpdateAnalyzeBtn() {
  const btn = document.getElementById('btAnalyzeBtn');
  if (btn) btn.disabled = !(btBeforeFile && btAfterFile);
}

function btSetQuery(q) {
  const el = document.getElementById('btQueryInput');
  if (el) el.value = q;
}

function btShowError(msg) {
  const el = document.getElementById('btError');
  if (el) { el.textContent = msg; el.classList.remove('hidden'); }
}

function btHideError() {
  const el = document.getElementById('btError');
  if (el) el.classList.add('hidden');
}

// ---- Progress Stepper ----

const BT_STEPS = [
  'Input validation',
  'Image preprocessing',
  'Image alignment',
  'Change detection',
  'Region extraction',
  'Generating change map',
  'Generating change overlay',
  'AI interpretation',
  'Assembling results'
];

function btShowProgress() {
  const el = document.getElementById('btProgress');
  if (!el) return;
  el.classList.remove('hidden');
  el.innerHTML = BT_STEPS.map((s, i) =>
    '<div class="bt-progress-step pending" id="btStep' + i + '">' +
    '<div class="bt-step-icon"></div>' +
    '<span>' + s + '</span>' +
    '</div>'
  ).join('');
}

function btSetStepState(index, state) {
  // state: 'pending', 'active', 'done'
  const el = document.getElementById('btStep' + index);
  if (!el) return;
  el.classList.remove('pending', 'active', 'done');
  el.classList.add(state);
  const icon = el.querySelector('.bt-step-icon');
  if (icon) {
    if (state === 'done') icon.textContent = 'âœ“';
    else if (state === 'active') icon.textContent = 'â—';
    else icon.textContent = '';
  }
}

function btAnimateProgress() {
  return new Promise(resolve => {
    let i = 0;
    function tick() {
      if (i > 0) btSetStepState(i - 1, 'done');
      if (i < BT_STEPS.length) {
        btSetStepState(i, 'active');
        i++;
        setTimeout(tick, 350);
      } else {
        resolve();
      }
    }
    tick();
  });
}

function btFinishAllSteps() {
  for (let i = 0; i < BT_STEPS.length; i++) {
    btSetStepState(i, 'done');
  }
}

// ---- Main Analysis Function ----

async function runBiTemporalAnalysis() {
  if (!btBeforeFile || !btAfterFile) {
    btShowError('Please upload both a BEFORE and AFTER image.');
    return;
  }

  const query = (document.getElementById('btQueryInput').value || '').trim() ||
    'What changed between these images?';

  // UI state
  const btn = document.getElementById('btAnalyzeBtn');
  btn.disabled = true;
  btn.textContent = 'ANALYZINGâ€¦';
  btHideError();
  document.getElementById('btResults').classList.add('hidden');

  // Show and animate progress
  btShowProgress();
  const progressDone = btAnimateProgress();

  try {
    // Build form data
    const fd = new FormData();
    fd.append('query', query);
    fd.append('before_image', btBeforeFile, btBeforeFile.name || 'before.png');
    fd.append('after_image', btAfterFile, btAfterFile.name || 'after.png');

    // Call API
    const resp = await fetch(BT_CHANGE_API, { method: 'POST', body: fd });

    // Wait for progress animation to finish
    await progressDone;
    btFinishAllSteps();

    if (!resp.ok) {
      let detail = 'Analysis failed (HTTP ' + resp.status + ')';
      try {
        const errData = await resp.json();
        if (errData && errData.detail) detail = errData.detail;
      } catch (e) { /* ignore parse error */ }
      btShowError(detail);
      btn.disabled = false;
      btn.textContent = 'ANALYZE CHANGES';
      return;
    }

    const data = await resp.json();
    btRenderResults(data);

  } catch (e) {
    await progressDone;
    btFinishAllSteps();
    console.error('Bi-temporal analysis error:', e);
    btShowError(
      'Could not reach the backend at ' + BT_CHANGE_API +
      '. Make sure the FastAPI server is running. Error: ' + (e.message || e)
    );
  }

  btn.disabled = false;
  btn.textContent = 'ANALYZE CHANGES';
}

// ---- Render Results ----

function btRenderResults(data) {
  const container = document.getElementById('btResults');
  container.classList.remove('hidden');

  // Before / After images
  const beforeImg = document.getElementById('btResultBefore');
  const afterImg = document.getElementById('btResultAfter');
  if (data.before_image) beforeImg.src = 'data:image/png;base64,' + data.before_image;
  else if (btBeforeDataUrl) beforeImg.src = btBeforeDataUrl;
  if (data.after_image) afterImg.src = 'data:image/png;base64,' + data.after_image;
  else if (btAfterDataUrl) afterImg.src = btAfterDataUrl;

  // Change Map
  const cmImg = document.getElementById('btResultChangeMap');
  if (data.change_map) cmImg.src = 'data:image/png;base64,' + data.change_map;

  // Change Overlay
  const ovImg = document.getElementById('btResultOverlay');
  if (data.change_overlay) ovImg.src = 'data:image/png;base64,' + data.change_overlay;

  // AI Explanation
  const expl = document.getElementById('btExplanation');
  expl.textContent = data.summary || 'No explanation available.';

  // Change Summary Table
  const tbody = document.getElementById('btSummaryBody');
  if (data.changes && data.changes.length > 0) {
    tbody.innerHTML = data.changes.map(c =>
      '<tr>' +
      '<td>R' + c.region_id + '</td>' +
      '<td>' + c.location_description + '</td>' +
      '<td>' + (c.area_fraction * 100).toFixed(2) + '%</td>' +
      '<td>' + (c.mean_magnitude * 100).toFixed(0) + '%</td>' +
      '<td>' + (c.confidence * 100).toFixed(0) + '%</td>' +
      '</tr>'
    ).join('');
  } else {
    tbody.innerHTML = '<tr><td colspan="5" style="text-align:center;color:var(--tx3);">No significant change regions detected</td></tr>';
  }

  // Stats cards
  const statsRow = document.getElementById('btStatsRow');
  const confPct = typeof data.overall_confidence === 'number'
    ? (data.overall_confidence < 1 ? (data.overall_confidence * 100).toFixed(0) : data.overall_confidence.toFixed(0))
    : 'â€”';
  statsRow.innerHTML =
    '<div class="bt-stat-card">' +
    '  <div class="bt-stat-value">' + (data.changed_area_percentage || 0).toFixed(1) + '%</div>' +
    '  <div class="bt-stat-label">CHANGED AREA</div>' +
    '</div>' +
    '<div class="bt-stat-card">' +
    '  <div class="bt-stat-value">' + confPct + '%</div>' +
    '  <div class="bt-stat-label">CONFIDENCE</div>' +
    '</div>' +
    '<div class="bt-stat-card">' +
    '  <div class="bt-stat-value">' + (data.changes ? data.changes.length : 0) + '</div>' +
    '  <div class="bt-stat-label">CHANGE REGIONS</div>' +
    '</div>' +
    '<div class="bt-stat-card">' +
    '  <div class="bt-stat-value">' + (data.simulated ? 'Rule-Based' : 'LLaVA') + '</div>' +
    '  <div class="bt-stat-label">AI MODEL</div>' +
    '</div>';

  // Execution Trace
  const traceEl = document.getElementById('btTrace');
  if (data.execution_trace && data.execution_trace.length > 0) {
    traceEl.innerHTML = data.execution_trace.map(step =>
      '<div class="bt-trace-step">' +
      '<span class="bt-trace-icon">âœ“</span>' +
      '<span>' + step + '</span>' +
      '</div>'
    ).join('');
  } else {
    traceEl.innerHTML = '<div class="bt-trace-step"><span class="bt-trace-icon">â€”</span><span>No trace available</span></div>';
  }

  // Add to history
  historyLog.push({
    query: data.query || 'Bi-temporal change analysis',
    task: 'Bi-Temporal Change Analysis',
    confidence: confPct,
    lat: selectedScene.lat,
    lon: selectedScene.lon,
    sceneId: getSceneId(),
    time: new Date().toLocaleTimeString()
  });

  // Scroll to results
  setTimeout(() => container.scrollIntoView({ behavior: 'smooth', block: 'start' }), 200);
}

// ---- Reset ----

function btReset() {
  btBeforeFile = null;
  btAfterFile = null;
  btBeforeDataUrl = null;
  btAfterDataUrl = null;

  // Reset upload slots
  const slotBefore = document.getElementById('btSlotBefore');
  const slotAfter = document.getElementById('btSlotAfter');
  if (slotBefore) {
    slotBefore.classList.remove('filled');
    slotBefore.innerHTML =
      '<div class="bt-slot-icon">â‡ª</div>' +
      '<div class="bt-slot-label">BEFORE IMAGE</div>' +
      '<div class="bt-slot-sub">Earlier / before observation</div>' +
      '<input class="bt-file" type="file" id="btFileBefore" accept="image/png,image/jpeg,image/tiff,.png,.jpg,.jpeg,.tif,.tiff" onchange="btHandleUpload(\'before\', this)">';
  }
  if (slotAfter) {
    slotAfter.classList.remove('filled');
    slotAfter.innerHTML =
      '<div class="bt-slot-icon">â‡ª</div>' +
      '<div class="bt-slot-label">AFTER IMAGE</div>' +
      '<div class="bt-slot-sub">Later / after observation</div>' +
      '<input class="bt-file" type="file" id="btFileAfter" accept="image/png,image/jpeg,image/tiff,.png,.jpg,.jpeg,.tif,.tiff" onchange="btHandleUpload(\'after\', this)">';
  }

  // Reset query
  const qi = document.getElementById('btQueryInput');
  if (qi) qi.value = '';

  // Reset button
  btUpdateAnalyzeBtn();

  // Hide results and progress
  document.getElementById('btResults').classList.add('hidden');
  document.getElementById('btProgress').classList.add('hidden');
  btHideError();

  // Scroll to top
  window.scrollTo({ top: 0, behavior: 'smooth' });
}


/* ===================== OPTICAL + SAR FUSION ===================== */

let fusionOptFile = null;
let fusionSarFile = null;
let fusionOptDataUrl = null;
let fusionSarDataUrl = null;

function fusionTriggerUpload(which) {
  const input = document.getElementById(which === 'opt' ? 'fusionFileOpt' : 'fusionFileSar');
  if (input) input.click();
}

function fusionHandleUpload(which, input) {
  if (!input.files || !input.files[0]) return;
  const file = input.files[0];

  const mime = detectUploadMime(file);
  if (!mime) {
    fusionShowError('Unsupported image format. Use PNG, JPEG, or TIFF.');
    return;
  }
  if (file.size > MAX_UPLOAD_BYTES) {
    fusionShowError('Image too large (max 12 MB).');
    return;
  }

  const reader = new FileReader();
  reader.onload = function () {
    const dataUrl = reader.result;
    if (which === 'opt') {
      fusionOptFile = file;
      fusionOptDataUrl = dataUrl;
      fusionRenderSlot('opt', dataUrl, file.name);
    } else {
      fusionSarFile = file;
      fusionSarDataUrl = dataUrl;
      fusionRenderSlot('sar', dataUrl, file.name);
    }
    fusionUpdateAnalyzeBtn();
    fusionHideError();
  };
  reader.readAsDataURL(file);
}

function fusionRenderSlot(which, dataUrl, filename) {
  const slot = document.getElementById(which === 'opt' ? 'fusionSlotOpt' : 'fusionSlotSar');
  if (!slot) return;
  const label = which === 'opt' ? 'OPTICAL IMAGE' : 'SAR RADAR IMAGE';
  slot.classList.add('filled');
  slot.innerHTML =
    '<div class="fusion-tag">' + label + '</div>' +
    '<img src="' + dataUrl + '" alt="' + label + '">' +
    '<div class="fusion-ready">READY</div>' +
    '<input class="fusion-file" type="file" accept="image/png,image/jpeg,image/tiff,.png,.jpg,.jpeg,.tif,.tiff" onchange="fusionHandleUpload(\'' + which + '\', this)">';
}

function fusionUpdateAnalyzeBtn() {
  const btn = document.getElementById('fusionAnalyzeBtn');
  if (btn) btn.disabled = !(fusionOptFile && fusionSarFile);
}

function fusionSetQuery(q) {
  const el = document.getElementById('fusionQueryInput');
  if (el) el.value = q;
}

function fusionShowError(msg) {
  const el = document.getElementById('fusionError');
  if (el) { el.textContent = msg; el.classList.remove('hidden'); }
}

function fusionHideError() {
  const el = document.getElementById('fusionError');
  if (el) el.classList.add('hidden');
}

const FUSION_STEPS = [
  'Input validation & modality verification',
  'Optical radiometric preprocessing',
  'SAR despeckling & dB calibration',
  'Spatial alignment & grid matching',
  'Optical spectral & edge extraction',
  'SAR backscatter feature extraction',
  'Cross-modal fusion & agreement mapping',
  'Generating visual composite products',
  'Query-aware multimodal reasoning',
  'Assembling results'
];

function fusionShowProgress() {
  const el = document.getElementById('fusionProgress');
  if (!el) return;
  el.classList.remove('hidden');
  el.innerHTML = FUSION_STEPS.map((s, i) =>
    '<div class="fusion-progress-step pending" id="fusionStep' + i + '">' +
    '<div class="fusion-step-icon"></div>' +
    '<span>' + s + '</span>' +
    '</div>'
  ).join('');
}

function fusionSetStepState(index, state) {
  const el = document.getElementById('fusionStep' + index);
  if (!el) return;
  el.classList.remove('pending', 'active', 'done');
  el.classList.add(state);
  const icon = el.querySelector('.fusion-step-icon');
  if (icon) {
    if (state === 'done') icon.textContent = 'âœ“';
    else if (state === 'active') icon.textContent = 'â—';
    else icon.textContent = '';
  }
}

function fusionAnimateProgress() {
  return new Promise(resolve => {
    let i = 0;
    function tick() {
      if (i > 0) fusionSetStepState(i - 1, 'done');
      if (i < FUSION_STEPS.length) {
        fusionSetStepState(i, 'active');
        i++;
        setTimeout(tick, 320);
      } else {
        resolve();
      }
    }
    tick();
  });
}

function fusionFinishAllSteps() {
  for (let i = 0; i < FUSION_STEPS.length; i++) {
    fusionSetStepState(i, 'done');
  }
}

async function runFusionAnalysis() {
  if (!fusionOptFile || !fusionSarFile) {
    fusionShowError('Please upload both an OPTICAL and SAR radar image.');
    return;
  }

  const query = (document.getElementById('fusionQueryInput').value || '').trim() ||
    'Analyze this scene using fused optical and SAR satellite imagery.';
  const methodSelect = document.getElementById('fusionMethodSelect');
  const fusionMethod = (methodSelect && methodSelect.value) || 'composite';

  const btn = document.getElementById('fusionAnalyzeBtn');
  btn.disabled = true;
  btn.textContent = 'FUSING IMAGERYâ€¦';
  fusionHideError();
  document.getElementById('fusionResults').classList.add('hidden');

  fusionShowProgress();
  const progressDone = fusionAnimateProgress();

  try {
    const fd = new FormData();
    fd.append('query', query);
    fd.append('optical_image', fusionOptFile, fusionOptFile.name || 'optical.png');
    fd.append('sar_image', fusionSarFile, fusionSarFile.name || 'sar.png');
    fd.append('fusion_method', fusionMethod);

    const resp = await fetch(FUSION_API, { method: 'POST', body: fd });

    await progressDone;
    fusionFinishAllSteps();

    if (!resp.ok) {
      let detail = 'Fusion analysis failed (HTTP ' + resp.status + ')';
      try {
        const errData = await resp.json();
        if (errData && errData.detail) detail = errData.detail;
      } catch (e) { /* ignore */ }
      fusionShowError(detail);
      btn.disabled = false;
      btn.textContent = 'RUN FUSION ANALYSIS';
      return;
    }

    const data = await resp.json();
    fusionRenderResults(data);

  } catch (e) {
    await progressDone;
    fusionFinishAllSteps();
    console.error('Fusion analysis error:', e);
    fusionShowError(
      'Could not reach backend at ' + FUSION_API +
      '. Make sure FastAPI server is running. Error: ' + (e.message || e)
    );
  }

  btn.disabled = false;
  btn.textContent = 'RUN FUSION ANALYSIS';
}

function fusionRenderResults(data) {
  const container = document.getElementById('fusionResults');
  container.classList.remove('hidden');

  // Modality Triplet
  const optImg = document.getElementById('fusionResultOpt');
  const sarImg = document.getElementById('fusionResultSar');
  const compImg = document.getElementById('fusionResultComposite');

  if (data.optical_image) optImg.src = 'data:image/png;base64,' + data.optical_image;
  else if (fusionOptDataUrl) optImg.src = fusionOptDataUrl;

  if (data.sar_image) sarImg.src = 'data:image/png;base64,' + data.sar_image;
  else if (fusionSarDataUrl) sarImg.src = fusionSarDataUrl;

  if (data.fusion_visualization) compImg.src = 'data:image/png;base64,' + data.fusion_visualization;

  // Evidence Breakdown
  const optEv = document.getElementById('fusionOpticalEvidence');
  const sarEv = document.getElementById('fusionSarEvidence');
  const fusedEv = document.getElementById('fusionFusedInterp');

  if (optEv) optEv.textContent = data.optical_evidence || 'Optical evidence not available.';
  if (sarEv) sarEv.textContent = data.sar_evidence || 'SAR radar evidence not available.';
  if (fusedEv) fusedEv.textContent = data.fused_interpretation || data.summary || 'Fused interpretation not available.';

  // Agreement Map
  const mapImg = document.getElementById('fusionResultEvidenceMap');
  const mapSec = document.getElementById('fusionEvidenceMapSec');
  if (data.evidence_map && mapImg) {
    mapImg.src = 'data:image/png;base64,' + data.evidence_map;
    if (mapSec) mapSec.classList.remove('hidden');
  } else if (mapSec) {
    mapSec.classList.add('hidden');
  }

  // Cross-modal features table
  const tbody = document.getElementById('fusionFeatureBody');
  if (data.features && data.features.length > 0) {
    tbody.innerHTML = data.features.map(f =>
      '<tr>' +
      '<td><strong>F' + f.id + '</strong></td>' +
      '<td><span class="fusion-cat-badge ' + (f.category.includes('Water') ? 'water' : 'urban') + '">' + f.category + '</span></td>' +
      '<td>' + f.location_description + '</td>' +
      '<td>' + f.optical_characteristics + '</td>' +
      '<td>' + f.sar_characteristics + '</td>' +
      '<td>' + (f.agreement_score * 100).toFixed(0) + '%</td>' +
      '<td>' + (f.confidence * 100).toFixed(0) + '%</td>' +
      '</tr>'
    ).join('');
  } else {
    tbody.innerHTML = '<tr><td colspan="7" style="text-align:center;color:var(--tx3);padding:14px;">No discrete cross-modal feature regions isolated (continuous surface response)</td></tr>';
  }

  // Stats cards
  const statsRow = document.getElementById('fusionStatsRow');
  const confPct = typeof data.confidence === 'number' ? data.confidence : 'â€”';
  const agreePct = typeof data.modality_agreement_percentage === 'number' ? data.modality_agreement_percentage.toFixed(1) : 'â€”';
  const alignTag = data.alignment_method === 'geo_referenced' ? 'GeoTIFF CRS' : 'Dimension-Matched';

  statsRow.innerHTML =
    '<div class="fusion-stat-card">' +
    '  <div class="fusion-stat-value">' + confPct + '%</div>' +
    '  <div class="fusion-stat-label">CONFIDENCE SCORE</div>' +
    '</div>' +
    '<div class="fusion-stat-card">' +
    '  <div class="fusion-stat-value">' + agreePct + '%</div>' +
    '  <div class="fusion-stat-label">SENSOR AGREEMENT</div>' +
    '</div>' +
    '<div class="fusion-stat-card">' +
    '  <div class="fusion-stat-value">' + (data.features ? data.features.length : 0) + '</div>' +
    '  <div class="fusion-stat-label">CROSS-MODAL FEATURES</div>' +
    '</div>' +
    '<div class="fusion-stat-card">' +
    '  <div class="fusion-stat-value" style="font-size:16px;margin-top:6px;">' + alignTag + '</div>' +
    '  <div class="fusion-stat-label">ALIGNMENT METHOD</div>' +
    '</div>' +
    '<div class="fusion-stat-card">' +
    '  <div class="fusion-stat-value" style="font-size:16px;margin-top:6px;">' + (data.simulated ? 'Rule-Based Engine' : 'LLaVA VLM') + '</div>' +
    '  <div class="fusion-stat-label">REASONING ENGINE</div>' +
    '</div>';

  // Diagnostics Panel
  const diagGrid = document.getElementById('fusionDiagGrid');

  if (diagGrid && data.diagnostics) {
    const d = data.diagnostics;
    const optStats = d.optical_stats || {};
    const sarStats = d.sar_stats || {};
    const fusionM = d.fusion_metrics || {};

    const optNdwi = optStats.ndwi_stats || {};
    const b08Str = optStats.b08_stats ? ('B08 (NIR) Mean/Med: ' + optStats.b08_stats.mean + ' / ' + optStats.b08_stats.median) : null;
    const optLines = [
      'B02 (Blue) Mean/Med: ' + (optStats.b02_stats ? optStats.b02_stats.mean + ' / ' + optStats.b02_stats.median : optStats.b02_blue_mean || 'â€”'),
      'B03 (Green) Mean/Med: ' + (optStats.b03_stats ? optStats.b03_stats.mean + ' / ' + optStats.b03_stats.median : optStats.b03_green_mean || 'â€”'),
      'B04 (Red) Mean/Med: ' + (optStats.b04_stats ? optStats.b04_stats.mean + ' / ' + optStats.b04_stats.median : optStats.b04_red_mean || 'â€”'),
      b08Str,
      'NDWI Mean / Median / Std: ' + (optNdwi.mean !== undefined ? optNdwi.mean + ' / ' + optNdwi.median + ' / ' + optNdwi.std : 'â€”'),
      'NDWI Threshold: >= ' + (d.ndwi_threshold !== undefined ? d.ndwi_threshold : 'â€”'),
      'Optical Water Candidate: ' + (d.optical_water_candidate_pct !== undefined ? d.optical_water_candidate_pct + '%' : 'â€”'),
      d.optical_crs ? 'CRS: ' + d.optical_crs : null
    ].filter(Boolean);

    const sarVv = sarStats.vv_stats || {};
    const sarVh = sarStats.vh_stats || {};
    const sarLines = [
      'VV Mean / Median / Std: ' + (sarVv.mean !== undefined ? sarVv.mean + ' / ' + sarVv.median + ' / ' + sarVv.std + ' dB' : 'â€”'),
      'VV Percentiles [p5, p25, p75, p95]: [' + (sarVv.p5 || 'â€”') + ', ' + (sarVv.p25 || 'â€”') + ', ' + (sarVv.p75 || 'â€”') + ', ' + (sarVv.p95 || 'â€”') + '] dB',
      sarVh.mean !== undefined ? 'VH Mean / Median: ' + sarVh.mean + ' / ' + sarVh.median + ' dB' : null,
      'Water Backscatter Thresh: <= ' + (d.water_backscatter_threshold_db !== undefined ? d.water_backscatter_threshold_db + ' dB' : 'â€”'),
      'SAR Water Candidate: ' + (d.sar_water_candidate_pct !== undefined ? d.sar_water_candidate_pct + '%' : 'â€”'),
      'Double-Bounce Structures: ' + (d.sar_double_bounce_pct !== undefined ? d.sar_double_bounce_pct + '%' : 'â€”'),
      d.sar_crs ? 'CRS: ' + d.sar_crs : null
    ].filter(Boolean);

    const alignLines = [
      'Alignment: ' + (d.alignment_method === 'geo_referenced' ? 'GeoTIFF Reprojection' : 'Dimension-Matched'),
      'Geo Metadata Used: ' + (d.geo_metadata_used ? 'Yes (True CRS/Transform)' : 'No (Fallback)'),
      d.common_crs ? 'Common CRS: ' + d.common_crs : null,
      d.common_grid_resolution ? 'Grid Resolution: ' + d.common_grid_resolution : null,
      'Optical+SAR Consensus Water: ' + (d.optical_sar_consensus_water_pct !== undefined ? d.optical_sar_consensus_water_pct + '%' : 'â€”'),
      'Optical-Only / SAR-Only Water: ' + (d.optical_only_water_pct !== undefined ? d.optical_only_water_pct + '%' : 'â€”') + ' / ' + (d.sar_only_water_pct !== undefined ? d.sar_only_water_pct + '%' : 'â€”'),
      'Modality Agreement: ' + (d.modality_agreement_pct !== undefined ? d.modality_agreement_pct + '%' : 'â€”'),
      'Inundation IoU (Jaccard): ' + (d.inundation_iou_pct !== undefined ? d.inundation_iou_pct + '%' : 'â€”'),
      'Baseline Status: ' + (d.permanent_water_handling_status || 'single_date_candidate')
    ].filter(Boolean);

    diagGrid.innerHTML =
      '<div class="fusion-diag-card">' +
      '  <div class="fusion-diag-head">OPTICAL MULTISPECTRAL STATS</div>' +
      '  <div class="fusion-diag-body">' + optLines.map(l => '<div class="fusion-diag-row">' + l + '</div>').join('') + '</div>' +
      '</div>' +
      '<div class="fusion-diag-card">' +
      '  <div class="fusion-diag-head">SAR RADAR POLARIMETRY STATS</div>' +
      '  <div class="fusion-diag-body">' + sarLines.map(l => '<div class="fusion-diag-row">' + l + '</div>').join('') + '</div>' +
      '</div>' +
      '<div class="fusion-diag-card">' +
      '  <div class="fusion-diag-head">GEOSPATIAL & CONSENSUS METRICS</div>' +
      '  <div class="fusion-diag-body">' + alignLines.map(l => '<div class="fusion-diag-row">' + l + '</div>').join('') + '</div>' +
      '</div>';
  }


  // Execution Trace
  const traceEl = document.getElementById('fusionTrace');

  if (data.execution_trace && data.execution_trace.length > 0) {
    traceEl.innerHTML = data.execution_trace.map(step =>
      '<div class="fusion-trace-step">' +
      '<span class="fusion-trace-icon">âœ“</span>' +
      '<span>' + step + '</span>' +
      '</div>'
    ).join('');
  } else {
    traceEl.innerHTML = '<div class="fusion-trace-step"><span class="fusion-trace-icon">â€”</span><span>No trace available</span></div>';
  }

  // Add to History
  historyLog.push({
    query: data.query || 'Optical + SAR multimodal fusion',
    task: 'Optical + SAR Fusion',
    confidence: confPct,
    lat: selectedScene.lat,
    lon: selectedScene.lon,
    sceneId: getSceneId(),
    time: new Date().toLocaleTimeString()
  });

  setTimeout(() => container.scrollIntoView({ behavior: 'smooth', block: 'start' }), 200);
}

function fusionReset() {
  fusionOptFile = null;
  fusionSarFile = null;
  fusionOptDataUrl = null;
  fusionSarDataUrl = null;

  const slotOpt = document.getElementById('fusionSlotOpt');
  const slotSar = document.getElementById('fusionSlotSar');
  if (slotOpt) {
    slotOpt.classList.remove('filled');
    slotOpt.innerHTML =
      '<div class="fusion-slot-icon">â‡ª</div>' +
      '<div class="fusion-slot-label">OPTICAL IMAGE</div>' +
      '<div class="fusion-slot-sub">Multispectral / RGB satellite scene (GeoTIFF / PNG / JPEG)</div>' +
      '<input class="fusion-file" type="file" id="fusionFileOpt" accept="image/png,image/jpeg,image/tiff,.png,.jpg,.jpeg,.tif,.tiff" onchange="fusionHandleUpload(\'opt\', this)">';
  }
  if (slotSar) {
    slotSar.classList.remove('filled');
    slotSar.innerHTML =
      '<div class="fusion-slot-icon">â‡ª</div>' +
      '<div class="fusion-slot-label">SAR RADAR IMAGE</div>' +
      '<div class="fusion-slot-sub">Radar backscatter / amplitude scene (GeoTIFF / PNG / JPEG)</div>' +
      '<input class="fusion-file" type="file" id="fusionFileSar" accept="image/png,image/jpeg,image/tiff,.png,.jpg,.jpeg,.tif,.tiff" onchange="fusionHandleUpload(\'sar\', this)">';
  }

  const qi = document.getElementById('fusionQueryInput');
  if (qi) qi.value = '';

  fusionUpdateAnalyzeBtn();
  document.getElementById('fusionResults').classList.add('hidden');
  document.getElementById('fusionProgress').classList.add('hidden');
  fusionHideError();

  window.scrollTo({ top: 0, behavior: 'smooth' });
}


/* ===================== WINDOW EXPORTS ===================== */
Object.assign(window, {
  placeMarker, searchLocation, requestBrowserLocation, setGeoMode, startAoiDraw, addAoiVertex, finishAoi, useAoi, clearAoi,
  analyzeThisScene, useLocation, runAnalysis, newAnalysis, switchView, showLanding, launchApp, placeComparePoint, runComparison,
  setPreviewMode, saveCoordFormat, setComparePick, centerScene, hudZoom, pickDemo, loadDemo, restoreHistory, coordToSeed, getSceneId,
  captureMapAsImage, clearCapture, ingestUploadedImage, collectGeminiImages,
  // Bi-temporal
  switchAnalysisMode, btTriggerUpload, btHandleUpload, btSetQuery, runBiTemporalAnalysis, btReset,
  // Optical + SAR Fusion
  fusionTriggerUpload, fusionHandleUpload, fusionSetQuery, runFusionAnalysis, fusionReset
});

