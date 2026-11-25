// Driver Drowsiness Alert: runs fully in the browser.
// Face landmarks: MediaPipe Face Landmarker (vendored in /vendor so the app works without a CDN).
import { FaceLandmarker, FilesetResolver } from "./vendor/mediapipe/vision_bundle.mjs";

const $ = (id) => document.getElementById(id);
const video = $("video"), overlay = $("overlay"), ctx = overlay.getContext("2d");
const screenEl = $("screen"), banner = $("banner"), pill = $("statusPill");

// Landmark indices (MediaPipe 478-point mesh). Six points per eye for the eye aspect ratio:
// [outer corner, upper 1, upper 2, inner corner, lower 2, lower 1]
const EYE_R = [33, 160, 158, 133, 153, 144];
const EYE_L = [362, 385, 387, 263, 373, 380];
const RING_R = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246];
const RING_L = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466];
const MOUTH = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191];

const settings = {
  closedSec: 1.5, sens: 0.70, tilt: 20, vol: 0.8, voice: true, mesh: true,
};
const bind = (id, out, fmt, key, scale = 1) => {
  const el = $(id), o = $(out);
  const upd = () => { settings[key] = +el.value * scale; o.textContent = fmt(+el.value); };
  el.addEventListener("input", upd); upd();
};
bind("closedSec", "closedSecOut", (v) => v.toFixed(1), "closedSec");
bind("sens", "sensOut", (v) => v + "%", "sens", 0.01);
bind("tilt", "tiltOut", (v) => v + "°", "tilt");
bind("vol", "volOut", (v) => v + "%", "vol", 0.01);
$("voiceOn").addEventListener("change", (e) => (settings.voice = e.target.checked));
$("meshOn").addEventListener("change", (e) => (settings.mesh = e.target.checked));

let landmarker = null, stream = null, running = false, lastVideoTime = -1;
let audio = null, siren = null, silencedUntil = 0;

const S = {};
function resetState() {
  Object.assign(S, {
    startedAt: performance.now(),
    calib: { until: performance.now() + 3000, ear: [], pitch: [], roll: [] },
    baseEar: null, basePitch: 0, baseRoll: 0,
    closedSince: null, blinkTimes: [], samples: [],
    headOffSince: null, noFaceSince: null,
    yawnSince: null, yawnTimes: [], yawns: 0,
    alarms: 0, alarmReason: null, warnUntil: 0, lastSpoken: {}, lastTiredWarn: 0,
    alertness: 100,
  });
  $("yawnVal").textContent = "0"; $("alarmVal").textContent = "0";
}

// ---------- geometry ----------
const dist = (a, b, w, h) => Math.hypot((a.x - b.x) * w, (a.y - b.y) * h);
function ear(lm, idx, w, h) {
  const p = idx.map((i) => lm[i]);
  return (dist(p[1], p[5], w, h) + dist(p[2], p[4], w, h)) / (2 * dist(p[0], p[3], w, h));
}
function headPose(matrix) {
  // 4x4 column-major facial transformation matrix -> pitch / yaw / roll in degrees
  const m = matrix.data, r = (row, col) => m[col * 4 + row];
  const deg = 180 / Math.PI;
  return {
    pitch: Math.atan2(r(2, 1), r(2, 2)) * deg,
    yaw: Math.asin(Math.max(-1, Math.min(1, -r(2, 0)))) * deg,
    roll: Math.atan2(r(1, 0), r(0, 0)) * deg,
  };
}
const median = (a) => { const s = [...a].sort((x, y) => x - y); return s.length ? s[s.length >> 1] : 0; };
const angleDiff = (a, b) => ((a - b + 540) % 360) - 180;

// ---------- sound + voice ----------
function initAudio() {
  if (!audio) audio = new (window.AudioContext || window.webkitAudioContext)();
  if (audio.state === "suspended") audio.resume();
}
function startSiren() {
  if (siren) return;
  initAudio();
  const osc = audio.createOscillator(), gain = audio.createGain();
  osc.type = "square";
  gain.gain.value = 0.25 * settings.vol;
  osc.connect(gain).connect(audio.destination);
  const t0 = audio.currentTime;
  for (let i = 0; i < 400; i++) osc.frequency.setValueAtTime(i % 2 ? 1320 : 880, t0 + i * 0.25);
  osc.start();
  siren = { osc, gain };
}
function stopSiren() {
  if (!siren) return;
  try { siren.osc.stop(); } catch {}
  siren = null;
}
function chime() {
  initAudio();
  const osc = audio.createOscillator(), gain = audio.createGain(), t = audio.currentTime;
  osc.type = "sine"; osc.frequency.setValueAtTime(660, t); osc.frequency.setValueAtTime(990, t + 0.18);
  gain.gain.setValueAtTime(0.0001, t);
  gain.gain.exponentialRampToValueAtTime(0.3 * settings.vol + 0.001, t + 0.03);
  gain.gain.exponentialRampToValueAtTime(0.0001, t + 0.5);
  osc.connect(gain).connect(audio.destination); osc.start(t); osc.stop(t + 0.55);
}
function say(text, key = text, cooldown = 6000) {
  if (!settings.voice || !("speechSynthesis" in window)) return;
  const now = performance.now();
  if (S.lastSpoken && now - (S.lastSpoken[key] || -1e9) < cooldown) return;
  if (S.lastSpoken) S.lastSpoken[key] = now;
  speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text);
  u.rate = 1.05; u.volume = Math.max(0.2, settings.vol);
  speechSynthesis.speak(u);
}

// ---------- UI helpers ----------
function log(msg, cls = "") {
  const li = document.createElement("li");
  const t = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  li.innerHTML = `<time>${t}</time><span class="${cls}"></span>`;
  li.querySelector("span").textContent = msg;
  const ol = $("log");
  ol.querySelector(".muted")?.remove();
  ol.prepend(li);
  while (ol.children.length > 50) ol.lastChild.remove();
}
function showBanner(text, kind = "") {
  banner.hidden = !text; banner.textContent = text || ""; banner.className = "banner " + kind;
}
function setStatus(text, kind = "") { pill.hidden = false; pill.className = "status-pill " + kind; $("statusText").textContent = text; }

function raiseAlarm(reason, speech) {
  const now = performance.now();
  if (S.alarmReason !== reason) {
    if (!S.alarmReason) { S.alarms++; $("alarmVal").textContent = S.alarms; }
    S.alarmReason = reason;
    log("ALARM: " + reason, "bad");
  }
  showBanner(reason, "");
  screenEl.classList.add("alarm");
  $("silenceBtn").hidden = false;
  if (now > silencedUntil) startSiren();
  say(speech, "alarm:" + reason, 5000);
}
function clearAlarm() {
  if (!S.alarmReason) return;
  log("Alarm cleared: you look alert again");
  S.alarmReason = null;
  stopSiren();
  screenEl.classList.remove("alarm");
  $("silenceBtn").hidden = true;
  showBanner(performance.now() < S.warnUntil ? banner.textContent : "");
}
function warn(text, speech, key) {
  const now = performance.now();
  if (now - (S.lastSpoken["warn:" + key] || -1e9) < 30000) { if (!S.alarmReason) showBanner(text, "warn"); S.warnUntil = now + 3000; return; }
  log(text, "warn");
  chime();
  say(speech, "warn:" + key, 30000);
  S.warnUntil = now + 4000;
  if (!S.alarmReason) showBanner(text, "warn");
}

// ---------- drawing ----------
function draw(lm, w, h, closed) {
  overlay.width = w; overlay.height = h;
  ctx.clearRect(0, 0, w, h);
  if (!lm || !settings.mesh) return;
  let x0 = 1, y0 = 1, x1 = 0, y1 = 0;
  for (const p of lm) { x0 = Math.min(x0, p.x); y0 = Math.min(y0, p.y); x1 = Math.max(x1, p.x); y1 = Math.max(y1, p.y); }
  const color = S.alarmReason ? "#ff3b3b" : closed ? "#ff8a1f" : "#2fd27a";
  ctx.lineWidth = Math.max(2, w / 320);
  ctx.strokeStyle = color; ctx.setLineDash([10, 8]);
  ctx.strokeRect(x0 * w - 10, y0 * h - 10, (x1 - x0) * w + 20, (y1 - y0) * h + 20);
  ctx.setLineDash([]);
  const ring = (idx, c) => {
    ctx.beginPath();
    idx.forEach((i, k) => (k ? ctx.lineTo(lm[i].x * w, lm[i].y * h) : ctx.moveTo(lm[i].x * w, lm[i].y * h)));
    ctx.closePath(); ctx.strokeStyle = c; ctx.stroke();
  };
  ring(RING_L, color); ring(RING_R, color); ring(MOUTH, "#f5b301aa");
  ctx.fillStyle = "#ffffffcc";
  for (const i of [...EYE_L, ...EYE_R]) { ctx.beginPath(); ctx.arc(lm[i].x * w, lm[i].y * h, ctx.lineWidth, 0, 7); ctx.fill(); }
}

// ---------- main loop ----------
function loop() {
  if (!running) return;
  if (video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const now = performance.now();
    const res = landmarker.detectForVideo(video, now);
    process(res, now);
  }
  requestAnimationFrame(loop);
}

function process(res, now) {
  const w = video.videoWidth, h = video.videoHeight;
  const lm = res.faceLandmarks?.[0];
  const calibrating = now < S.calib.until || S.baseEar === null;

  if (!lm) {
    draw(null, w, h);
    S.closedSince = null; S.headOffSince = null;
    S.noFaceSince ??= now;
    setStatus("No face found", "warn");
    $("stateText").textContent = "Looking for your face";
    if (calibrating) { S.calib.until = Math.max(S.calib.until, now + 1500); showBanner("Face the camera to calibrate", "info"); }
    else if (now - S.noFaceSince > 4000) warn("Eyes on the road", "Please keep your eyes on the road.", "noface");
    updateStats(null, now);
    return;
  }
  if (S.noFaceSince) S.noFaceSince = null;

  const e = (ear(lm, EYE_L, w, h) + ear(lm, EYE_R, w, h)) / 2;
  const pose = res.facialTransformationMatrixes?.[0] ? headPose(res.facialTransformationMatrixes[0]) : { pitch: 0, yaw: 0, roll: 0 };
  const mar = dist(lm[13], lm[14], w, h) / dist(lm[78], lm[308], w, h);

  if (calibrating) {
    S.calib.ear.push(e); S.calib.pitch.push(pose.pitch); S.calib.roll.push(pose.roll);
    const left = Math.max(0, (S.calib.until - now) / 1000);
    showBanner(`Calibrating… keep your eyes open (${left.toFixed(1)}s)`, "info");
    setStatus("Calibrating", "idle");
    $("stateText").textContent = "Learning your open eyes";
    if (now >= S.calib.until && S.calib.ear.length > 10) {
      S.baseEar = Math.max(0.18, median(S.calib.ear));
      S.basePitch = median(S.calib.pitch); S.baseRoll = median(S.calib.roll);
      showBanner("");
      log(`Calibrated. Normal eye openness ${S.baseEar.toFixed(3)}`);
      say("Monitoring started. Drive safely.", "start", 1000);
    }
    draw(lm, w, h, false);
    updateStats({ e, pose, mar }, now);
    return;
  }

  // Eyes
  const closed = e < S.baseEar * settings.sens;
  if (closed) S.closedSince ??= now;
  else if (S.closedSince) {
    const d = now - S.closedSince;
    if (d > 60 && d < 500) S.blinkTimes.push(now);
    S.closedSince = null;
  }
  S.samples.push({ t: now, closed });
  while (S.samples.length && now - S.samples[0].t > 60000) S.samples.shift();
  while (S.blinkTimes.length && now - S.blinkTimes[0] > 60000) S.blinkTimes.shift();

  // Head
  const dPitch = angleDiff(pose.pitch, S.basePitch), dRoll = angleDiff(pose.roll, S.baseRoll);
  const headOff = Math.abs(dPitch) > settings.tilt || Math.abs(dRoll) > settings.tilt + 5;
  if (headOff) S.headOffSince ??= now; else S.headOffSince = null;

  // Yawn: mouth wide open for over 1.2 s
  if (mar > 0.6) S.yawnSince ??= now;
  else {
    if (S.yawnSince && now - S.yawnSince > 1200) {
      S.yawns++; S.yawnTimes.push(now); $("yawnVal").textContent = S.yawns; log("Yawn detected", "warn");
    }
    S.yawnSince = null;
  }
  while (S.yawnTimes.length && now - S.yawnTimes[0] > 300000) S.yawnTimes.shift();

  const perclos = S.samples.filter((s) => s.closed).length / Math.max(1, S.samples.length);

  // Decide
  if (S.closedSince && now - S.closedSince >= settings.closedSec * 1000) {
    raiseAlarm("Wake up! Eyes closed", "Wake up! Your eyes are closing. Please pull over and rest.");
  } else if (S.headOffSince && now - S.headOffSince >= 2000) {
    raiseAlarm(Math.abs(dPitch) > settings.tilt ? "Head dropping" : "Head tilting", "Wake up! Your head is dropping. Lift your head and focus on the road.");
  } else {
    clearAlarm();
    const elapsed = now - S.startedAt;
    if (S.yawnTimes.length >= 3) warn("You look tired", "You have yawned several times. Consider taking a break soon.", "yawn");
    else if (elapsed > 30000 && perclos > 0.2) warn("You look tired", "Your eyes are closing often. Consider taking a break soon.", "perclos");
  }
  if (!S.alarmReason && now > S.warnUntil && !banner.hidden && !banner.classList.contains("info")) showBanner("");

  const kind = S.alarmReason ? "bad" : closed || headOff ? "warn" : "";
  setStatus(S.alarmReason ? "ALARM" : closed ? "Eyes closed" : headOff ? "Head moved" : "Eyes open", kind);
  draw(lm, w, h, closed);
  updateStats({ e, pose, mar, dPitch, dRoll, perclos }, now);
}

function updateStats(d, now) {
  const secs = Math.floor((now - S.startedAt) / 1000);
  $("timeVal").textContent = `${Math.floor(secs / 60)}:${String(secs % 60).padStart(2, "0")}`;
  if (!d) return;
  const open = S.baseEar ? Math.min(1.3, d.e / S.baseEar) : null;
  $("earVal").textContent = open === null ? d.e.toFixed(2) : Math.round(Math.min(1, open) * 100) + "%";
  $("earBar").style.width = Math.min(100, (open ?? d.e / 0.35) * 77) + "%";
  $("earBar").style.background = open !== null && open < settings.sens ? "var(--red)" : "var(--green)";
  $("rollVal").textContent = (d.dRoll ?? 0).toFixed(0) + "°";
  $("pitchVal").textContent = (d.dPitch ?? 0).toFixed(0) + "°";
  if (S.baseEar === null) return;
  const mins = Math.min(1, (now - S.startedAt) / 60000);
  $("bpmVal").textContent = mins > 0.15 ? Math.round(S.blinkTimes.length / mins) : "…";
  $("perclosVal").textContent = Math.round((d.perclos ?? 0) * 100) + "%";

  let target = 100 - Math.min(55, (d.perclos ?? 0) * 220) - Math.min(25, S.yawnTimes.length * 8);
  if (S.closedSince) target -= 25;
  if (S.headOffSince) target -= 20;
  if (S.alarmReason) target = Math.min(target, 15);
  S.alertness += (Math.max(0, target) - S.alertness) * 0.08;
  const a = Math.round(S.alertness);
  const g = $("gauge");
  g.style.setProperty("--level", a);
  g.className = "gauge " + (a < 40 ? "low" : a < 70 ? "mid" : "");
  $("alertness").textContent = a;
  $("stateText").textContent = S.alarmReason ? "Danger: wake up!" : a < 40 ? "Very drowsy" : a < 70 ? "Getting tired" : "Alert";
}

// ---------- start / stop ----------
async function loadModel() {
  if (landmarker) return;
  const files = await FilesetResolver.forVisionTasks(new URL("./vendor/mediapipe/wasm", import.meta.url).href);
  const opts = (delegate) => ({
    baseOptions: { modelAssetPath: new URL("./models/face_landmarker.task", import.meta.url).href, delegate },
    runningMode: "VIDEO", numFaces: 1,
    outputFacialTransformationMatrixes: true, outputFaceBlendshapes: false,
  });
  try { landmarker = await FaceLandmarker.createFromOptions(files, opts("GPU")); }
  catch { landmarker = await FaceLandmarker.createFromOptions(files, opts("CPU")); }
}

async function start() {
  const btn = $("startBtn");
  btn.disabled = true; btn.textContent = "Loading face tracker…";
  initAudio();
  try {
    if (!navigator.mediaDevices?.getUserMedia) throw new Error("This browser can't open the camera. Try Chrome, Edge, Safari or Firefox over https.");
    const [s] = await Promise.all([
      navigator.mediaDevices.getUserMedia({ video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } }, audio: false }),
      loadModel(),
    ]);
    stream = s;
    video.srcObject = stream;
    await video.play();
  } catch (err) {
    btn.disabled = false; btn.textContent = "Try again";
    const msg = err.name === "NotAllowedError" ? "Camera permission was blocked. Allow the camera in your browser's address bar and try again."
      : err.name === "NotFoundError" ? "No camera was found on this device." : (err.message || String(err));
    document.querySelector(".intro .fine").textContent = msg;
    log("Could not start: " + msg, "bad");
    return;
  }
  $("intro").hidden = true;
  resetState();
  running = true; lastVideoTime = -1;
  $("stopBtn").disabled = false; $("recalBtn").disabled = false;
  log("Camera on. Calibrating");
  requestAnimationFrame(loop);
}

function stop() {
  running = false;
  stopSiren(); speechSynthesis?.cancel?.();
  stream?.getTracks().forEach((t) => t.stop()); stream = null;
  screenEl.classList.remove("alarm"); showBanner(""); pill.hidden = true;
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  $("intro").hidden = false;
  $("startBtn").disabled = false; $("startBtn").textContent = "Start again";
  $("stopBtn").disabled = true; $("recalBtn").disabled = true; $("silenceBtn").hidden = true;
  $("stateText").textContent = "Stopped";
  log("Monitoring stopped");
}

$("startBtn").addEventListener("click", start);
$("stopBtn").addEventListener("click", stop);
$("recalBtn").addEventListener("click", () => {
  S.baseEar = null; S.calib = { until: performance.now() + 3000, ear: [], pitch: [], roll: [] };
  clearAlarm(); log("Recalibrating");
});
$("silenceBtn").addEventListener("click", () => { silencedUntil = performance.now() + 10000; stopSiren(); log("Alarm silenced for 10 s"); });
$("testBtn").addEventListener("click", () => {
  initAudio(); startSiren();
  const prev = S.lastSpoken; S.lastSpoken = null; say("Wake up! This is a test of the drowsiness alarm."); S.lastSpoken = prev;
  screenEl.classList.add("alarm");
  setTimeout(() => { stopSiren(); if (!S.alarmReason) screenEl.classList.remove("alarm"); }, 2500);
});
document.addEventListener("visibilitychange", () => {
  if (document.hidden && running) log("Tab hidden: keep this page visible so monitoring continues", "warn");
});
resetState();
