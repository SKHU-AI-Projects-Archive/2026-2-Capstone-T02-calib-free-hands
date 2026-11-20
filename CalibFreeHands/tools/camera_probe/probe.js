"use strict";

const [role, key] = location.pathname.split('/').slice(1);
const workspace = document.querySelector('#workspace');
const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const cameraControls = {
  focusMode: {label: '초점', values: ['focusDistance']},
  exposureMode: {label: '노출', values: ['exposureTime', 'iso']},
  whiteBalanceMode: {label: '화이트밸런스', values: ['colorTemperature']},
};

async function signal(camera, body) {
  const suffix = camera ? `?camera=${camera}` : '';
  const options = body ? {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body),
  } : {};
  const response = await fetch(`/signal/${key}${suffix}`, options);
  if (!response.ok) throw new Error(`연결 요청 오류 (${response.status})`);
  return response.json();
}

async function localDescription(peer, description) {
  await peer.setLocalDescription(description);
  const deadline = performance.now() + 7000;
  while (peer.iceGatheringState !== 'complete') {
    if (performance.now() >= deadline) throw new Error('네트워크 주소 수집 시간이 초과되었습니다.');
    await wait(50);
  }
  return peer.localDescription.toJSON();
}

function makePeer() {
  return new RTCPeerConnection({iceServers: []});
}

function connectionLabel(state) {
  return {
    new: '연결 대기', connecting: '연결 중', connected: '연결됨',
    disconnected: '연결 끊김', failed: '연결 실패', closed: '연결 종료',
  }[state] ?? state;
}

function showPhone() {
  workspace.innerHTML = `
    <button id="start">카메라 연결</button><button id="lock" disabled>잠금 다시 시험</button>
    <p id="status" role="status">PC의 수신 화면을 먼저 열어 주세요.</p>
    <video muted autoplay playsinline></video>
    <details open><summary>카메라 설정</summary><pre id="settings"></pre></details>`;
  const start = document.querySelector('#start');
  const lock = document.querySelector('#lock');
  const status = document.querySelector('#status');
  const preview = document.querySelector('video');
  const diagnostic = document.querySelector('#settings');
  let stream, peer, callId;

  function stop() {
    peer?.close();
    stream?.getTracks().forEach((track) => track.stop());
    stream = undefined;
    callId = undefined;
    preview.srcObject = null;
    lock.disabled = true;
    start.textContent = '카메라 연결';
  }

  function showSettings(track, attempts) {
    const {deviceId, groupId, ...settings} = track.getSettings();
    const {deviceId: _device, groupId: _group, ...capabilities} = track.getCapabilities?.() ?? {};
    diagnostic.textContent = JSON.stringify({settings, capabilities, attempts}, null, 2);
  }

  start.onclick = async () => {
    if (stream) {
      stop();
      status.textContent = '연결 종료';
      return;
    }
    start.disabled = true;
    try {
      if (!isSecureContext || !navigator.mediaDevices?.getUserMedia) {
        throw new Error('카메라 권한을 사용하려면 신뢰된 HTTPS 주소로 여세요.');
      }
      status.textContent = '카메라 권한 확인 중';
      stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {facingMode: {ideal: 'environment'}, width: {ideal: 1920}, height: {ideal: 1080}, frameRate: {ideal: 30}},
      });
      const track = stream.getVideoTracks()[0];
      if ('contentHint' in track) track.contentHint = 'detail';
      preview.srcObject = stream;
      showSettings(track);
      peer = makePeer();
      peer.addTrack(track, stream);
      peer.onconnectionstatechange = () => { status.textContent = connectionLabel(peer.connectionState); };
      const id = crypto.randomUUID();
      const description = await localDescription(peer, await peer.createOffer());
      await signal(null, {id, description});
      status.textContent = 'PC 응답 대기';
      const deadline = performance.now() + 20000;
      for (;;) {
        const response = await signal(null);
        if (response.id === id && response.answer) {
          await peer.setRemoteDescription(response.answer);
          break;
        }
        if (response.id !== id || performance.now() >= deadline) throw new Error('PC 응답을 받지 못했습니다.');
        await wait(300);
      }
      start.textContent = '연결 끊기';
      callId = id;
      lock.disabled = false;
      lock.click();
    } catch (error) {
      stop();
      status.textContent = error.message;
    } finally {
      start.disabled = false;
    }
  };

  lock.onclick = async () => {
    const track = stream?.getVideoTracks()[0];
    const id = callId;
    if (!track || !id) return;
    lock.disabled = true;
    const capabilities = track.getCapabilities?.() ?? {};
    const current = track.getSettings();
    const attempts = {};
    // Chromium handles camera controls separately from video format constraints.
    const cameraConstraints = {};
    for (const [mode, {values}] of Object.entries(cameraControls)) {
      if (!capabilities[mode]?.includes('manual')) {
        attempts[mode] = '수동 모드 미노출';
        continue;
      }
      const patch = {[mode]: {exact: 'manual'}};
      for (const name of values) {
        const range = capabilities[name];
        const value = current[name];
        if (range && Number.isFinite(value) &&
            value >= (range.min ?? -Infinity) && value <= (range.max ?? Infinity)) {
          patch[name] = {exact: value};
        }
      }
      try {
        await track.applyConstraints({...cameraConstraints, ...patch});
        Object.assign(cameraConstraints, patch);
      } catch (error) {
        attempts[mode] = `${error.name}: ${error.constraint || error.message}`;
      }
    }
    const first = track.getSettings();
    try {
      await wait(2000);
      if (id !== callId) return;
      const latest = track.getSettings();
      const controls = {};
      for (const [mode, {values}] of Object.entries(cameraControls)) {
        if (cameraConstraints[mode]) {
          attempts[mode] = latest[mode] === 'manual'
            ? '수동 모드 읽기 확인' : '수동 모드 읽기 미확인';
        }
        const readings = values
          .filter(name => Number.isFinite(first[name]) && Number.isFinite(latest[name]))
          .map(name => ({
            name, before: current[name], first: first[name], latest: latest[name],
            inRange: !!capabilities[name] &&
              latest[name] >= (capabilities[name].min ?? -Infinity) &&
              latest[name] <= (capabilities[name].max ?? Infinity),
          }));
        controls[mode] = {
          result: attempts[mode], initialMode: current[mode],
          firstMode: first[mode], mode: latest[mode], readings,
        };
      }
      showSettings(track, attempts);
      const {width, height, frameRate} = latest;
      await signal(null, {id, diagnostics: {capture: {width, height, frameRate}, controls}});
    } catch (error) {
      if (id === callId) diagnostic.textContent += `\nPC에 시험 결과를 보내지 못했습니다. ${error.message}`;
    } finally {
      if (id === callId) lock.disabled = false;
    }
  };
  window.addEventListener('pagehide', stop);
}

function showViewer() {
  workspace.innerHTML = `
    <button data-camera="A" aria-pressed="true">카메라 A</button>
    <button data-camera="B" aria-pressed="false">카메라 B</button>
    <div class="video-area"><video id="feed-A" muted autoplay playsinline></video>
    <video id="feed-B" class="inactive-feed" muted autoplay playsinline></video></div>
    <p class="muted" id="status-A" role="status">A · 연결 대기</p>
    <pre id="diagnostics-A"></pre>
    <p class="muted" id="status-B" role="status">B · 연결 대기</p>
    <pre id="diagnostics-B"></pre>
    <p class="muted">잠금 결과는 약 2초 간격으로 읽은 브라우저 설정입니다. 실제 렌즈 동작은 영상의 장면 변화로 확인할 수 있습니다.</p>`;
  const buttons = [...document.querySelectorAll('[data-camera]')];
  for (const button of buttons) {
    button.onclick = () => {
      for (const item of buttons) {
        const active = item === button;
        item.setAttribute('aria-pressed', String(active));
        const feed = document.querySelector(`#feed-${item.dataset.camera}`);
        feed.classList.toggle('inactive-feed', !active);
        if (active && feed.srcObject) feed.play().catch(() => {});
      }
    };
    receive(button.dataset.camera);
  }
}

function renderDiagnostics(camera, video, report) {
  const output = document.querySelector(`#diagnostics-${camera}`);
  const {width, height, frameRate} = report?.capture ?? {};
  const capture = Number.isFinite(width) && Number.isFinite(height)
    ? `${width}×${height}${Number.isFinite(frameRate) ? ` · ${Math.round(frameRate)} fps` : ''}`
    : '대기';
  const received = video.srcObject && video.videoWidth && video.videoHeight
    ? `${video.videoWidth}×${video.videoHeight}` : '대기';
  const lines = [`1080p 요청 · 휴대폰 캡처 ${capture} · PC 수신 ${received}`];
  for (const [mode, {label}] of Object.entries(cameraControls)) {
    const control = report?.controls?.[mode];
    if (!control || typeof control.result !== 'string') continue;
    const reportedMode = typeof control.mode === 'string' ? control.mode : '미보고';
    const modeReadback = control.initialMode && control.initialMode !== reportedMode
      ? `${control.initialMode} → ${reportedMode}` : reportedMode;
    lines.push(`${label} · ${control.result} (${modeReadback})`);
    if (control.firstMode && control.firstMode !== control.mode) {
      lines.push(`  모드 변화 ${control.firstMode} → ${control.mode ?? '미보고'}`);
    }
    for (const reading of Array.isArray(control.readings) ? control.readings : []) {
      if (!Number.isFinite(reading?.first) || !Number.isFinite(reading.latest)) continue;
      const qualifier = !reading.inRange ? ' · 지원 범위 밖' : reading.first === reading.latest ? ' · 두 읽기 동일' : ' · 값 변동';
      const before = Number.isFinite(reading.before) ? `${reading.before} → ` : '';
      lines.push(`  ${reading.name} ${before}${reading.first} → ${reading.latest}${qualifier}`);
    }
  }
  if (received !== '대기' && !report) lines.push('휴대폰 잠금 결과 대기');
  const text = lines.join('\n');
  if (output.textContent !== text) output.textContent = text;
}

async function receive(camera) {
  const video = document.querySelector(`#feed-${camera}`);
  const status = document.querySelector(`#status-${camera}`);
  let activeId, peer;
  for (;;) {
    try {
      const session = await signal(camera);
      if (session.id && session.id !== activeId) {
        activeId = session.id;
        peer?.close();
        video.srcObject = null;
        status.textContent = `${camera} · 연결 중`;
        peer = makePeer();
        const active = peer;
        let remoteStream;
        active.onconnectionstatechange = () => {
          if (peer !== active) return;
          const state = active.connectionState;
          status.textContent = `${camera} · ${connectionLabel(state)}`;
          if (state === 'connected' && remoteStream) video.srcObject = remoteStream;
          if (['failed', 'closed', 'disconnected'].includes(state)) video.srcObject = null;
        };
        active.ontrack = ({track}) => {
          remoteStream = new MediaStream([track]);
          if (peer === active) video.srcObject = remoteStream;
        };
        await active.setRemoteDescription(session.offer);
        const description = await localDescription(active, await active.createAnswer());
        await signal(camera, {id: activeId, description});
      }
      renderDiagnostics(camera, video, session.diagnostics);
    } catch (error) {
      status.textContent = `${camera} · ${error.message}`;
    }
    await wait(350);
  }
}

if (role === 'phone') showPhone();
if (role === 'view') showViewer();
