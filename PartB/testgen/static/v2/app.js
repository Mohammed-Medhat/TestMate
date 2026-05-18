/* TestMate v2 — App Logic */

// ── State ──
let discoveredFiles = [];
let selectedFiles = new Set();
let activeFile = null;
let eventSource = null;

// ── API Calls ──
async function discoverFiles() {
    const url = document.getElementById('repo-url').value.trim();
    if (!url) return;
    const branch = document.getElementById('repo-branch').value.trim();
    const btn = document.getElementById('btn-discover');
    btn.disabled = true; btn.innerHTML = '<span class="spinner"></span> Scanning...';
    try {
        const res = await fetch('/api/discover', {
            method: 'POST', headers: {'Content-Type':'application/json'},
            body: JSON.stringify({url, branch})
        });
        const data = await res.json();
        if (data.error) { addLog('❌ ' + data.error, 'error'); return; }
        discoveredFiles = data.files || [];
        renderFileTree(discoveredFiles, data.repo_name);
        addLog(`✅ Found ${discoveredFiles.length} testable files in ${data.repo_name}`, 'success');
    } catch(e) { addLog('❌ ' + e.message, 'error'); }
    finally { btn.disabled = false; btn.textContent = '🔍 Discover Files'; }
}

function renderFileTree(files, repoName) {
    const el = document.getElementById('file-list');
    if (!files.length) { el.innerHTML = '<div class="ft-empty">No testable files found</div>'; return; }
    selectedFiles.clear();
    let html = `<div class="ft-item" style="font-weight:600;color:var(--text)"><span class="ft-icon">📁</span><span class="ft-name">${repoName || 'repo'}/</span></div>`;
    files.forEach((f, i) => {
        selectedFiles.add(i);
        const parts = f.path.split('/');
        const name = parts[parts.length - 1];
        const dir = parts.length > 1 ? parts.slice(0, -1).join('/') + '/' : '';
        html += `<div class="ft-item selected" data-idx="${i}" onclick="toggleFile(${i}, this)">
            <input type="checkbox" checked onclick="event.stopPropagation();toggleFile(${i},this.parentElement)">
            <span class="ft-icon">📄</span>
            <span class="ft-name" title="${f.path}">${dir ? '<span style="color:var(--text3)">' + dir + '</span>' : ''}${name}</span>
            <span class="ft-meta">${f.functions}fn ${f.classes}cls</span>
        </div>`;
    });
    el.innerHTML = html;
    document.getElementById('fe-count').textContent = `${files.length}/${files.length}`;
}

function toggleFile(idx, el) {
    if (selectedFiles.has(idx)) { selectedFiles.delete(idx); el.classList.remove('selected'); el.querySelector('input').checked = false; }
    else { selectedFiles.add(idx); el.classList.add('selected'); el.querySelector('input').checked = true; }
    document.getElementById('fe-count').textContent = `${selectedFiles.size}/${discoveredFiles.length}`;
}

function toggleSelectAll() {
    const all = selectedFiles.size === discoveredFiles.length;
    document.querySelectorAll('.ft-item[data-idx]').forEach(el => {
        const idx = parseInt(el.dataset.idx);
        if (all) { selectedFiles.delete(idx); el.classList.remove('selected'); el.querySelector('input').checked = false; }
        else { selectedFiles.add(idx); el.classList.add('selected'); el.querySelector('input').checked = true; }
    });
    document.getElementById('fe-count').textContent = `${selectedFiles.size}/${discoveredFiles.length}`;
}

// ── Run Evaluation ──
async function startEvaluation() {
    if (!selectedFiles.size) { addLog('⚠️ Select files first', 'warning'); return; }
    const files = [...selectedFiles].map(i => discoveredFiles[i]);
    const url = document.getElementById('repo-url').value.trim();
    const branch = document.getElementById('repo-branch').value.trim();
    const btn = document.getElementById('btn-run');
    btn.disabled = true;
    setStatus('running', 'Running...');
    try {
        const res = await fetch('/api/run', {
            method: 'POST', headers: {'Content-Type':'application/json'},
            body: JSON.stringify({
                url, branch, files,
                docker: document.getElementById('docker-mode')?.checked || false,
                deep_scan: document.getElementById('deep-scan-mode')?.checked || false,
                max_retries: parseInt(document.getElementById('max-retries')?.value || '3'),
                hitl: document.getElementById('hitl-mode')?.checked || false,
                intense: document.getElementById('intense-mode')?.checked || false,
            })
        });
        const data = await res.json();
        if (data.error) { addLog('❌ ' + data.error, 'error'); btn.disabled = false; setStatus('idle','Idle'); return; }
        startSSE();
    } catch(e) { addLog('❌ ' + e.message, 'error'); btn.disabled = false; setStatus('idle','Idle'); }
}

// ── SSE Streaming ──
function startSSE() {
    if (eventSource) eventSource.close();
    eventSource = new EventSource('/api/stream');
    eventSource.onmessage = (e) => {
        try {
            const msg = JSON.parse(e.data);
            handleSSE(msg);
        } catch(err) {}
    };
    eventSource.onerror = () => { eventSource.close(); eventSource = null; };
}

function handleSSE(msg) {
    switch(msg.type) {
        case 'log': addLog(msg.message, msg.level); addTrace(msg.message, msg.level); break;
        case 'progress': updateProgress(msg.current, msg.total, msg.file); break;
        case 'pipeline_stage': setPipelineStage(msg.stage); break;
        case 'ai_status': updateAIStatus(msg.status, msg.detail, msg.target); break;
        case 'code_stream': streamCode(msg.code, msg.filename, msg.target, msg.is_retry); break;
        case 'code_clear': clearCodeStream(); break;
        case 'result': showResults(msg.data); break;
        case 'review_request': showReview(msg); break;
        case 'complete': onComplete(); break;
    }
}

// ── Terminal / Logs ──
function addLog(msg, level) {
    const body = document.getElementById('terminal-body');
    const div = document.createElement('div');
    div.className = 'log-' + (level || 'info');
    div.textContent = msg;
    body.appendChild(div);
    body.scrollTop = body.scrollHeight;
}

function addTrace(msg, level) {
    if (!msg || msg.startsWith('─') || msg.startsWith('═') || !msg.trim()) return;
    const el = document.getElementById('agent-trace');
    if (el.querySelector('.ft-empty')) el.innerHTML = '';
    const div = document.createElement('div');
    div.style.cssText = 'font-size:.62rem;padding:2px 4px;color:var(--text3);border-left:2px solid ' +
        (level === 'success' ? 'var(--green)' : level === 'error' ? 'var(--red)' : 'var(--cyan)');
    div.textContent = msg.substring(0, 80);
    el.appendChild(div);
    el.scrollTop = el.scrollHeight;
    while (el.children.length > 40) el.removeChild(el.firstChild);
}

function updateProgress(current, total, file) {
    const pct = Math.round((current / total) * 100);
    document.getElementById('term-progress').style.width = pct + '%';
    document.getElementById('term-file').textContent = file || '';
    document.getElementById('term-count').textContent = `${current}/${total}`;
}

// ── Pipeline ──
function setPipelineStage(stage) {
    const stages = ['scan','model','gen','audit','eval'];
    const idx = stages.indexOf(stage);
    document.querySelectorAll('.step').forEach((el, i) => {
        el.classList.remove('done','active','pending');
        if (i < idx) el.classList.add('done');
        else if (i === idx) el.classList.add('active');
        else el.classList.add('pending');
    });
}

// ── AI Status ──
function updateAIStatus(status, detail, target) {
    const bar = document.getElementById('ai-bar');
    bar.classList.add('active');
    const icons = {loading:'⏳',analyzing:'🔍',thinking:'🧠',writing:'✍️',validating:'✅',retrying:'🔄'};
    document.getElementById('ai-text').textContent = (icons[status]||'🤖') + ' ' + (status||'');
    document.getElementById('ai-detail').textContent = detail || '';
    document.getElementById('ai-target').textContent = target || '';
}

// ── Status ──
function setStatus(state, text) {
    const dot = document.getElementById('status-dot');
    dot.className = 'status-dot ' + state;
    document.getElementById('status-text').textContent = text;
}

function onComplete() {
    setStatus('done', 'Complete');
    document.getElementById('btn-run').disabled = false;
    document.getElementById('ai-bar').classList.remove('active');
    if (eventSource) { eventSource.close(); eventSource = null; }
}

// ── Tabs ──
function switchTab(tab) {
    document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.tab === tab));
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    const panel = document.getElementById('tab-' + tab);
    if (panel) panel.classList.add('active');
    document.getElementById('tab-welcome').classList.remove('active');
    document.getElementById('cov-bar').style.display = tab === 'heatmap' ? 'flex' : 'none';
}

// ── Results ──
function showResults(data) {
    if (!data || data.error) return;
    // Update rings
    updateRing('ring-cov', 'ring-cov-val', data.line_coverage || 0, '%');
    updateRing('ring-pass', 'ring-pass-val', data.pass_rate || 0, '%');
    updateRing('ring-mut', 'ring-mut-val', data.mutation_score || 0, '%');
    updateRing('ring-bugs', 'ring-bugs-val', data.bugs_found || 0, '', true);
    // Coverage
    document.getElementById('cov-covered').textContent = data.passed_tests || 0;
    document.getElementById('cov-uncovered').textContent = data.failed_tests || 0;
    document.getElementById('cov-total').textContent = data.total_tests || 0;
    document.getElementById('cov-pct').textContent = (data.line_coverage || 0).toFixed(1) + '%';
    // Bugs table
    if (data.bug_reports && data.bug_reports.length) {
        renderBugs(data.bug_reports);
    }
    // Load coverage heatmap for first file
    if (data.per_source_files && data.per_source_files.length) {
        loadCoverage(data.per_source_files[0].source_file);
        loadTestCode(data.per_source_files[0].source_file);
    }
}

function updateRing(ringId, valId, value, suffix, isCount) {
    const circ = 87.96;
    const pct = isCount ? Math.min(value * 10, 100) : Math.min(value, 100);
    const offset = circ - (circ * pct / 100);
    document.getElementById(ringId).style.strokeDashoffset = offset;
    document.getElementById(valId).textContent = (isCount ? value : value.toFixed(1)) + (suffix || '');
    document.getElementById(valId).style.color = pct > 0 ? '' : '';
}

function renderBugs(bugs) {
    const tbody = document.getElementById('bugs-tbody');
    document.getElementById('bugs-empty').style.display = 'none';
    tbody.innerHTML = bugs.map((b, i) => `<tr>
        <td style="font-weight:700;color:var(--red)">ZD-${String(i+1).padStart(3,'0')}</td>
        <td><span class="sev-badge ${b.confidence || 'critical'}">${(b.confidence || 'Critical').charAt(0).toUpperCase() + (b.confidence||'critical').slice(1)}</span></td>
        <td><strong>${b.bug_type || 'Unknown'}</strong><br><span style="font-size:.68rem;color:var(--text3)">${(b.description || '').substring(0, 80)}...</span></td>
        <td class="fname">${b.source_file || '—'}</td>
        <td><button class="btn-details">View Details</button></td>
    </tr>`).join('');
}

// ── Coverage Heatmap ──
async function loadCoverage(filename) {
    try {
        const res = await fetch('/api/coverage/' + filename);
        const data = await res.json();
        if (data.error || !data.source) return;
        renderVSCode('heatmap-viewer', data.source, data.covered_lines, data.uncovered_lines);
        switchTab('heatmap');
    } catch(e) {}
}

// ── Test Code ──
async function loadTestCode(filename) {
    try {
        const res = await fetch('/api/testcode/' + filename);
        const data = await res.json();
        if (data.error || !data.source) return;
        document.getElementById('tc-filename').textContent = data.filename || 'test_*.py';
        renderVSCode('testcode-viewer', data.source);
    } catch(e) {}
}

function copyTestCode() {
    const viewer = document.getElementById('testcode-viewer');
    const lines = viewer.querySelectorAll('.line-code');
    const text = Array.from(lines).map(l => l.textContent).join('\n');
    navigator.clipboard.writeText(text);
    const btn = document.getElementById('btn-copy');
    btn.textContent = '✅ Copied!';
    setTimeout(() => btn.textContent = '📋 Copy', 2000);
}

// ── VS Code Renderer ──
function renderVSCode(containerId, source, covered, uncovered) {
    const el = document.getElementById(containerId);
    const lines = source.split('\n');
    covered = covered || []; uncovered = uncovered || [];
    el.innerHTML = lines.map((line, i) => {
        const num = i + 1;
        const isCov = covered.includes(num);
        const isUncov = uncovered.includes(num);
        const cls = isCov ? 'covered' : isUncov ? 'uncovered' : '';
        const ind = isCov ? 'covered' : isUncov ? 'uncovered' : '';
        const esc = highlightPy(line.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'));
        return `<div class="code-line ${cls}"><span class="line-num">${num}</span><span class="line-ind ${ind}"></span><span class="line-code">${esc}</span></div>`;
    }).join('');
}

// ── Python Syntax Highlighting (VS Code style) ──
function highlightPy(line) {
    // Comments
    const cmtIdx = findCommentStart(line);
    let code = line, comment = '';
    if (cmtIdx >= 0) { code = line.substring(0, cmtIdx); comment = `<span class="s-cmt">${line.substring(cmtIdx)}</span>`; }
    // Decorators
    code = code.replace(/^(\s*)(@\w+)/g, '$1<span class="s-dec">$2</span>');
    // Strings
    code = code.replace(/("""[\s\S]*?"""|'''[\s\S]*?'''|"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')/g, '<span class="s-str">$1</span>');
    // Keywords
    const kws = 'def|class|return|import|from|if|elif|else|for|while|try|except|finally|with|as|raise|pass|break|continue|and|or|not|in|is|None|True|False|self|async|await|yield|lambda|assert|global|nonlocal';
    code = code.replace(new RegExp('\\b(' + kws + ')\\b', 'g'), '<span class="s-kw">$1</span>');
    // Functions
    code = code.replace(/\b(\w+)(\s*\()/g, '<span class="s-fn">$1</span>$2');
    // Numbers
    code = code.replace(/\b(\d+\.?\d*)\b/g, '<span class="s-num">$1</span>');
    return code + comment;
}

function findCommentStart(line) {
    let inStr = false, strCh = '';
    for (let i = 0; i < line.length; i++) {
        const ch = line[i];
        if (inStr) { if (ch === strCh && line[i-1] !== '\\') inStr = false; }
        else if (ch === '"' || ch === "'") { inStr = true; strCh = ch; }
        else if (ch === '#') return i;
    }
    return -1;
}

// ── Code Streaming ──
let _streamInterval = null;
function streamCode(code, filename, target, isRetry) {
    switchTab('testcode');
    if (filename) document.getElementById('tc-filename').textContent = filename;
    const viewer = document.getElementById('testcode-viewer');
    if (isRetry) viewer.innerHTML = '';
    const lines = code.split('\n');
    let idx = 0;
    const base = viewer.querySelectorAll('.code-line').length;
    if (_streamInterval) clearInterval(_streamInterval);
    _streamInterval = setInterval(() => {
        if (idx >= lines.length) { clearInterval(_streamInterval); _streamInterval = null; viewer.querySelectorAll('.streaming-cursor').forEach(c=>c.remove()); return; }
        viewer.querySelectorAll('.streaming-cursor').forEach(c=>c.remove());
        const num = base + idx + 1;
        const esc = highlightPy(lines[idx].replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'));
        const div = document.createElement('div');
        div.className = 'code-line streaming';
        div.innerHTML = `<span class="line-num">${num}</span><span class="line-ind"></span><span class="line-code">${esc}<span class="streaming-cursor"></span></span>`;
        viewer.appendChild(div);
        viewer.scrollTop = viewer.scrollHeight;
        idx++;
    }, 50);
}

function clearCodeStream() {
    if (_streamInterval) { clearInterval(_streamInterval); _streamInterval = null; }
    document.getElementById('testcode-viewer').innerHTML = '';
}

// ── HITL Review ──
function showReview(msg) {
    document.getElementById('review-overlay').classList.add('active');
    document.getElementById('review-test-count').textContent = msg.num_tests || 0;
    document.getElementById('review-target-file').textContent = msg.target_file || '—';
    document.getElementById('review-code-editor').value = msg.test_code || '';
}

async function submitReview(decision) {
    document.getElementById('review-overlay').classList.remove('active');
    const editedCode = document.getElementById('review-code-editor').value;
    try {
        await fetch('/api/review', {
            method:'POST', headers:{'Content-Type':'application/json'},
            body: JSON.stringify({decision, edited_code: editedCode})
        });
    } catch(e) {}
}

// Escape key closes review
document.addEventListener('keydown', e => { if (e.key === 'Escape') document.getElementById('review-overlay').classList.remove('active'); });

// ── Config Modal ──
function toggleConfig() {
    document.getElementById('config-overlay').classList.toggle('active');
}

// ── Theme ──
function toggleTheme() {
    const html = document.documentElement;
    const current = html.getAttribute('data-theme') || 'dark';
    const next = current === 'dark' ? 'light' : 'dark';
    html.setAttribute('data-theme', next);
    localStorage.setItem('testmate-theme', next);
    document.getElementById('theme-toggle').textContent = next === 'dark' ? '☀️' : '🌙';
}

// ── Landing ──
function scrollToDashboard() {
    document.getElementById('dashboard-section').scrollIntoView({behavior:'smooth'});
}

// ── Init ──
(function() {
    try {
        const t = localStorage.getItem('testmate-theme');
        if (t) document.documentElement.setAttribute('data-theme', t);
    } catch(e) {}
})();
