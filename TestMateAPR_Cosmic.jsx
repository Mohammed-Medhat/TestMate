import { useState, useEffect, useRef } from "react";

const PAGES = ["home", "upload", "pipeline", "code", "tests", "attempts"];

const SUSPICIOUS = [
  { line: 12, score: 0.91, code: "    return self.items.pop()" },
  { line: 14, score: 0.74, code: "    if len(self.items) == 0:" },
  { line: 7, score: 0.52, code: "    self.items.append(item)" },
];

const BUGGY = `class BoundedStack:
    def __init__(self, max_size):
        self.items = []
        self.max_size = max_size

    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()

    def is_empty(self):
        return len(self.items) == 0`;

const FIXED = `class BoundedStack:
    def __init__(self, max_size):
        self.items = []
        self.max_size = max_size

    def push(self, item):
        if len(self.items) >= self.max_size:
            raise OverflowError("Stack is full")
        self.items.append(item)

    def pop(self):
        if self.is_empty():
            raise IndexError("Stack is empty")
        return self.items.pop()

    def is_empty(self):
        return len(self.items) == 0`;

const ATTEMPTS = [
  { n: 1, status: "fail", patch: "Removed bounds check", result: "AssertionError on test_overflow" },
  { n: 2, status: "fail", patch: "Added partial guard", result: "IndexError on test_empty_pop" },
  { n: 3, status: "success", patch: "Full guard + overflow", result: "All 5 tests passed ✅" },
];

const PIPELINE_STEPS = [
  { id: 1, label: "Run Tests", icon: "🧪" },
  { id: 2, label: "Fault Localization", icon: "🔍" },
  { id: 3, label: "Prompt Generation", icon: "📝" },
  { id: 4, label: "Model Repair", icon: "🤖" },
  { id: 5, label: "Apply Patch", icon: "🔧" },
  { id: 6, label: "Re-run Tests", icon: "✅" },
];

const TESTS = [
  { name: "test_push_basic", status: "pass" },
  { name: "test_pop_basic", status: "pass" },
  { name: "test_overflow", status: "fail", trace: "OverflowError not raised when stack full" },
  { name: "test_empty_pop", status: "fail", trace: "IndexError not raised on empty stack" },
  { name: "test_is_empty", status: "pass" },
];

export default function App() {
  const [page, setPage] = useState("home");
  const [running, setRunning] = useState(false);
  const [step, setStep] = useState(0);
  const [logs, setLogs] = useState([]);
  const [done, setDone] = useState(false);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [stars] = useState(() => Array.from({ length: 60 }, (_, i) => ({
    x: Math.random() * 100, y: Math.random() * 100,
    size: Math.random() * 2 + .5, opacity: Math.random() * .7 + .2,
    delay: Math.random() * 4
  })));
  const logRef = useRef(null);

  useEffect(() => { if (logRef.current) logRef.current.scrollTop = 9999; }, [logs]);

  const runPipeline = () => {
    setRunning(true); setStep(0); setLogs([]); setDone(false);
    setPage("pipeline");
    const msgs = [
      [0, "info", "🚀 Starting APR pipeline…"],
      [600, "warn", "🧪 Running pytest… 2 tests FAILED"],
      [1200, "info", "🔍 SBFL analysis: top suspicious line 12 (score 0.91)"],
      [1800, "info", "📝 Building repair prompt with stack trace…"],
      [2400, "info", "🤖 Calling TestMate model (attempt 1/3)…"],
      [3000, "error", "❌ Patch 1 rejected — tests still failing"],
      [3600, "info", "🤖 Calling TestMate model (attempt 2/3)…"],
      [4200, "error", "❌ Patch 2 rejected — IndexError persists"],
      [4800, "info", "🤖 Calling TestMate model (attempt 3/3)…"],
      [5400, "success", "🔧 AST patch applied — BoundedStack guards added"],
      [6000, "success", "✅ All 5 tests PASSED — Bug fixed!"],
    ];
    msgs.forEach(([t, type, m]) => {
      setTimeout(() => {
        setLogs(p => [...p, { type, m, ts: new Date().toLocaleTimeString("en-GB") }]);
        const s = Math.floor(t / 600);
        setStep(Math.min(s, 6));
        if (t === 6000) { setDone(true); setRunning(false); }
      }, t);
    });
  };

  return (
    <div style={{
      minHeight: "100vh", fontFamily: "'DM Sans', system-ui, sans-serif",
      background: "#0e0818",
      color: "#c8b8e8",
      overflow: "auto",
      position: "relative",
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;600;700;800&display=swap');
        * { box-sizing:border-box; margin:0; padding:0; }
        ::-webkit-scrollbar{width:4px;height:4px}
        ::-webkit-scrollbar-track{background:transparent}
        ::-webkit-scrollbar-thumb{background:rgba(168,85,247,.3);border-radius:2px}
        @keyframes twinkle{0%,100%{opacity:.2}50%{opacity:.9}}
        @keyframes float{0%,100%{transform:translateY(0)}50%{transform:translateY(-12px)}}
        @keyframes glow{0%,100%{box-shadow:0 0 20px rgba(168,85,247,.3)}50%{box-shadow:0 0 40px rgba(168,85,247,.6)}}
        @keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
        @keyframes slideUp{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:translateY(0)}}
        @keyframes spin{to{transform:rotate(360deg)}}
        @keyframes shimmer{0%{background-position:-200% 0}100%{background-position:200% 0}}
        .page-enter{animation:slideUp .4s ease both}
        .nav-item{cursor:pointer;padding:8px 16px;border-radius:20px;font-size:13px;font-weight:500;transition:all .2s;color:rgba(200,184,232,.6)}
        .nav-item:hover{color:#e8d8ff;background:rgba(168,85,247,.1)}
        .nav-item.active{color:#e8d8ff;background:rgba(168,85,247,.2);border:1px solid rgba(168,85,247,.3)}
        .card{background:rgba(255,255,255,.04);border:1px solid rgba(168,85,247,.15);border-radius:20px;backdrop-filter:blur(12px)}
        .card:hover{border-color:rgba(168,85,247,.35);box-shadow:0 8px 32px rgba(168,85,247,.12)}
        .btn-primary{background:linear-gradient(135deg,#9333ea,#7c3aed);border:none;border-radius:14px;color:#fff;font-family:inherit;font-weight:600;font-size:14px;cursor:pointer;padding:13px 28px;transition:all .2s;box-shadow:0 4px 20px rgba(147,51,234,.4)}
        .btn-primary:hover{transform:translateY(-2px);box-shadow:0 8px 30px rgba(147,51,234,.6)}
        .btn-outline{background:transparent;border:1px solid rgba(168,85,247,.4);border-radius:14px;color:#c8b8e8;font-family:inherit;font-weight:500;font-size:14px;cursor:pointer;padding:12px 28px;transition:all .2s}
        .btn-outline:hover{background:rgba(168,85,247,.1);border-color:rgba(168,85,247,.7)}
        .step-done{background:rgba(52,211,153,.12);border:1px solid rgba(52,211,153,.3);color:#34d399}
        .step-run{background:rgba(168,85,247,.15);border:1px solid rgba(168,85,247,.5);color:#c084fc;animation:glow 2s ease-in-out infinite}
        .step-fail{background:rgba(239,68,68,.1);border:1px solid rgba(239,68,68,.3);color:#f87171}
        .step-idle{background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.07);color:rgba(200,184,232,.3)}
        .log-info{color:#a78bfa}.log-success{color:#34d399}.log-warn{color:#fbbf24}.log-error{color:#f87171}
        .heatbar{height:8px;border-radius:4px;background:linear-gradient(90deg,rgba(239,68,68,.15),rgba(239,68,68,1));transition:width .8s ease}
        input[type=file]{display:none}
        .upload-zone{border:2px dashed rgba(168,85,247,.3);border-radius:16px;padding:32px;text-align:center;cursor:pointer;transition:all .2s;background:rgba(168,85,247,.03)}
        .upload-zone:hover{border-color:rgba(168,85,247,.7);background:rgba(168,85,247,.07)}
      `}</style>

      {/* Stars */}
      <div style={{ position: "fixed", inset: 0, zIndex: 0, pointerEvents: "none", overflow: "hidden" }}>
        {stars.map((s, i) => (
          <div key={i} style={{
            position: "absolute", left: `${s.x}%`, top: `${s.y}%`,
            width: s.size, height: s.size, borderRadius: "50%",
            background: "#fff", opacity: s.opacity,
            animation: `twinkle ${2 + s.delay}s ease-in-out infinite`,
            animationDelay: `${s.delay}s`
          }} />
        ))}
        {/* Nebula blobs */}
        <div style={{ position: "absolute", top: "-20%", left: "-10%", width: "60%", height: "60%", borderRadius: "50%", background: "radial-gradient(circle,rgba(147,51,234,.12) 0%,transparent 70%)", filter: "blur(40px)" }} />
        <div style={{ position: "absolute", bottom: "-20%", right: "-10%", width: "50%", height: "50%", borderRadius: "50%", background: "radial-gradient(circle,rgba(99,102,241,.1) 0%,transparent 70%)", filter: "blur(40px)" }} />
        <div style={{ position: "absolute", top: "40%", left: "50%", width: "30%", height: "30%", borderRadius: "50%", background: "radial-gradient(circle,rgba(236,72,153,.06) 0%,transparent 70%)", filter: "blur(30px)" }} />
      </div>

      {/* NAVBAR */}
      <nav style={{
        position: "sticky", top: 0, zIndex: 100,
        background: "rgba(14,8,24,.85)",
        backdropFilter: "blur(20px)",
        borderBottom: "1px solid rgba(168,85,247,.1)",
        padding: "12px 32px",
        display: "flex", alignItems: "center", gap: 8,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginRight: "auto" }}>
          <div style={{
            width: 34, height: 34, borderRadius: 10,
            background: "linear-gradient(135deg,#9333ea,#6366f1)",
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: 16, boxShadow: "0 0 16px rgba(147,51,234,.5)",
          }}>⚙</div>
          <div>
            <div style={{ fontSize: 15, fontWeight: 700, color: "#f0e8ff", fontFamily: "'Space Grotesk',sans-serif" }}>TestMate</div>
            <div style={{ fontSize: 9, color: "rgba(168,85,247,.6)", letterSpacing: "2px", textTransform: "uppercase" }}>Auto Program Repair</div>
          </div>
        </div>
        {[["home", "Home"], ["upload", "Upload"], ["pipeline", "Pipeline"], ["code", "Code Diff"], ["tests", "Tests"], ["attempts", "Attempts"]].map(([id, label]) => (
          <div key={id} className={`nav-item${page === id ? " active" : ""}`} onClick={() => setPage(id)}>{label}</div>
        ))}
      </nav>

      {/* PAGES */}
      <div style={{ position: "relative", zIndex: 1, maxWidth: 1100, margin: "0 auto", padding: "0 24px 60px" }}>

        {/* ── HOME ── */}
        {page === "home" && (
          <div className="page-enter">
            {/* Hero */}
            <div style={{ textAlign: "center", padding: "80px 20px 60px" }}>
              <div style={{
                display: "inline-flex", alignItems: "center", gap: 8,
                background: "rgba(168,85,247,.1)", border: "1px solid rgba(168,85,247,.25)",
                borderRadius: 20, padding: "6px 16px", fontSize: 12, color: "#c084fc",
                marginBottom: 28, letterSpacing: "1px", textTransform: "uppercase",
              }}>✦ Graduation Project 2025</div>

              <h1 style={{
                fontFamily: "'Space Grotesk',sans-serif",
                fontSize: "clamp(36px,6vw,68px)", fontWeight: 800,
                color: "#f0e8ff", lineHeight: 1.1, marginBottom: 20,
                background: "linear-gradient(135deg,#e879f9,#c084fc,#818cf8)",
                WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
              }}>
                Automated<br />Program Repair
              </h1>

              <p style={{ fontSize: 18, color: "rgba(200,184,232,.7)", maxWidth: 520, margin: "0 auto 40px", lineHeight: 1.7 }}>
                Fix Python bugs automatically using AI-powered multi-teacher distillation,
                SBFL fault localization, and smart patch generation.
              </p>

              <div style={{ display: "flex", gap: 14, justifyContent: "center", flexWrap: "wrap" }}>
                <button className="btn-primary" onClick={() => setPage("upload")}>🚀 Start Demo</button>
                <button className="btn-outline" onClick={() => setPage("pipeline")}>View Pipeline ↗</button>
              </div>
            </div>

            {/* How it works */}
            <div style={{ marginBottom: 48 }}>
              <SectionTitle>How It Works</SectionTitle>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(160px,1fr))", gap: 14, marginTop: 24 }}>
                {[
                  { icon: "🧪", n: "01", t: "Run Tests", d: "Execute pytest to detect failing test cases" },
                  { icon: "🔍", n: "02", t: "SBFL Analysis", d: "Rank suspicious lines using spectrum analysis" },
                  { icon: "🤖", n: "03", t: "AI Repair", d: "Generate patches with multi-teacher distillation" },
                  { icon: "🔧", n: "04", t: "Apply Patch", d: "AST-safe replacement of buggy functions" },
                  { icon: "✅", n: "05", t: "Verify", d: "Re-run tests to confirm the bug is fixed" },
                ].map(s => (
                  <div key={s.n} className="card" style={{ padding: 22, transition: "all .25s", cursor: "default", textAlign: "center" }}>
                    <div style={{ fontSize: 28, marginBottom: 10 }}>{s.icon}</div>
                    <div style={{ fontSize: 10, color: "rgba(168,85,247,.5)", letterSpacing: "2px", marginBottom: 6 }}>{s.n}</div>
                    <div style={{ fontSize: 13, fontWeight: 600, color: "#e8d8ff", marginBottom: 8 }}>{s.t}</div>
                    <div style={{ fontSize: 11, color: "rgba(200,184,232,.5)", lineHeight: 1.6 }}>{s.d}</div>
                  </div>
                ))}
              </div>
            </div>

            {/* Stats */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 16 }}>
              {[
                { val: "70%", lbl: "Manual effort reduced" },
                { val: "3×", lbl: "Faster than single-teacher" },
                { val: "60%", lbl: "Debugging time saved" },
              ].map(s => (
                <div key={s.lbl} className="card" style={{ padding: 28, textAlign: "center" }}>
                  <div style={{ fontFamily: "'Space Grotesk',sans-serif", fontSize: 42, fontWeight: 800, background: "linear-gradient(135deg,#e879f9,#818cf8)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>{s.val}</div>
                  <div style={{ fontSize: 12, color: "rgba(200,184,232,.5)", marginTop: 6 }}>{s.lbl}</div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ── UPLOAD ── */}
        {page === "upload" && (
          <div className="page-enter" style={{ maxWidth: 600, margin: "60px auto 0" }}>
            <SectionTitle>Upload Python Project</SectionTitle>
            <p style={{ fontSize: 13, color: "rgba(200,184,232,.5)", marginBottom: 32 }}>Upload your buggy Python file and test file to begin automated repair.</p>

            {["Python Source File", "Test File"].map((lbl, i) => (
              <div key={lbl} style={{ marginBottom: 20 }}>
                <div style={{ fontSize: 12, color: "rgba(200,184,232,.6)", marginBottom: 10, letterSpacing: ".5px" }}>{lbl}</div>
                <label className="upload-zone" style={{ display: "block" }}>
                  <input type="file" accept=".py" onChange={e => setUploadedFile(e.target.files[0]?.name)} />
                  <div style={{ fontSize: 24, marginBottom: 10, opacity: .5 }}>📄</div>
                  <div style={{ fontSize: 13, color: "rgba(200,184,232,.5)" }}>
                    {i === 0 && uploadedFile ? <span style={{ color: "#c084fc" }}>✓ {uploadedFile}</span> : <>Drop <span style={{ color: "#c084fc" }}>.py file</span> or click</>}
                  </div>
                </label>
              </div>
            ))}

            <div style={{ marginBottom: 28 }}>
              <div style={{ fontSize: 12, color: "rgba(200,184,232,.6)", marginBottom: 12 }}>Or use a demo example:</div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
                {["BoundedStack", "Fibonacci", "MaxInList", "BinarySearch"].map(ex => (
                  <div key={ex} className="card" onClick={() => setUploadedFile(ex + ".py")} style={{
                    padding: "14px 18px", cursor: "pointer",
                    border: uploadedFile === ex + ".py" ? "1px solid rgba(168,85,247,.6)" : "",
                    background: uploadedFile === ex + ".py" ? "rgba(168,85,247,.1)" : "",
                  }}>
                    <div style={{ fontSize: 12, fontWeight: 600, color: "#e8d8ff" }}>{ex}</div>
                    <div style={{ fontSize: 10, color: "rgba(200,184,232,.4)", marginTop: 3 }}>coding · sample bug</div>
                  </div>
                ))}
              </div>
            </div>

            <button className="btn-primary" style={{ width: "100%", fontSize: 15 }} onClick={runPipeline}>
              ▶ &nbsp; Run Automatic Repair
            </button>
          </div>
        )}

        {/* ── PIPELINE ── */}
        {page === "pipeline" && (
          <div className="page-enter">
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 32, paddingTop: 40 }}>
              <div>
                <SectionTitle>Repair Pipeline</SectionTitle>
                <p style={{ fontSize: 13, color: "rgba(200,184,232,.5)" }}>
                  {done ? "✅ Pipeline complete — bug fixed successfully!" : running ? "⏳ Pipeline running…" : "Start the pipeline from Upload page"}
                </p>
              </div>
              {done && <div style={{
                background: "rgba(52,211,153,.12)", border: "1px solid rgba(52,211,153,.3)",
                borderRadius: 20, padding: "8px 20px", fontSize: 13, color: "#34d399", fontWeight: 600,
              }}>🎉 Fixed in 3 attempts</div>}
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 14, marginBottom: 28 }}>
              {PIPELINE_STEPS.map((s, i) => {
                const st = step > i ? "done" : step === i && running ? "run" : done && i < 6 ? "done" : "idle";
                return (
                  <div key={s.id} className={`card step-${st}`} style={{ padding: "20px 22px", display: "flex", alignItems: "center", gap: 14, transition: "all .3s" }}>
                    <div style={{ fontSize: 22 }}>{s.icon}</div>
                    <div>
                      <div style={{ fontSize: 10, opacity: .5, marginBottom: 3 }}>STEP {s.id}</div>
                      <div style={{ fontSize: 13, fontWeight: 600 }}>{s.label}</div>
                    </div>
                    <div style={{ marginLeft: "auto", fontSize: 18 }}>
                      {st === "done" ? "✅" : st === "run" ? <span style={{ display: "inline-block", animation: "spin .8s linear infinite" }}>⟳</span> : st === "fail" ? "❌" : "—"}
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Live Log */}
            <div className="card" style={{ overflow: "hidden" }}>
              <div style={{ padding: "12px 20px", borderBottom: "1px solid rgba(168,85,247,.1)", display: "flex", alignItems: "center", gap: 8 }}>
                <div style={{ display: "flex", gap: 5 }}>
                  {["#ff5f57", "#febc2e", "#28c840"].map(c => <div key={c} style={{ width: 10, height: 10, borderRadius: "50%", background: c }} />)}
                </div>
                <span style={{ fontSize: 11, color: "rgba(200,184,232,.4)", letterSpacing: "1px", textTransform: "uppercase", marginLeft: 6 }}>Pipeline Log</span>
                {running && <div style={{ marginLeft: "auto", width: 7, height: 7, borderRadius: "50%", background: "#9333ea", boxShadow: "0 0 8px #9333ea", animation: "pulse 1s infinite" }} />}
              </div>
              <div ref={logRef} style={{ padding: 20, height: 260, overflowY: "auto", fontFamily: "monospace", fontSize: 12, lineHeight: 1.9 }}>
                {logs.length === 0 && <div style={{ color: "rgba(200,184,232,.2)" }}>// Waiting…</div>}
                {logs.map((l, i) => (
                  <div key={i}>
                    <span style={{ color: "rgba(200,184,232,.2)", marginRight: 10 }}>[{l.ts}]</span>
                    <span className={`log-${l.type}`}>{l.m}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* ── CODE DIFF ── */}
        {page === "code" && (
          <div className="page-enter">
            <div style={{ paddingTop: 40, marginBottom: 28 }}>
              <SectionTitle>Code Diff — Before vs After</SectionTitle>
              <p style={{ fontSize: 13, color: "rgba(200,184,232,.5)" }}>Suspicious lines highlighted · AST-safe patch applied</p>
            </div>

            {/* SBFL Heatmap */}
            <div className="card" style={{ padding: 24, marginBottom: 20 }}>
              <div style={{ fontSize: 12, color: "rgba(168,85,247,.7)", letterSpacing: "2px", textTransform: "uppercase", marginBottom: 16 }}>SBFL Fault Localization</div>
              {SUSPICIOUS.map(s => (
                <div key={s.line} style={{ marginBottom: 14 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 5, fontSize: 12 }}>
                    <span style={{ color: "#c084fc" }}>Line {s.line}</span>
                    <span style={{ color: "#f87171", fontWeight: 600 }}>score {s.score}</span>
                  </div>
                  <div style={{ background: "rgba(255,255,255,.04)", borderRadius: 6, padding: "6px 12px", fontFamily: "monospace", fontSize: 11, color: "rgba(200,184,232,.5)", marginBottom: 6 }}>
                    <span style={{ background: `rgba(239,68,68,${s.score * .3})`, padding: "1px 4px", borderRadius: 3 }}>{s.code}</span>
                  </div>
                  <div style={{ height: 6, background: "rgba(255,255,255,.05)", borderRadius: 3, overflow: "hidden" }}>
                    <div className="heatbar" style={{ width: `${s.score * 100}%` }} />
                  </div>
                </div>
              ))}
            </div>

            {/* Diff */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
              {[
                { label: "BEFORE", badge: "BUGGY", bc: "#f87171", bg: "rgba(239,68,68,.08)", code: BUGGY, marker: "r" },
                { label: "AFTER", badge: "FIXED", bc: "#34d399", bg: "rgba(52,211,153,.06)", code: FIXED, marker: "g" },
              ].map(p => (
                <div key={p.label} className="card" style={{ overflow: "hidden", background: p.bg }}>
                  <div style={{ padding: "10px 16px", borderBottom: "1px solid rgba(168,85,247,.1)", display: "flex", alignItems: "center", gap: 8, background: "rgba(0,0,0,.2)" }}>
                    <div style={{ display: "flex", gap: 4 }}>
                      {["#ff5f57", "#febc2e", "#28c840"].map(c => <div key={c} style={{ width: 9, height: 9, borderRadius: "50%", background: c }} />)}
                    </div>
                    <span style={{ fontSize: 11, color: "rgba(200,184,232,.4)", marginLeft: 5 }}>{p.label}</span>
                    <div style={{ marginLeft: "auto", background: p.bg, border: `1px solid ${p.bc}40`, borderRadius: 4, padding: "2px 8px", fontSize: 9, color: p.bc, letterSpacing: ".5px" }}>{p.badge}</div>
                  </div>
                  <div style={{ padding: 16, overflowX: "auto" }}>
                    <pre style={{ fontFamily: "monospace", fontSize: 12, lineHeight: 1.75, margin: 0 }}>
                      {p.code.split("\n").map((line, i) => {
                        const suspicious = p.marker === "r" && SUSPICIOUS.some(s => s.code.trim() === line.trim());
                        return (
                          <div key={i} style={{
                            color: p.marker === "r" ? (suspicious ? "#fca5a5" : "#94a3b8") : "#86efac",
                            background: suspicious ? "rgba(239,68,68,.12)" : "transparent",
                            paddingLeft: 4, borderRadius: 3, display: "block",
                          }}>
                            <span style={{ color: p.marker === "r" ? "rgba(239,68,68,.4)" : "rgba(52,211,153,.4)", marginRight: 8, userSelect: "none" }}>{p.marker === "r" ? "−" : "+"}</span>
                            {line}
                          </div>
                        );
                      })}
                    </pre>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ── TESTS ── */}
        {page === "tests" && (
          <div className="page-enter" style={{ paddingTop: 40 }}>
            <SectionTitle>Test Results</SectionTitle>

            {/* Summary */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 14, margin: "24px 0 32px" }}>
              {[
                { v: 5, l: "Tests Run", c: "#a78bfa" },
                { v: 3, l: "Passed", c: "#34d399" },
                { v: 2, l: "Failed", c: "#f87171" },
                { v: "60%", l: "Pass Rate", c: "#fbbf24" },
              ].map(s => (
                <div key={s.l} className="card" style={{ padding: 20, textAlign: "center" }}>
                  <div style={{ fontFamily: "'Space Grotesk',sans-serif", fontSize: 32, fontWeight: 800, color: s.c }}>{s.v}</div>
                  <div style={{ fontSize: 11, color: "rgba(200,184,232,.4)", marginTop: 4 }}>{s.l}</div>
                </div>
              ))}
            </div>

            {/* Test list */}
            <div style={{ display: "flex", flexDirection: "column", gap: 10, marginBottom: 28 }}>
              {TESTS.map(t => (
                <div key={t.name} className="card" style={{
                  padding: "16px 22px", display: "flex", alignItems: "flex-start", gap: 14,
                  borderColor: t.status === "pass" ? "rgba(52,211,153,.2)" : "rgba(239,68,68,.2)",
                  background: t.status === "pass" ? "rgba(52,211,153,.04)" : "rgba(239,68,68,.04)",
                }}>
                  <div style={{ fontSize: 18 }}>{t.status === "pass" ? "✅" : "❌"}</div>
                  <div style={{ flex: 1 }}>
                    <div style={{ fontFamily: "monospace", fontSize: 13, color: t.status === "pass" ? "#86efac" : "#fca5a5", marginBottom: t.trace ? 6 : 0 }}>{t.name}</div>
                    {t.trace && <div style={{ fontSize: 11, color: "rgba(200,184,232,.4)", fontFamily: "monospace" }}>{t.trace}</div>}
                  </div>
                  <div style={{
                    fontSize: 11, padding: "3px 10px", borderRadius: 12,
                    background: t.status === "pass" ? "rgba(52,211,153,.1)" : "rgba(239,68,68,.1)",
                    color: t.status === "pass" ? "#34d399" : "#f87171", fontWeight: 600
                  }}>
                    {t.status.toUpperCase()}
                  </div>
                </div>
              ))}
            </div>

            {/* Failure trace */}
            <div className="card" style={{ overflow: "hidden" }}>
              <div style={{ padding: "10px 18px", borderBottom: "1px solid rgba(168,85,247,.1)", fontSize: 11, color: "rgba(200,184,232,.4)", letterSpacing: "1px", textTransform: "uppercase" }}>Failure Trace</div>
              <div style={{ padding: 18, fontFamily: "monospace", fontSize: 12, lineHeight: 1.9, color: "#f87171" }}>
                <div><span style={{ color: "rgba(200,184,232,.3)" }}>E</span>  OverflowError: Stack overflow not raised</div>
                <div><span style={{ color: "rgba(200,184,232,.3)" }}>E</span>  AssertionError: Expected OverflowError at line 12</div>
                <div style={{ marginTop: 8, color: "rgba(200,184,232,.3)" }}>SBFL rank → Line 12 (score 0.91) · Line 14 (score 0.74)</div>
              </div>
            </div>
          </div>
        )}

        {/* ── ATTEMPTS ── */}
        {page === "attempts" && (
          <div className="page-enter" style={{ paddingTop: 40 }}>
            <SectionTitle>Repair Attempts Timeline</SectionTitle>
            <p style={{ fontSize: 13, color: "rgba(200,184,232,.5)", marginBottom: 36 }}>TestMate tried 3 patches — fixed on attempt 3</p>

            {/* Timeline */}
            <div style={{ position: "relative" }}>
              <div style={{ position: "absolute", left: 24, top: 0, bottom: 0, width: 2, background: "linear-gradient(180deg,rgba(168,85,247,.5),rgba(168,85,247,.05))", borderRadius: 1 }} />
              {ATTEMPTS.map((a, i) => (
                <div key={a.n} style={{ display: "flex", gap: 24, marginBottom: 28, paddingLeft: 0, animation: `slideUp .4s ease ${i * .15}s both` }}>
                  {/* dot */}
                  <div style={{
                    width: 50, height: 50, borderRadius: "50%", flexShrink: 0,
                    display: "flex", alignItems: "center", justifyContent: "center",
                    fontSize: 18, zIndex: 1,
                    background: a.status === "success" ? "rgba(52,211,153,.15)" : a.status === "fail" ? "rgba(239,68,68,.12)" : "rgba(168,85,247,.1)",
                    border: `2px solid ${a.status === "success" ? "rgba(52,211,153,.5)" : "rgba(239,68,68,.35)"}`,
                    boxShadow: a.status === "success" ? "0 0 16px rgba(52,211,153,.3)" : "none",
                  }}>
                    {a.status === "success" ? "🎉" : "❌"}
                  </div>

                  <div className="card" style={{
                    flex: 1, padding: "18px 22px",
                    borderColor: a.status === "success" ? "rgba(52,211,153,.3)" : "rgba(239,68,68,.2)",
                    background: a.status === "success" ? "rgba(52,211,153,.05)" : "rgba(239,68,68,.03)",
                  }}>
                    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 10 }}>
                      <div style={{ fontFamily: "'Space Grotesk',sans-serif", fontSize: 15, fontWeight: 700, color: "#e8d8ff" }}>Attempt {a.n}</div>
                      <div style={{
                        fontSize: 11, padding: "3px 12px", borderRadius: 12, fontWeight: 600,
                        background: a.status === "success" ? "rgba(52,211,153,.1)" : "rgba(239,68,68,.1)",
                        color: a.status === "success" ? "#34d399" : "#f87171",
                      }}>{a.status === "success" ? "SUCCESS" : "FAILED"}</div>
                    </div>
                    <div style={{ display: "flex", gap: 20, fontSize: 12, color: "rgba(200,184,232,.5)" }}>
                      <div><span style={{ color: "rgba(168,85,247,.6)" }}>Patch: </span>{a.patch}</div>
                    </div>
                    <div style={{
                      marginTop: 8, fontSize: 12, fontFamily: "monospace",
                      color: a.status === "success" ? "#86efac" : "#fca5a5",
                      background: "rgba(0,0,0,.2)", borderRadius: 8, padding: "8px 12px", marginTop: 12,
                    }}>{a.result}</div>
                  </div>
                </div>
              ))}
            </div>

            {/* Final */}
            <div style={{
              marginTop: 16, padding: 28, borderRadius: 20, textAlign: "center",
              background: "linear-gradient(135deg,rgba(52,211,153,.08),rgba(99,102,241,.08))",
              border: "1px solid rgba(52,211,153,.25)",
            }}>
              <div style={{ fontSize: 36, marginBottom: 12 }}>🎉</div>
              <div style={{ fontFamily: "'Space Grotesk',sans-serif", fontSize: 22, fontWeight: 800, color: "#e8d8ff", marginBottom: 8 }}>Bug Fixed Successfully</div>
              <div style={{ fontSize: 13, color: "rgba(200,184,232,.5)" }}>TestMate repaired BoundedStack in 3 attempts · All 5 tests now passing</div>
              <button className="btn-primary" style={{ marginTop: 20 }} onClick={() => setPage("code")}>View Code Diff →</button>
            </div>
          </div>
        )}

      </div>
    </div>
  );
}

function SectionTitle({ children }) {
  return (
    <h2 style={{
      fontFamily: "'Space Grotesk',sans-serif",
      fontSize: 24, fontWeight: 700,
      color: "#f0e8ff",
      display: "flex", alignItems: "center", gap: 12,
    }}>
      <div style={{ width: 4, height: 24, background: "linear-gradient(180deg,#9333ea,#6366f1)", borderRadius: 2 }} />
      {children}
    </h2>
  );
}