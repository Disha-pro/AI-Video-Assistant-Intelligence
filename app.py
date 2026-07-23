import streamlit as st
import streamlit.components.v1 as components
import time
import random
from dotenv import load_dotenv
from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500;600&display=swap');

/* ── Root Variables — phosphor / signal theme ── */
:root {
    --bg: #05080a;
    --surface: #0d1410;
    --surface-2: #141c18;
    --border: #1e2b23;
    --accent: #39FF6A;
    --accent-glow: #8CFFB0;
    --accent-dim: rgba(57,255,106,0.12);
    --amber: #FFB020;
    --amber-dim: rgba(255,176,32,0.12);
    --text: #E8FFEE;
    --text-muted: #5E7B6C;
    --danger: #FF5C5C;
}

/* ── Global Reset ── */
html, body, [class*="css"] {
    font-family: 'JetBrains Mono', monospace;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

/* Transparent so the animated network canvas (attached to <body>, behind
   this element) shows through; html/body carries the solid fallback color. */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stHeader"],
.main { background: transparent !important; }

/* Faint signal grid background */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background-image:
        linear-gradient(rgba(57, 255, 106, 0.035) 1px, transparent 1px),
        linear-gradient(90deg, rgba(57, 255, 106, 0.035) 1px, transparent 1px);
    background-size: 42px 42px;
    pointer-events: none;
    z-index: 0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Headings ── */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Syne', sans-serif !important;
    color: var(--text) !important;
}

/* ── Hero Title ── */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2rem, 5vw, 3.5rem);
    font-weight: 800;
    line-height: 1.1;
    margin: 0;
    background: linear-gradient(135deg, #ffffff 0%, var(--accent-glow) 55%, var(--accent) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-shadow: 0 0 34px rgba(57,255,106,0.18);
}

.hero-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: var(--text-muted);
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-top: 0.5rem;
}

/* ── Hero waveform — signature element ── */
.hero-wave {
    display: flex;
    align-items: center;
    gap: 3px;
    height: 52px;
    margin: 1.4rem 0 0.6rem 0;
    opacity: 0.9;
}
.hero-wave .wbar {
    flex: 1;
    min-width: 2px;
    background: linear-gradient(180deg, var(--accent-glow), var(--accent));
    border-radius: 2px;
    box-shadow: 0 0 7px rgba(57,255,106,0.35);
    animation: waveMove 1.7s ease-in-out infinite;
    transform-origin: center;
}
@keyframes waveMove {
    0%, 100% { transform: scaleY(0.22); opacity: 0.55; }
    50%      { transform: scaleY(1);    opacity: 1; }
}

/* ── Cards ── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.card:hover {
    border-color: var(--accent);
    box-shadow: 0 0 0 1px rgba(57,255,106,0.15), 0 8px 28px rgba(0,0,0,0.35);
}
.card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: linear-gradient(180deg, var(--accent), var(--amber));
}

.card-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.card-content {
    font-size: 0.875rem;
    line-height: 1.7;
    color: var(--text);
}

/* ── Accent Badge ── */
.badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.badge-signal { background: var(--accent-dim); color: var(--accent-glow); border: 1px solid rgba(57,255,106,0.3); }
.badge-amber  { background: var(--amber-dim);  color: var(--amber);       border: 1px solid rgba(255,176,32,0.3); }
.badge-dim    { background: rgba(255,255,255,0.04); color: var(--text-muted); border: 1px solid var(--border); }

/* ── Input & Buttons ── */
.stTextInput > div > div > input,
.stSelectbox > div > div {
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(57,255,106,0.2) !important;
}

.stButton > button {
    background: linear-gradient(135deg, var(--accent), #1fa851) !important;
    color: #041007 !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.875rem !important;
    letter-spacing: 0.05em !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.2s !important;
    text-transform: uppercase !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 26px rgba(57,255,106,0.45) !important;
}
.stButton > button[kind="secondary"] {
    background: var(--surface-2) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
}

/* ── Pipeline step rows — equalizer status ── */
.step-row {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    padding: 0.6rem 0.85rem;
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 8px;
    margin: 0.35rem 0;
    font-size: 0.78rem;
    transition: border-color 0.2s;
}
.step-row.is-active { border-color: rgba(57,255,106,0.4); }
.step-row.is-done    { border-color: rgba(57,255,106,0.2); }

.step-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: var(--text-muted);
    width: 1.3rem;
    flex-shrink: 0;
}
.step-label { flex: 1; color: var(--text); }

.eq { display: flex; align-items: flex-end; gap: 2px; height: 15px; flex-shrink: 0; }
.eq .bar {
    width: 3px;
    background: var(--border);
    border-radius: 1px;
    height: 30%;
    transition: height 0.3s ease, background 0.3s ease;
}
.eq.done .bar {
    background: var(--accent);
    height: 100%;
    box-shadow: 0 0 5px rgba(57,255,106,0.5);
}
.eq.active .bar {
    background: var(--accent);
    animation: eqPulse 0.9s ease-in-out infinite;
}
.eq.active .bar:nth-child(1) { animation-delay: 0s; }
.eq.active .bar:nth-child(2) { animation-delay: 0.15s; }
.eq.active .bar:nth-child(3) { animation-delay: 0.3s; }
.eq.active .bar:nth-child(4) { animation-delay: 0.1s; }
.eq.active .bar:nth-child(5) { animation-delay: 0.25s; }
@keyframes eqPulse { 0%, 100% { height: 25%; } 50% { height: 100%; } }

/* ── Chat ── */
.chat-container {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem;
    max-height: 420px;
    overflow-y: auto;
    margin-bottom: 1rem;
}
.chat-msg { margin-bottom: 1rem; display: flex; flex-direction: column; gap: 0.2rem; }
.chat-label { font-size: 0.65rem; font-weight: 700; letter-spacing: 0.15em; text-transform: uppercase; }
.chat-bubble { display: inline-block; padding: 0.6rem 1rem; border-radius: 10px; font-size: 0.85rem; line-height: 1.6; max-width: 90%; }

.user-label  { color: var(--accent-glow); }
.bot-label   { color: var(--amber); }
.user-bubble { background: var(--accent-dim); border: 1px solid rgba(57,255,106,0.25); align-self: flex-end; }
.bot-bubble  { background: var(--amber-dim);  border: 1px solid rgba(255,176,32,0.25); align-self: flex-start; }

/* ── Divider ── */
hr { border: none !important; border-top: 1px solid var(--border) !important; margin: 1.5rem 0 !important; }

/* ── Transcript box ── */
.transcript-box {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.25rem;
    font-size: 0.82rem;
    line-height: 1.8;
    max-height: 300px;
    overflow-y: auto;
    color: var(--text-muted);
    white-space: pre-wrap;
    word-break: break-word;
}

/* ── Stale Streamlit elements ── */
.stProgress > div > div > div { background: var(--accent) !important; }
.stSpinner > div { border-top-color: var(--accent) !important; }
[data-testid="stMarkdownContainer"] p { color: var(--text) !important; }
label { color: var(--text-muted) !important; font-size: 0.8rem !important; }

/* scrollbar */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }
</style>
""", unsafe_allow_html=True)

# ─── Animated Background — living signal network ────────────────────────────────
# Injects a fixed full-viewport canvas into the parent document (not the
# component's own iframe) so it sits behind every Streamlit element. Drifting
# nodes connect when close, occasionally amber, and gently part around the
# cursor — read as a live neural / signal network rather than decoration.
components.html("""
<script>
(function() {
    const doc = window.parent.document;
    if (doc.getElementById('ai-neural-bg')) return;  // already running, skip

    const canvas = document.createElement('canvas');
    canvas.id = 'ai-neural-bg';
    Object.assign(canvas.style, {
        position: 'fixed', top: '0', left: '0',
        width: '100vw', height: '100vh',
        zIndex: '-1', pointerEvents: 'none',
    });
    doc.body.insertBefore(canvas, doc.body.firstChild);
    const ctx = canvas.getContext('2d');

    let w, h;
    const mouse = { x: -9999, y: -9999 };

    function resize() {
        w = canvas.width = window.innerWidth;
        h = canvas.height = window.innerHeight;
    }
    resize();
    window.addEventListener('resize', resize);
    doc.addEventListener('mousemove', (e) => { mouse.x = e.clientX; mouse.y = e.clientY; });

    const COUNT = Math.min(85, Math.floor((window.innerWidth * window.innerHeight) / 17000));
    const nodes = Array.from({ length: COUNT }, () => ({
        x: Math.random() * w,
        y: Math.random() * h,
        vx: (Math.random() - 0.5) * 0.25,
        vy: (Math.random() - 0.5) * 0.25,
        amber: Math.random() < 0.12,
    }));

    const MAX_DIST = 130;

    function tick() {
        ctx.clearRect(0, 0, w, h);

        for (const n of nodes) {
            n.x += n.vx;
            n.y += n.vy;
            if (n.x < 0 || n.x > w) n.vx *= -1;
            if (n.y < 0 || n.y > h) n.vy *= -1;

            const dx = mouse.x - n.x, dy = mouse.y - n.y;
            const d = Math.sqrt(dx * dx + dy * dy);
            if (d < 140) {
                n.x -= dx * 0.0025;
                n.y -= dy * 0.0025;
            }
        }

        for (let i = 0; i < nodes.length; i++) {
            for (let j = i + 1; j < nodes.length; j++) {
                const a = nodes[i], b = nodes[j];
                const dx = a.x - b.x, dy = a.y - b.y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < MAX_DIST) {
                    const alpha = (1 - dist / MAX_DIST) * 0.18;
                    ctx.strokeStyle = `rgba(57,255,106,${alpha})`;
                    ctx.lineWidth = 1;
                    ctx.beginPath();
                    ctx.moveTo(a.x, a.y);
                    ctx.lineTo(b.x, b.y);
                    ctx.stroke();
                }
            }
        }

        for (const n of nodes) {
            ctx.beginPath();
            ctx.fillStyle = n.amber ? 'rgba(255,176,32,0.75)' : 'rgba(140,255,176,0.85)';
            ctx.shadowBlur = 6;
            ctx.shadowColor = n.amber ? 'rgba(255,176,32,0.8)' : 'rgba(57,255,106,0.8)';
            ctx.arc(n.x, n.y, n.amber ? 1.6 : 1.8, 0, Math.PI * 2);
            ctx.fill();
        }

        requestAnimationFrame(tick);
    }
    tick();
})();
</script>
""", height=0, width=0)

# ─── Session State Init ──────────────────────────────────────────────────────────
for key, default in {
    "result": None,
    "chat_history": [],
    "processing": False,
    "pipeline_done": False,
    "pipeline_steps": {},
    "wave_seed": random.randint(0, 10_000),
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─── Helpers ────────────────────────────────────────────────────────────────────
def render_hero_wave(bars: int = 46):
    """Signature element: a live-looking waveform strip, each bar animating
    on its own offset so it reads like a continuous audio signal."""
    rnd = random.Random(st.session_state.wave_seed)
    spans = "".join(
        f'<div class="wbar" style="animation-delay:-{rnd.uniform(0, 1.7):.2f}s;"></div>'
        for _ in range(bars)
    )
    st.markdown(f'<div class="hero-wave">{spans}</div>', unsafe_allow_html=True)

def step_status(steps: dict, key: str) -> str:
    return steps.get(key, "pending")

def render_step_bar(label: str, key: str, num: str):
    state = step_status(st.session_state.pipeline_steps, key)
    row_cls = "is-active" if state == "active" else ("is-done" if state == "done" else "")
    eq_cls = state if state in ("active", "done") else ""
    st.markdown(f"""
    <div class="step-row {row_cls}">
        <span class="step-num">{num}</span>
        <span class="step-label">{label}</span>
        <div class="eq {eq_cls}">
            <div class="bar"></div><div class="bar"></div><div class="bar"></div>
            <div class="bar"></div><div class="bar"></div>
        </div>
    </div>""", unsafe_allow_html=True)

# ─── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="hero-title" style="font-size:1.6rem">🎬 AI<br>Video</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Meeting Intelligence</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<span class="badge badge-signal">Input</span>', unsafe_allow_html=True)
    source = st.text_input("YouTube URL or File Path", placeholder="https://youtube.com/watch?v=... or /path/to/file.mp4")

    language = st.selectbox("Language", ["english", "hinglish"], index=0)

    run_btn = st.button("⚡  Analyse", use_container_width=True)

    if st.session_state.pipeline_done:
        st.markdown("---")
        st.markdown('<span class="badge badge-signal">Pipeline Status</span>', unsafe_allow_html=True)
        for num, step, label in [
            ("01", "audio",      "Audio Processing"),
            ("02", "transcript", "Transcription"),
            ("03", "title",      "Title Generation"),
            ("04", "summary",    "Summarisation"),
            ("05", "extract",    "Extraction"),
            ("06", "rag",        "RAG Engine"),
        ]:
            render_step_bar(label, step, num)

# ─── Main Area ──────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">AI Video Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Transcribe · Summarise · Chat with your meetings</div>', unsafe_allow_html=True)
render_hero_wave()
st.markdown("---")

# ── Run Pipeline ────────────────────────────────────────────────────────────────
if run_btn:
    if not source.strip():
        st.error("Please enter a YouTube URL or file path.")
    else:
        st.session_state.pipeline_done = False
        st.session_state.result = None
        st.session_state.chat_history = []
        st.session_state.pipeline_steps = {}
        st.session_state.wave_seed = random.randint(0, 10_000)

        progress_placeholder = st.empty()

        def update_step(key, state):
            st.session_state.pipeline_steps[key] = state

        try:
            with progress_placeholder.container():
                st.info("⚙️ Pipeline running — see sidebar for live status…")

            update_step("audio", "active")
            chunks = process_input(source)
            update_step("audio", "done")

            update_step("transcript", "active")
            transcript = transcribe_all(chunks, language)
            update_step("transcript", "done")

            update_step("title", "active")
            title = generate_title(transcript)
            update_step("title", "done")

            update_step("summary", "active")
            summary = summarize(transcript)
            update_step("summary", "done")

            update_step("extract", "active")
            action_items  = extract_action_items(transcript)
            decisions     = extract_key_decisions(transcript)
            questions     = extract_questions(transcript)
            update_step("extract", "done")

            update_step("rag", "active")
            rag_chain = build_rag_chain(transcript)
            update_step("rag", "done")

            st.session_state.result = {
                "title": title,
                "transcript": transcript,
                "summary": summary,
                "action_items": action_items,
                "key_decisions": decisions,
                "open_questions": questions,
                "rag_chain": rag_chain,
            }
            st.session_state.pipeline_done = True
            progress_placeholder.success("✅ Analysis complete!")
            time.sleep(0.5)
            progress_placeholder.empty()
            st.rerun()

        except Exception as e:
            for k in ["audio","transcript","title","summary","extract","rag"]:
                if st.session_state.pipeline_steps.get(k) == "active":
                    st.session_state.pipeline_steps[k] = "pending"
            progress_placeholder.error(f"❌ Error: {e}")

# ── Results ──────────────────────────────────────────────────────────────────────
if st.session_state.result:
    r = st.session_state.result

    # Title banner
    st.markdown(f"""
    <div class="card">
        <div class="card-title">📌 Session Title</div>
        <div style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:700;color:var(--text)">
            {r['title']}
        </div>
    </div>""", unsafe_allow_html=True)

    # Top row: summary + transcript
    col1, col2 = st.columns([3, 2], gap="medium")

    with col1:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">📋 Summary</div>
            <div class="card-content">{r['summary']}</div>
        </div>""", unsafe_allow_html=True)

    with col2:
        with st.expander("📝 Full Transcript", expanded=False):
            st.markdown(f'<div class="transcript-box">{r["transcript"]}</div>', unsafe_allow_html=True)

    # Second row: action items | decisions | questions
    c1, c2, c3 = st.columns(3, gap="medium")

    with c1:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">✅ Action Items</div>
            <div class="card-content">{r['action_items']}</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">🔑 Key Decisions</div>
            <div class="card-content">{r['key_decisions']}</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">❓ Open Questions</div>
            <div class="card-content">{r['open_questions']}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── RAG Chat ──────────────────────────────────────────────────────────────
    st.markdown('<div style="font-family:\'Syne\',sans-serif;font-size:1.2rem;font-weight:700;margin-bottom:1rem">💬 Chat with your Meeting</div>', unsafe_allow_html=True)

    # Chat history display
    if st.session_state.chat_history:
        chat_html = '<div class="chat-container">'
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                chat_html += f"""
                <div class="chat-msg" style="align-items:flex-end">
                    <span class="chat-label user-label">You</span>
                    <div class="chat-bubble user-bubble">{msg['content']}</div>
                </div>"""
            else:
                chat_html += f"""
                <div class="chat-msg" style="align-items:flex-start">
                    <span class="chat-label bot-label">🤖 Assistant</span>
                    <div class="chat-bubble bot-bubble">{msg['content']}</div>
                </div>"""
        chat_html += '</div>'
        st.markdown(chat_html, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="card" style="text-align:center;padding:2rem">
            <div style="font-size:2rem;margin-bottom:0.5rem">💬</div>
            <div style="color:var(--text-muted);font-size:0.85rem">Ask anything about your meeting transcript</div>
        </div>""", unsafe_allow_html=True)

    # Chat input
    chat_col1, chat_col2 = st.columns([5, 1], gap="small")
    with chat_col1:
        user_input = st.text_input("Your question", placeholder="What were the main decisions made?", label_visibility="collapsed")
    with chat_col2:
        send_btn = st.button("Send →", use_container_width=True)

    if send_btn and user_input.strip():
        with st.spinner("Thinking…"):
            answer = ask_question(r["rag_chain"], user_input.strip())
        st.session_state.chat_history.append({"role": "user",      "content": user_input.strip()})
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.rerun()

    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()

else:
    # Empty state
    st.markdown("""
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;padding:5rem 2rem;text-align:center">
        <div style="font-size:4rem;margin-bottom:1rem">🎬</div>
        <div style="font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:700;color:var(--text);margin-bottom:0.5rem">
            Ready to Analyse
        </div>
        <div style="color:var(--text-muted);font-size:0.85rem;max-width:380px;line-height:1.7">
            Paste a YouTube URL or local file path in the sidebar, choose your language, and hit <strong>Analyse</strong> to get started.
        </div>
        <div style="margin-top:2rem;display:flex;gap:1rem;flex-wrap:wrap;justify-content:center">
            <span class="badge badge-signal">Transcription</span>
            <span class="badge badge-amber">Summarisation</span>
            <span class="badge badge-signal">RAG Chat</span>
        </div>
    </div>""", unsafe_allow_html=True)


