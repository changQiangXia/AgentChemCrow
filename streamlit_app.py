from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any

import streamlit as st
import streamlit.components.v1 as components
from openai import APIStatusError

from chemcrow_lite.agent import ChemCrowLiteAgent
from chemcrow_lite.benchmarking import run_benchmark
from chemcrow_lite.config import Settings
from chemcrow_lite.healthcheck import run_healthcheck
from chemcrow_lite.prompting import BASELINE_SYSTEM_PROMPT, SYSTEM_PROMPT
from chemcrow_lite.tools.chromophore import chromophore_rf_screen
from chemcrow_lite.tools.reaction import reaction_outcome_heuristic, retrosynthesis_overview
from chemcrow_lite.tools.safety import control_chem_check, explosive_check, safety_summary


st.set_page_config(page_title="ChemCrow-Lite", page_icon="⚗️", layout="wide", initial_sidebar_state="collapsed")


STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Inter+Tight:wght@600;700;800&family=JetBrains+Mono:wght@500;600&display=swap');
:root{--ink:#0f141c;--ink-soft:rgba(15,20,28,.68);--ink-faint:rgba(15,20,28,.48);--white:#f7f8fb;--line:rgba(255,255,255,.5);--line-dark:rgba(255,255,255,.16)}
html,body,.stApp,.main,[data-testid="stAppViewContainer"]{scroll-behavior:smooth}
.stApp{background:radial-gradient(circle at 12% 8%,rgba(176,206,255,.24),transparent 24%),radial-gradient(circle at 84% 10%,rgba(210,227,255,.22),transparent 20%),radial-gradient(circle at 50% 112%,rgba(45,67,96,.18),transparent 36%),linear-gradient(180deg,#1b2432 0%,#2c3950 18%,#5c718d 46%,#cad4df 78%,#f3f5f8 100%);color:var(--ink);font-family:"Inter",-apple-system,BlinkMacSystemFont,"SF Pro Display","Helvetica Neue",sans-serif}
.stApp::before{content:"";position:fixed;inset:0;pointer-events:none;background:radial-gradient(circle at 50% -8%,rgba(255,255,255,.18),transparent 26%),radial-gradient(circle at 10% 24%,rgba(169,198,255,.16),transparent 18%),radial-gradient(circle at 88% 18%,rgba(255,255,255,.10),transparent 16%),linear-gradient(180deg,rgba(255,255,255,.06),rgba(255,255,255,0) 30%,rgba(10,24,44,.03) 100%);filter:blur(10px);z-index:0}
.stApp::after{content:"";position:fixed;inset:0;pointer-events:none;opacity:.16;mix-blend-mode:soft-light;background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='180' height='180' viewBox='0 0 180 180'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='1.08' numOctaves='2' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='180' height='180' filter='url(%23n)' opacity='0.24'/%3E%3C/svg%3E");background-size:220px 220px;z-index:0}
.block-container{position:relative;z-index:1;max-width:1180px;padding-top:1rem;padding-bottom:5rem}[data-testid="stHeader"]{background:transparent}
.launch-nav-shell{position:sticky;top:.9rem;z-index:50;margin-bottom:1.1rem}.launch-nav{display:flex;align-items:center;justify-content:space-between;gap:1rem;padding:.84rem 1rem;border-radius:999px;background:linear-gradient(180deg,rgba(30,42,60,.68),rgba(49,63,86,.48));border:1px solid var(--line-dark);box-shadow:0 24px 48px rgba(17,26,39,.18),inset 0 1px 0 rgba(255,255,255,.12);backdrop-filter:blur(18px) saturate(135%)}
.launch-brand{color:var(--white);font-size:.88rem;font-weight:700;letter-spacing:-.03em;white-space:nowrap}.launch-brand span{color:rgba(255,255,255,.54);margin-left:.3rem;font-weight:500}
.launch-links{display:flex;align-items:center;justify-content:center;gap:.38rem;flex-wrap:wrap;flex:1}.launch-links a,.hero-actions a{text-decoration:none!important;transition:all .22s ease}
.launch-links a{display:inline-flex;align-items:center;justify-content:center;min-height:36px;padding:0 .92rem;border-radius:999px;border:1px solid rgba(255,255,255,.1);background:rgba(255,255,255,.02);color:rgba(255,255,255,.72);font-size:.8rem;font-weight:600}.launch-links a:hover{color:#fff;background:rgba(255,255,255,.1);transform:translateY(-1px)}
.launch-status{display:inline-flex;align-items:center;padding:.34rem .72rem;border-radius:999px;background:rgba(255,255,255,.08);color:rgba(255,255,255,.72);border:1px solid rgba(255,255,255,.12);font:.74rem "JetBrains Mono",monospace;white-space:nowrap}
.hero-stage,.showcase{opacity:.86;transform:translateY(16px);transition:opacity .82s ease,transform .92s cubic-bezier(.22,1,.36,1);scroll-margin-top:6rem}.hero-stage.is-visible,.showcase.is-visible{opacity:1;transform:none}
.hero-stage{position:relative;overflow:hidden;padding:4.8rem 3.2rem 3rem;margin-bottom:1.2rem;border-radius:40px;background:radial-gradient(circle at 78% 16%,rgba(255,255,255,.16),transparent 18%),radial-gradient(circle at 18% 0%,rgba(170,198,255,.18),transparent 24%),linear-gradient(155deg,rgba(27,37,53,.94) 0%,rgba(47,61,85,.84) 44%,rgba(109,130,159,.60) 100%);border:1px solid rgba(255,255,255,.14);box-shadow:0 36px 100px rgba(24,34,49,.24),inset 0 1px 0 rgba(255,255,255,.12)}
.eyebrow,.section-eyebrow{font-family:"JetBrains Mono",monospace;text-transform:uppercase;letter-spacing:.18em}.eyebrow{color:rgba(255,255,255,.52);font-size:.76rem;margin-bottom:1rem}
.hero-title{margin:0;max-width:860px;color:#fff;font-family:"Inter Tight","Inter",sans-serif;font-size:clamp(3.4rem,7vw,5.7rem);line-height:.92;letter-spacing:-.08em;font-weight:800}.hero-copy{max-width:680px;margin:1rem 0 0;color:rgba(255,255,255,.74);font-size:1.02rem;line-height:1.7}
.hero-actions{display:flex;gap:.75rem;flex-wrap:wrap;margin-top:1.65rem}.hero-actions a{display:inline-flex;align-items:center;justify-content:center;min-height:48px;padding:0 1.2rem;border-radius:999px;font-weight:700;letter-spacing:-.02em}.hero-actions .primary{background:#fff;color:#101114}.hero-actions .secondary{background:rgba(255,255,255,.07);color:#fff;border:1px solid rgba(255,255,255,.12)}
.hero-strip,.feature-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:.9rem}.hero-strip{margin-top:2.2rem}
.hero-chip,.metric-card,.feature-card{border-radius:28px;border:1px solid var(--line);box-shadow:inset 0 1px 0 rgba(255,255,255,.64),0 20px 40px rgba(15,17,21,.08);backdrop-filter:blur(18px)}
.hero-chip{padding:1rem;background:linear-gradient(180deg,rgba(255,255,255,.1),rgba(255,255,255,.03));border-color:rgba(255,255,255,.1);box-shadow:inset 0 1px 0 rgba(255,255,255,.08),0 16px 32px rgba(0,0,0,.16)}
.hero-chip-label,.metric-label,.feature-kicker,.result-label{font-size:.74rem;text-transform:uppercase;letter-spacing:.14em}.hero-chip-label{color:rgba(255,255,255,.5);margin-bottom:.42rem}.metric-label,.feature-kicker{color:rgba(16,17,20,.42)}.hero-chip-value{color:#fff;font-size:1.14rem;font-weight:700;letter-spacing:-.03em;margin-bottom:.2rem}.hero-chip-sub{color:rgba(255,255,255,.62);font-size:.84rem;line-height:1.45}
.section-anchor{height:0;scroll-margin-top:6rem}
.section-number{display:inline-flex;align-items:center;justify-content:center;min-width:54px;height:28px;margin-bottom:.9rem;padding:0 .78rem;border-radius:999px;background:rgba(255,255,255,.28);border:1px solid rgba(255,255,255,.38);color:rgba(15,20,28,.42);font:.72rem "JetBrains Mono",monospace}
.section-eyebrow{color:rgba(15,20,28,.42);font-size:.74rem;margin-bottom:.7rem}
.section-title{margin:0;max-width:660px;color:var(--ink);font-family:"Inter Tight","Inter",sans-serif;font-size:clamp(2rem,4.4vw,3.5rem);line-height:.95;letter-spacing:-.06em;font-weight:800}
.section-copy{max-width:560px;margin:.75rem 0 0;color:var(--ink-soft);font-size:.98rem;line-height:1.66}
.metric-card,.feature-card{padding:1.18rem;background:linear-gradient(180deg,rgba(255,255,255,.72),rgba(255,255,255,.46))}
.metric-value{color:var(--ink);font-size:clamp(1.5rem,3vw,2rem);line-height:1;font-weight:800;letter-spacing:-.06em;margin-bottom:.35rem}.metric-sub,.feature-copy{color:var(--ink-soft);font-size:.9rem;line-height:1.55}.feature-title{color:var(--ink);font-size:1.18rem;line-height:1.08;font-weight:700;letter-spacing:-.04em;margin-bottom:.48rem}
.callout{display:inline-flex;align-items:center;min-height:34px;padding:0 .78rem;border-radius:999px;background:rgba(255,255,255,.28);border:1px solid rgba(255,255,255,.36);color:rgba(15,20,28,.58);font-size:.78rem;font-weight:600;margin:.2rem 0 .85rem}
.result-label{color:rgba(15,20,28,.52);font-family:"JetBrains Mono",monospace;margin:.9rem 0 .5rem;display:block}
.subtle{color:var(--ink-faint);font-size:.84rem;line-height:1.55}.footer-note{text-align:center;color:rgba(15,17,21,.44);font-size:.82rem;line-height:1.6;margin-top:.2rem}
label[data-testid="stWidgetLabel"] p{color:rgba(15,17,21,.58);font-size:.84rem;font-weight:600}[data-baseweb="base-input"],[data-baseweb="input"],[data-baseweb="select"],textarea{border-radius:18px!important}
div[data-baseweb="base-input"]>div,div[data-baseweb="select"]>div,textarea{background:rgba(255,255,255,.72)!important;border:1px solid rgba(255,255,255,.62)!important;box-shadow:inset 0 1px 0 rgba(255,255,255,.76),0 16px 30px rgba(15,17,21,.06);backdrop-filter:blur(18px)} textarea,input{color:var(--ink)!important}
div[data-baseweb="tab-list"]{background:rgba(255,255,255,.34);padding:.28rem;border-radius:999px;border:1px solid rgba(255,255,255,.52);box-shadow:inset 0 1px 0 rgba(255,255,255,.42);width:fit-content;margin-bottom:.55rem}
button[data-baseweb="tab"]{height:36px;padding:0 1rem;border-radius:999px!important;background:transparent!important;color:rgba(15,17,21,.52)!important;font-size:.82rem!important;font-weight:700!important;transition:all .2s ease}button[data-baseweb="tab"][aria-selected="true"]{background:rgba(15,17,21,.94)!important;color:#fff!important;box-shadow:0 12px 24px rgba(0,0,0,.14)}
div[data-testid="stButton"] button{min-height:44px;border-radius:999px;border:1px solid rgba(255,255,255,.18);background:linear-gradient(180deg,#1d1f26 0%,#0f1115 100%);color:#fff;padding:0 1.1rem;font-weight:700;letter-spacing:-.01em;box-shadow:0 16px 32px rgba(0,0,0,.16),inset 0 1px 0 rgba(255,255,255,.1)}div[data-testid="stButton"] button:hover{background:linear-gradient(180deg,#14161b 0%,#07080a 100%);transform:translateY(-1px)}
.stCodeBlock,[data-testid="stCodeBlock"]{border-radius:22px!important;border:1px solid rgba(255,255,255,.08)!important;background:rgba(3,4,6,.22)!important}
::-webkit-scrollbar{width:12px;height:12px}::-webkit-scrollbar-track{background:rgba(255,255,255,.08)}::-webkit-scrollbar-thumb{background:rgba(15,17,21,.24);border-radius:999px;border:2px solid rgba(255,255,255,.28)}
@media (max-width:900px){.launch-nav{border-radius:28px;align-items:flex-start}.launch-links{justify-content:flex-start}.launch-status{display:none}.hero-stage{padding:4rem 1.5rem 2.3rem}.showcase{padding:1.45rem;border-radius:28px}.hero-strip,.feature-grid{grid-template-columns:1fr}}
</style>
"""


STATE_DEFAULTS: dict[str, Any] = {
    "agent_result": None,
    "agent_error": None,
    "safety_result": None,
    "reaction_result": None,
    "retro_result": None,
    "chromophore_result": None,
    "benchmark_result": None,
}


def init_state() -> None:
    for key, default in STATE_DEFAULTS.items():
        st.session_state.setdefault(key, default)


@st.cache_data(show_spinner=False, ttl=300)
def cached_health_snapshot(provider_name: str, model: str, base_url: str) -> dict[str, Any]:
    del provider_name, model, base_url
    return run_healthcheck(Settings.from_env())


def inject_motion() -> None:
    components.html(
        """
        <script>
        const doc = window.parent.document;
        const root = doc.documentElement;
        root.style.scrollBehavior = "smooth";
        const app = doc.querySelector('[data-testid="stAppViewContainer"]');
        if (app) app.style.scrollBehavior = "smooth";
        doc.querySelectorAll('a[href^="#"]').forEach((link) => {
            if (link.dataset.bound === "1") return;
            link.dataset.bound = "1";
            link.addEventListener("click", (event) => {
                const target = doc.querySelector(link.getAttribute("href"));
                if (!target) return;
                event.preventDefault();
                target.scrollIntoView({ behavior: "smooth", block: "start" });
            });
        });
        const observer = new IntersectionObserver((entries) => {
            entries.forEach((entry) => { if (entry.isIntersecting) entry.target.classList.add("is-visible"); });
        }, { threshold: 0.18 });
        doc.querySelectorAll('.hero-stage').forEach((node) => observer.observe(node));
        </script>
        """,
        height=0,
    )


def begin_showcase(anchor: str, tone: str = "") -> None:
    del tone
    st.markdown(f'<div id="{anchor}" class="section-anchor"></div>', unsafe_allow_html=True)


def end_showcase() -> None:
    st.markdown('<div style="height:1.6rem"></div>', unsafe_allow_html=True)


def render_section_intro(index: str, eyebrow: str, title: str, copy: str) -> None:
    st.markdown(f'<div class="section-number">{escape(index)}</div><div class="section-eyebrow">{escape(eyebrow)}</div><h2 class="section-title">{escape(title)}</h2><p class="section-copy">{escape(copy)}</p>', unsafe_allow_html=True)


def render_metric_card(label: str, value: str, sub: str) -> None:
    st.markdown(f'<div class="metric-card"><div class="metric-label">{escape(label)}</div><div class="metric-value">{escape(value)}</div><div class="metric-sub">{escape(sub)}</div></div>', unsafe_allow_html=True)


def render_feature_grid(cards: list[tuple[str, str, str]]) -> None:
    markup = "".join(f'<div class="feature-card"><div class="feature-kicker">{escape(k)}</div><div class="feature-title">{escape(t)}</div><div class="feature-copy">{escape(c)}</div></div>' for k, t, c in cards)
    st.markdown(f'<div class="feature-grid">{markup}</div>', unsafe_allow_html=True)


def render_result_block(title: str, payload: Any) -> None:
    if payload is None:
        st.markdown('<div class="subtle">No output yet.</div>', unsafe_allow_html=True)
        return
    st.markdown(f'<div class="result-label">{escape(title)}</div>', unsafe_allow_html=True)
    if isinstance(payload, (dict, list)):
        st.json(payload, expanded=1)
        return
    rendered = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False, indent=2)
    language = "markdown" if isinstance(payload, str) else "json"
    st.code(rendered, language=language, wrap_lines=True, height=260)


def format_runtime_error(exc: Exception, provider_name: str) -> str:
    if isinstance(exc, APIStatusError):
        status_code = getattr(exc, "status_code", None)
        if status_code == 401:
            return f"{provider_name} API authentication failed (401). Check the API key and model permission."
        if status_code == 429:
            return f"{provider_name} API rate limit or quota issue (429). Check quota, retry later, or switch provider."
        return f"{provider_name} API returned status {status_code or 'unknown'}."
    return str(exc)


def list_reports(audit_dir: Path) -> list[Path]:
    return sorted(audit_dir.glob("benchmark_*.md"), key=lambda path: path.stat().st_mtime, reverse=True)

init_state()
settings = Settings.from_env()
health = cached_health_snapshot(settings.provider_name, settings.model, settings.llm_base_url)
reports = list_reports(settings.audit_dir)
latest_report = reports[0].name if reports else "No benchmark report yet"
semantic_status = health.get("semantic_scholar", {}).get("status", "n/a")

st.markdown(STYLE, unsafe_allow_html=True)
inject_motion()

st.markdown(
    f"""
    <div class="launch-nav-shell">
        <div class="launch-nav">
            <div class="launch-brand">ChemCrow-Lite <span>thesis prototype</span></div>
            <div class="launch-links">
                <a href="#overview">Overview</a>
                <a href="#agent">Agent</a>
                <a href="#safety">Safety</a>
                <a href="#reaction">Reaction</a>
                <a href="#chromophore">Chromophore</a>
                <a href="#benchmarks">Benchmarks</a>
            </div>
            <div class="launch-status">Live · {escape(settings.provider_name.upper())}</div>
        </div>
    </div>
    <section id="top" class="hero-stage is-visible">
        <div class="eyebrow">ChemCrow-Lite / keynote mode</div>
        <h1 class="hero-title">Safer chemistry.<br/>Staged with intent.</h1>
        <p class="hero-copy">A paper-aligned chemistry agent with explicit tool use, deterministic guardrails, auditable traces, and a lightweight chromophore workflow for MSc-level demos.</p>
        <div class="hero-actions">
            <a class="primary" href="#agent">Launch the demo</a>
            <a class="secondary" href="#benchmarks">View benchmarks</a>
        </div>
        <div class="hero-strip">
            <div class="hero-chip"><div class="hero-chip-label">Provider</div><div class="hero-chip-value">{escape(settings.provider_name.upper())}</div><div class="hero-chip-sub">{escape(settings.model)}</div></div>
            <div class="hero-chip"><div class="hero-chip-label">Health</div><div class="hero-chip-value">{escape(health['overall_status'].upper())}</div><div class="hero-chip-sub">PubChem {escape(health['pubchem']['status'])} · Semantic Scholar {escape(semantic_status)}</div></div>
            <div class="hero-chip"><div class="hero-chip-label">Latest report</div><div class="hero-chip-value">{escape(latest_report)}</div><div class="hero-chip-sub">Audit logs stay local and reviewable.</div></div>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)

begin_showcase("overview", "showcase-dark")
overview_left, overview_right = st.columns([1.05, 0.95], gap="large")
with overview_left:
    render_section_intro("00", "Overview", "Not a toy demo. A thesis-grade system sketch.", "The page now reads like a launch story: fewer tabs, stronger hierarchy, smoother motion, tighter copy.")
    st.markdown('<div class="callout">Paper spirit · tools, safety, audits, evaluation</div>', unsafe_allow_html=True)
with overview_right:
    render_feature_grid([
        ("Reasoning", "Tool calls stay visible.", "Agent answers remain concise, sourced by tools, and easy to audit."),
        ("Safety", "Refusal is deterministic.", "High-risk prompts trigger hard refusal or high-level-only behavior."),
        ("Workflow", "Chromophore ML stays practical.", "A compact screening pipeline mirrors the paper without huge datasets."),
    ])

m1, m2, m3 = st.columns(3, gap="large")
with m1:
    render_metric_card("Provider", settings.provider_name.upper(), f"Model · {settings.model}")
with m2:
    render_metric_card("System", health["overall_status"].upper(), f"LLM · {health['llm_provider']['status']}  |  PubChem · {health['pubchem']['status']}")
with m3:
    render_metric_card("Data", "READY", f"Chromophore set + task suite  |  Audit · {settings.audit_dir.name}")
end_showcase()

begin_showcase("agent")
a_intro, a_ui = st.columns([0.88, 1.12], gap="large")
with a_intro:
    render_section_intro("01", "Agent", "Ask. Then verify.", "This is the ChemCrow move: let the model decide when tools matter, then keep the chain auditable.")
    render_feature_grid([
        ("Mode", "Tool-Augmented", "Matches the paper’s spirit: model plus specialist tools."),
        ("Baseline", "No-Tools", "Useful for ablation and benchmark contrast."),
        ("Audit", "Every run logs steps.", "Answers, traces, and paths remain reviewable after the demo."),
    ])
with a_ui:
    prompt = st.text_area("Prompt", key="agent_prompt", value="What is the exact molecular weight of aspirin? Use tools if helpful.", height=150)
    mode = st.segmented_control("Mode", options=["Tool-Augmented", "No-Tools Baseline"], default="Tool-Augmented", key="agent_mode")
    if st.button("Run Agent", key="run_agent"):
        with st.spinner("Running chemistry agent..."):
            try:
                agent = ChemCrowLiteAgent(settings, use_tools=mode == "Tool-Augmented", system_prompt=SYSTEM_PROMPT if mode == "Tool-Augmented" else BASELINE_SYSTEM_PROMPT)
                result = agent.run(prompt)
                st.session_state["agent_result"] = {"answer": result.answer, "audit_path": str(result.audit_path), "steps": result.steps, "mode": mode}
                st.session_state["agent_error"] = None
            except Exception as exc:
                st.session_state["agent_error"] = format_runtime_error(exc, settings.provider_name.upper())
    if st.session_state["agent_error"]:
        st.error(st.session_state["agent_error"])
    render_result_block("Latest agent output", st.session_state["agent_result"])
end_showcase()

begin_showcase("safety")
s_intro, s_ui = st.columns([0.92, 1.08], gap="large")
with s_intro:
    render_section_intro("02", "Safety", "Refuse with structure.", "A strict MSc demo should never bluff its way through risky chemistry. Safety logic comes first.")
    render_feature_grid([
        ("Registry", "Controlled aliases included.", "Prompts match on chemical names and common variants."),
        ("Policy", "Operational detail gets stripped.", "Responses stay high-level near dangerous procedure."),
        ("UX", "Check risk before action.", "Classification is surfaced cleanly instead of buried in the answer."),
    ])
with s_ui:
    safety_query = st.text_input("Compound", key="safety_query", value="acetic anhydride")
    if st.button("Run Safety Check", key="run_safety"):
        st.session_state["safety_result"] = {
            "control_check": control_chem_check(safety_query, settings.controlled_chemicals_path),
            "explosive_check": explosive_check(safety_query),
            "safety_summary": safety_summary(safety_query, settings.controlled_chemicals_path),
        }
    render_result_block("Latest safety output", st.session_state["safety_result"])
end_showcase()

begin_showcase("reaction")
r_intro, r_ui = st.columns([0.92, 1.08], gap="large")
with r_intro:
    render_section_intro("03", "Reaction", "Reason conservatively.", "Instead of pretending to be a full synthesis engine, the demo explains what can and cannot be inferred from sparse reaction context.")
    render_feature_grid([
        ("Outcome", "Heuristic, not hallucinated.", "Underspecified inputs lead to caveats rather than fake precision."),
        ("Retro", "Route families only.", "The overview stays strategic and non-operational by design."),
        ("Safety", "Conditions matter.", "Missing gases or catalysts are explicitly called out."),
    ])
with r_ui:
    substrate_text = st.text_input("Substrates", key="reaction_substrates", value="1-Chloro-4-ethynylbenzene")
    reagent_text = st.text_input("Reagents / catalysts", key="reaction_reagents", value="Lindlar catalyst")
    condition_text = st.text_input("Conditions", key="reaction_conditions", value="no hydrogen gas, inert atmosphere")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Run Heuristic", key="run_reaction"):
            substrates = [item.strip() for item in substrate_text.split(",") if item.strip()]
            reagents = [item.strip() for item in reagent_text.split(",") if item.strip()]
            st.session_state["reaction_result"] = reaction_outcome_heuristic(substrates=substrates, reagents=reagents, conditions=condition_text or None)
    with c2:
        retro_target = st.text_input("Retrosynthesis target", key="retro_target", value="aspirin")
        if st.button("Generate Route Families", key="run_retro"):
            st.session_state["retro_result"] = retrosynthesis_overview(target=retro_target, controlled_chemicals_path=str(settings.controlled_chemicals_path))
    result_tab_1, result_tab_2 = st.tabs(["Outcome heuristic", "Retrosynthesis overview"])
    with result_tab_1:
        render_result_block("Outcome heuristic", st.session_state["reaction_result"])
    with result_tab_2:
        render_result_block("Retrosynthesis overview", st.session_state["retro_result"])
end_showcase()

begin_showcase("chromophore")
c_intro, c_ui = st.columns([0.9, 1.1], gap="large")
with c_intro:
    render_section_intro("04", "Chromophore", "Stay small. Still learn the idea.", "You asked for the paper’s essence without heavyweight downloads. This workflow keeps the ML tractable and thesis-friendly.")
    render_feature_grid([
        ("Data", "Use the local Excel set.", "No oversized crawl is needed for the core chromophore exercise."),
        ("Model", "Random forest screening.", "Simple, interpretable, and aligned with the lightweight replication goal."),
        ("Output", "Rank candidate molecules.", "Good enough to discuss methods, limits, and future extensions."),
    ])
with c_ui:
    target_nm = st.slider("Target absorption (nm)", min_value=300, max_value=700, value=369, key="target_nm")
    top_k = st.slider("Top candidates", min_value=1, max_value=10, value=5, key="top_k")
    if st.button("Run Chromophore Screen", key="run_chromophore"):
        st.session_state["chromophore_result"] = chromophore_rf_screen(data_path=settings.chromophore_data_path, target_nm=float(target_nm), top_k=int(top_k))
    render_result_block("Latest chromophore output", st.session_state["chromophore_result"])
end_showcase()

begin_showcase("benchmarks", "showcase-dark")
b_intro, b_ui = st.columns([0.9, 1.1], gap="large")
with b_intro:
    render_section_intro("05", "Benchmarks", "Measure the gap.", "The benchmark view keeps the paper-style comparison visible: tool use versus no tools, plus auditable markdown reports.")
    render_feature_grid([
        ("Compare", "Tool vs baseline.", "Show where external tools actually change the answer quality."),
        ("Review", "Markdown reports persist.", "Useful for advisor review, appendix figures, and experiment logs."),
        ("Scope", "Compact task suite.", "Small enough to iterate fast, strict enough to stay honest."),
    ])
with b_ui:
    bc1, bc2 = st.columns(2)
    with bc1:
        bench_limit = st.slider("Tasks", min_value=1, max_value=5, value=5, key="bench_limit")
    with bc2:
        bench_mode = st.segmented_control("Benchmark mode", options=["compare", "agent", "no_tools"], default="compare", key="bench_mode")
    if st.button("Run Benchmark", key="run_benchmark"):
        with st.spinner("Benchmark in progress..."):
            st.session_state["benchmark_result"] = run_benchmark(settings, limit=int(bench_limit), mode=str(bench_mode))
    render_result_block("Latest benchmark output", st.session_state["benchmark_result"])
    reports = list_reports(settings.audit_dir)
    if reports:
        selected_report = st.selectbox("Saved markdown reports", options=reports, format_func=lambda path: path.name, key="saved_reports")
        with st.expander("Open markdown report", expanded=False):
            render_result_block("Report preview", selected_report.read_text(encoding="utf-8"))
    else:
        st.markdown('<div class="subtle">No markdown report exists yet.</div>', unsafe_allow_html=True)
end_showcase()

st.markdown('<div class="footer-note">ChemCrow-Lite keeps the thesis story tight: tool orchestration, safety discipline, auditable experiments, and a lightweight paper-aligned chemistry workflow.</div>', unsafe_allow_html=True)
