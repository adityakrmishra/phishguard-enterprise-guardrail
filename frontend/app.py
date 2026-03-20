# Streamlit B2B compliance dashboard; sends text to the FastAPI analyze endpoint and displays verdict, metrics, and audit log.

import time
from datetime import datetime

import requests
import streamlit as st

st.set_page_config(
    page_title="PhishGuard Enterprise Guardrail",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .stApp { background-color: #0d1117; }

        .pg-header {
            background: linear-gradient(90deg, #0f2027, #203a43, #2c5364);
            border-radius: 12px;
            padding: 1.4rem 2rem;
            margin-bottom: 1.5rem;
        }
        .pg-header h1 {
            color: #e6edf3;
            font-size: 1.8rem;
            margin: 0;
            font-family: 'Segoe UI', sans-serif;
            letter-spacing: 0.02em;
        }
        .pg-header p {
            color: #8b949e;
            margin: 0.2rem 0 0 0;
            font-size: 0.88rem;
        }

        .input-card {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 10px;
            padding: 1.4rem 1.6rem;
            margin-bottom: 1rem;
        }

        .result-card {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 10px;
            padding: 1.4rem 1.6rem;
        }

        section[data-testid="stSidebar"] {
            background-color: #0d1117;
            border-right: 1px solid #21262d;
        }
        .sidebar-badge {
            display: inline-block;
            padding: 2px 10px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
        }
        .badge-green  { background: #1a4731; color: #3fb950; }
        .badge-yellow { background: #3d2f0b; color: #e3b341; }

        [data-testid="stMetricLabel"] { color: #8b949e !important; }
        [data-testid="stMetricValue"] { color: #e6edf3 !important; }

        hr { border-color: #21262d; }

        div.stButton > button {
            background: linear-gradient(135deg, #1f6feb, #388bfd);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.55rem 2rem;
            font-weight: 600;
            transition: opacity 0.2s;
        }
        div.stButton > button:hover { opacity: 0.85; }
    </style>
    """,
    unsafe_allow_html=True,
)

API_URL = "https://piddling-chandler-decadently.ngrok-free.dev/api/v1/analyze"

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.markdown("### PhishGuard")
    st.markdown(
        '<span class="sidebar-badge badge-green">● SYSTEM ONLINE</span>',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown("#### System Health")

    st.metric("API Latency", "12 ms", delta="-3 ms", delta_color="normal")
    st.metric("Uptime", "99.9 %", delta="0.1 %", delta_color="normal")
    st.metric("FAISS Index Size", "14 vectors")
    st.metric("Model", "MiniLM-L6-v2")
    st.metric("Requests (session)", len(st.session_state.history))

    st.markdown("---")
    st.markdown("#### Triage Thresholds")
    safe_threshold = st.slider("Safe threshold (L2)", 0.1, 1.0, 0.35, 0.05)
    scam_threshold = st.slider("Scam threshold (L2)", 0.1, 2.0, 0.75, 0.05)

    st.markdown("---")
    st.caption(f"Last refresh: {datetime.now().strftime('%H:%M:%S')}")
    st.caption("PhishGuard v0.2.0 · Internal Demo")

st.markdown(
    """
    <div class="pg-header">
        <h1>🛡️ PhishGuard Enterprise Guardrail</h1>
        <p>AI-powered semantic triage · Real-time phishing detection · B2B Compliance Dashboard</p>
    </div>
    """,
    unsafe_allow_html=True,
)

total     = len(st.session_state.history)
safe_n    = sum(1 for r in st.session_state.history if r["verdict"] == "SAFE")
scam_n    = sum(1 for r in st.session_state.history if r["verdict"] == "KNOWN_SCAM")
anomaly_n = sum(1 for r in st.session_state.history if r["verdict"] == "ANOMALY_NEEDS_LLM")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Scanned", total)
k2.metric("Safe", safe_n)
k3.metric("Known Scam", scam_n)
k4.metric("Anomalies", anomaly_n)

st.markdown("---")

left, right = st.columns([1.15, 1], gap="large")

with left:
    st.markdown("#### Submit Content for Analysis")
    with st.form("analyze_form", clear_on_submit=False):
        user_text = st.text_area(
            label="Paste URL, email body, or message content",
            placeholder="e.g.  URGENT: Your account has been suspended. Click to verify: bit.ly/secure-upd8",
            height=160,
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Analyse Now", use_container_width=True)

    st.markdown("**Quick test samples:**")
    sample_cols = st.columns(3)
    if sample_cols[0].button("Safe sample", use_container_width=True):
        user_text = "Your account statement for March 2024 is now available online."
        submitted = True
    if sample_cols[1].button("Scam sample", use_container_width=True):
        user_text = "URGENT: Click here to unlock your suspended bank account immediately!"
        submitted = True
    if sample_cols[2].button("Anomaly sample", use_container_width=True):
        user_text = "The quarterly GDP figures indicate a potential 3.2% economic contraction."
        submitted = True

with right:
    st.markdown("#### Analysis Result")

    if submitted and user_text.strip():
        with st.spinner("Routing through semantic triage..."):
            t0 = time.perf_counter()
            try:
                resp = requests.post(
                    API_URL,
                    json={"text": user_text.strip()},
                    timeout=30,
                )
                latency_ms = (time.perf_counter() - t0) * 1000
                resp.raise_for_status()
                data = resp.json()

                verdict              = data.get("verdict", "UNKNOWN")
                distance             = data.get("distance", 0.0)
                matched              = data.get("matched_template", "-")
                fraud_intent         = data.get("fraud_intent")
                compliance_reasoning = data.get("compliance_reasoning")

                st.session_state.history.append(
                    {"verdict": verdict, "distance": distance, "text": user_text[:80]}
                )

                if verdict == "SAFE":
                    st.success("✅ **SAFE** — Content appears legitimate.")
                elif verdict == "KNOWN_SCAM":
                    st.error("🚨 **KNOWN SCAM** — Pattern matches a catalogued phishing template.")
                else:
                    st.warning("⚠️ **ANOMALY** — Content is unusual. Escalating to LLM review.")

                m1, m2 = st.columns(2)
                m1.metric("Verdict",     verdict)
                m2.metric("L2 Distance", f"{distance:.4f}")

                m3, m4 = st.columns(2)
                m3.metric("Latency",    f"{latency_ms:.0f} ms")
                m4.metric("Confidence", "High" if distance < 0.4 else "Medium" if distance < 0.75 else "Low")

                st.markdown("**Nearest indexed template:**")
                st.info(f'"{matched}"')

                if verdict == "KNOWN_SCAM" and (fraud_intent or compliance_reasoning):
                    with st.expander("Audit Log", expanded=True):
                        st.markdown(
                            """
                            <div style="background:#161b22;border:1px solid #30363d;
                                        border-radius:8px;padding:1rem 1.2rem;">
                            """,
                            unsafe_allow_html=True,
                        )
                        st.markdown("##### Compliance Intelligence Report")
                        st.markdown("---")

                        col_a, col_b = st.columns([1, 2])
                        with col_a:
                            st.markdown("**Fraud Intent**")
                            st.markdown(
                                f'<span style="background:#3d0b0b;color:#f85149;'
                                f'padding:3px 10px;border-radius:20px;font-size:0.82rem;'
                                f'font-weight:600;">{fraud_intent or "Unknown"}</span>',
                                unsafe_allow_html=True,
                            )
                        with col_b:
                            st.markdown("**Compliance Reasoning**")
                            st.markdown(
                                f'<span style="color:#c9d1d9;font-size:0.9rem;">'
                                f'{compliance_reasoning or "Not available."}</span>',
                                unsafe_allow_html=True,
                            )

                        st.markdown("</div>", unsafe_allow_html=True)
                        st.caption(
                            "This report is generated for audit and compliance purposes only. "
                            "Powered by PhishGuard AI · RBI / NPCI AML Guidelines."
                        )

            except requests.exceptions.ConnectionError:
                st.error("Cannot reach the FastAPI backend at `localhost:8000`. Is the server running?")
            except requests.exceptions.Timeout:
                st.error("Request timed out after 30 s.")
            except requests.exceptions.HTTPError as exc:
                st.error(f"API error {exc.response.status_code}: {exc.response.text[:200]}")

    elif submitted and not user_text.strip():
        st.warning("Please enter some text before submitting.")
    else:
        st.markdown(
            """
            <div style="color:#8b949e; padding: 2rem 1rem; text-align:center; border: 1px dashed #30363d; border-radius:8px; margin-top:0.5rem;">
                Submit a text snippet on the left<br>to see the triage verdict here.
            </div>
            """,
            unsafe_allow_html=True,
        )

if st.session_state.history:
    st.markdown("---")
    st.markdown("#### Session History")

    ICONS = {"SAFE": "✅", "KNOWN_SCAM": "🚨", "ANOMALY_NEEDS_LLM": "⚠️"}
    rows = []
    for i, item in enumerate(reversed(st.session_state.history), 1):
        rows.append({
            "#": i,
            "Verdict": f"{ICONS.get(item['verdict'], '?')} {item['verdict']}",
            "L2 Distance": f"{item['distance']:.4f}",
            "Input (truncated)": item["text"],
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)

    if st.button("Clear history"):
        st.session_state.history = []
        st.rerun()
