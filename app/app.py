"""Streamlit interface for the IT Support AI demonstration platform."""

import hashlib
import html
import io
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bulk_processing import BulkInputError, bulk_frame_from_records, process_bulk_frame
from src.demo_tickets import DEMO_TICKET_BODIES, clear_demo_tickets, load_demo_tickets
from src.service_factory import create_application_service, resolve_database_path


st.set_page_config(
    page_title="IT Support AI",
    page_icon="✳",
    layout="wide",
    initial_sidebar_state="auto",
)

st.markdown(
    """
    <style>
    :root {
        --canvas:#FFF9E8; --paper:#FFFFFF; --ink:#171717; --muted:#6B625D;
        --line:#E8DED2; --lav:#FFF0E8; --pink:#FFF0E8; --sky:#FFF4EA;
        --mint:#FFF4EA; --orange:#FFF0E8; --yellow:#FFF5DF; --accent:#F45F63;
        --shadow:0 4px 14px rgba(56,38,24,.035); --radius:15px;
    }
    html, body, [data-testid="stAppViewContainer"], .stApp {
        background:var(--canvas); color:var(--ink);
        font-family:Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
                     "Segoe UI", sans-serif;
    }
    [data-testid="stAppViewContainer"] { background:var(--canvas); }
    [data-testid="stHeader"] { background:rgba(255,249,232,.96); height:2.7rem; }
    [data-testid="stToolbar"] { visibility:visible; background:transparent; }
    [data-testid="stToolbar"] button:not([data-testid="stExpandSidebarButton"]) { display:none; }
    footer, #MainMenu { visibility:hidden; height:0; }
    .block-container { max-width:1320px; padding-top:3rem; padding-bottom:3rem; }
    [data-testid="stSidebar"] { background:#FFF5E8; border-right:1px solid #E8DED2;
        min-width:245px; max-width:245px; }
    [data-testid="stSidebar"] > div:first-child { padding:1.15rem .85rem 1rem; }
    [data-testid="stExpandSidebarButton"] button,
    [data-testid="stSidebarCollapseButton"] button {
        color:#777B85; background:transparent; border:0; box-shadow:none;
    }
    .brand { padding:.55rem .65rem 1.45rem; }
    .brand-mark { display:inline-flex; width:31px; height:31px; align-items:center;
        justify-content:center; border-radius:10px; background:#FFE3D8; color:#C9444B;
        font-weight:750; margin-bottom:.75rem; }
    .brand-name { color:var(--ink); font-size:.96rem; font-weight:730; letter-spacing:-.025em; }
    .brand-subtitle { color:var(--muted); font-size:.74rem; margin-top:.28rem; }
    .nav-section { color:#858894; font-size:.64rem; font-weight:730; letter-spacing:.13em;
        padding:.7rem .72rem .35rem; }
    [data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] { gap:.11rem; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        border:1px solid transparent; border-left:2px solid transparent; border-radius:8px;
        margin:0; padding:.43rem .58rem; background:transparent;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover { background:#ECEAE7; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
        background:#FFFFFF; border-color:#E8DED2; border-left-color:#F45F63;
        box-shadow:0 2px 7px rgba(32,35,45,.035);
    }
    [data-testid="stSidebar"] label[data-testid="stRadioOption"] > div > div:first-child { display:none; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label p {
        color:#555B66; font-size:.83rem; margin:0;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) p {
        color:var(--ink); font-weight:630;
    }
    .sidebar-footer { border-top:1px solid #E3E0DC; margin-top:2rem; padding:1rem .68rem 0;
        color:#626975; font-size:.75rem; }
    .status-dot { color:#54A77B; padding-right:.35rem; }
    .eyebrow { color:#C9444B; font-size:.67rem; font-weight:740; letter-spacing:.13em;
        margin:0 0 .7rem; text-transform:uppercase; line-height:1.35; }
    .page-title { color:var(--ink); font-size:clamp(1.8rem, 3vw, 2.35rem);
        font-weight:710; letter-spacing:-.05em; line-height:1.13; margin:0; }
    .page-subtitle { color:var(--muted); font-size:.94rem; line-height:1.55;
        margin:.55rem 0 0; }
    .page-head { display:block; position:relative; margin:0 0 1.65rem; padding-top:.1rem; }
    .section-heading { color:#777B86; font-size:.68rem; font-weight:730;
        letter-spacing:.105em; margin:1.55rem 0 .7rem; text-transform:uppercase; }
    .kpi-card { min-height:116px; padding:1rem 1.1rem; border:1px solid rgba(30,35,50,.045);
        border-radius:var(--radius); box-shadow:var(--shadow); }
    .kpi-label { color:#656B77; font-size:.79rem; font-weight:570; }
    .kpi-value { color:var(--ink); font-size:2rem; font-weight:720; letter-spacing:-.055em;
        margin-top:.45rem; line-height:1; }
    .card { background:var(--paper); border:1px solid var(--line); border-radius:var(--radius);
        box-shadow:var(--shadow); padding:1.15rem 1.25rem; }
    .empty-state { background:#FFF3E9;
        border:1px solid #E8DED2; border-radius:18px; padding:1.55rem 1.65rem;
        box-shadow:var(--shadow); }
    .empty-title { color:var(--ink); font-size:1.12rem; font-weight:670; margin-bottom:.32rem; }
    .muted { color:var(--muted); font-size:.84rem; line-height:1.5; }
    h2, h3 { color:var(--ink); letter-spacing:-.025em; }
    h2 { font-size:1.08rem !important; font-weight:670 !important; }
    h3 { font-size:.96rem !important; font-weight:640 !important; }
    p, li { color:#464C58; }
    [data-testid="stTextArea"] label p, [data-testid="stSelectbox"] label p,
    [data-testid="stFileUploader"] label p { color:#424854; font-size:.85rem; font-weight:610; }
    [data-testid="stTextArea"] textarea { background:#FFF !important; border:1px solid #E2E2E8 !important;
        border-radius:11px !important; color:var(--ink) !important; line-height:1.6;
        padding:1rem !important; }
    [data-testid="stTextArea"] textarea:focus { border-color:#A49BEA !important;
        box-shadow:0 0 0 2px rgba(130,119,220,.14) !important; }
    [data-testid="stTextArea"] textarea::placeholder { color:#969BA5 !important; }
    [data-testid="stForm"] { border:0; padding:0; }
    .stButton button, [data-testid="stFormSubmitButton"] button,
    [data-testid="stDownloadButton"] button {
        border-radius:9px; font-size:.84rem; font-weight:650; min-height:2.45rem;
        padding:.34rem 1rem; transition:background 120ms ease, transform 120ms ease;
    }
    .stButton button[kind="primary"], [data-testid="stFormSubmitButton"] button[kind="primary"] {
        background:#F45F63; border:1px solid #F45F63; color:#FFF;
    }
    .stButton button[kind="primary"]:hover,
    [data-testid="stFormSubmitButton"] button[kind="primary"]:hover { background:#D94D53; color:#FFF; }
    button[data-testid="stBaseButton-primaryFormSubmit"],
    button[data-testid="stBaseButton-primary"] {
        background:#F45F63 !important; border:1px solid #F45F63 !important;
        color:#FFFFFF !important;
    }
    button[data-testid="stBaseButton-primaryFormSubmit"] *,
    button[data-testid="stBaseButton-primary"] * { color:#FFFFFF !important; }
    button[data-testid="stBaseButton-primaryFormSubmit"]:hover,
    button[data-testid="stBaseButton-primary"]:hover {
        background:#D94D53 !important; border-color:#D94D53 !important;
    }
    .stButton button:not([kind="primary"]), [data-testid="stDownloadButton"] button {
        background:#FFF; border:1px solid #E2E2E8; color:#343945;
    }
    .stButton button:hover, [data-testid="stDownloadButton"] button:hover { transform:translateY(-1px); }
    [data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:11px; overflow:hidden; }
    [data-testid="stVerticalBlockBorderWrapper"] { background:#FFF; border-color:var(--line);
        border-radius:var(--radius); box-shadow:var(--shadow); }
    [data-testid="stSelectbox"] > div > div, [data-testid="stFileUploader"] section {
        background:#FFF; border-color:#E2E2E8; border-radius:10px;
    }
    [data-testid="stFileUploader"] button {
        background:#FFF0E8 !important; border:1px solid #E8DED2 !important;
        color:#B94449 !important; border-radius:8px !important;
    }
    [data-testid="stFileUploader"] button * { color:#B94449 !important; }
    [data-testid="stAlert"] { background:#FFF; border:1px solid var(--line); border-radius:10px; }
    [data-testid="stProgressBar"] > div > div { background:#F45F63; }
    hr { border-color:var(--line) !important; }
    .result-card { height:100%; min-height:105px; background:#FFF; border:1px solid var(--line);
        border-radius:13px; box-shadow:var(--shadow); padding:1.05rem 1.1rem; }
    .result-label { color:#777D88; font-size:.67rem; font-weight:740; letter-spacing:.1em;
        margin-bottom:.62rem; text-transform:uppercase; }
    .result-value { color:var(--ink); font-size:1.18rem; font-weight:700; }
    .badge { display:inline-block; padding:.26rem .58rem; border-radius:999px;
        font-size:.72rem; font-weight:670; white-space:nowrap; }
    .badge-high { background:#FCE7EE; color:#98506A; }
    .badge-medium { background:#FFF2D8; color:#8B6B2D; }
    .badge-low { background:#E7F5EC; color:#397453; }
    .badge-within { background:#E7F5EC; color:#397453; }
    .badge-risk { background:#FFF0DF; color:#94622E; }
    .badge-breached { background:#FBE8E8; color:#A15353; }
    .flow-step { background:#FFF; border:1px solid var(--line); border-radius:10px;
        color:#555B66; padding:.72rem .55rem; text-align:center; font-size:.8rem; font-weight:600; }
    .flow-done { background:#FFF0E8; border-color:#E8DED2; color:#B94449; }
    .model-card { background:#FFF; border:1px solid var(--line); border-radius:15px;
        box-shadow:var(--shadow); padding:1.2rem 1.25rem; height:100%; }
    .model-accent { height:5px; border-radius:5px; margin:-.2rem 0 1rem; }
    .model-name { color:var(--ink); font-size:1.03rem; font-weight:680; }
    .model-type { color:var(--muted); font-size:.82rem; margin-top:.23rem; }
    .model-score { color:var(--ink); font-size:1.5rem; font-weight:710; letter-spacing:-.04em; }
    .model-score-label { color:var(--muted); font-size:.72rem; margin-top:.1rem; }
    .model-scores { display:grid; grid-template-columns:1fr 1fr; gap:1rem; margin-top:1.1rem; }
    .ticket-row { background:#FFF; border:1px solid var(--line); border-radius:11px;
        margin-bottom:.48rem; padding:.77rem .85rem; }
    @media (max-width:850px) {
        .block-container { padding:1.45rem 1rem 2rem; }
        [data-testid="stSidebar"] { min-width:220px; max-width:220px; }
        .page-head { margin-bottom:1.25rem; }
        .kpi-card { min-height:100px; padding:.85rem; }
        .kpi-value { font-size:1.65rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_application_service(database_path=""):
    """Share model and database handles across Streamlit reruns."""
    return create_application_service(database_path or None)


def _utc_now():
    return datetime.now(timezone.utc)


def _safe_html(value):
    return html.escape(str(value))


def _page_header(eyebrow, title, subtitle):
    st.markdown(
        '<div class="page-head"><div class="eyebrow">{0}</div>'
        '<h1 class="page-title">{1}</h1><p class="page-subtitle">{2}</p></div>'.format(
            _safe_html(eyebrow), _safe_html(title), _safe_html(subtitle)
        ),
        unsafe_allow_html=True,
    )


def _kpi_card(label, value, background):
    st.markdown(
        '<div class="kpi-card" style="background:{0}">'
        '<div class="kpi-label">{1}</div><div class="kpi-value">{2}</div></div>'.format(
            background, _safe_html(label), _safe_html(value)
        ),
        unsafe_allow_html=True,
    )


def _badge(value, kind):
    return '<span class="badge badge-{0}">{1}</span>'.format(kind, _safe_html(value))


def _priority_kind(priority):
    return {"high": "high", "medium": "medium", "low": "low"}.get(
        str(priority).casefold(), "low"
    )


def _sla_kind(status):
    return {"Within SLA": "within", "At Risk": "risk", "Breached": "breached"}.get(
        status, "within"
    )


def _relative_time(timestamp):
    try:
        value = timestamp[:-1] + "+00:00" if timestamp.endswith("Z") else timestamp
        created = datetime.fromisoformat(value).astimezone(timezone.utc)
        elapsed = max(0, (_utc_now() - created).total_seconds())
    except (AttributeError, TypeError, ValueError):
        return str(timestamp)
    if elapsed < 60:
        return "Just now"
    if elapsed < 3600:
        return "{0}m ago".format(int(elapsed // 60))
    if elapsed < 86400:
        return "{0}h ago".format(int(elapsed // 3600))
    return created.strftime("%b %d, %H:%M UTC")


def _sidebar_navigation():
    pages = {
        "Overview": "Overview",
        "Incoming Tickets": "Overview",
        "Bulk Processing": "Overview",
        "Ticket History": "Operations",
        "Analytics": "Operations",
        "AI Models": "System",
    }
    if "active_page" not in st.session_state:
        st.session_state["active_page"] = "Overview"
    current = st.session_state["active_page"]

    groups = {
        "nav_overview": ["Overview", "Incoming Tickets", "Bulk Processing"],
        "nav_operations": ["Ticket History", "Analytics"],
        "nav_system": ["AI Models"],
    }
    for key, options in groups.items():
        st.session_state.setdefault(key, current if current in options else None)

    def update_page(source_key):
        selected = st.session_state.get(source_key)
        if selected:
            st.session_state["active_page"] = selected
            for other_key in groups:
                if other_key != source_key:
                    st.session_state[other_key] = None

    st.sidebar.markdown(
        '<div class="brand"><div class="brand-mark">✳</div>'
        '<div class="brand-name">IT Support AI</div>'
        '<div class="brand-subtitle">Intelligent ticket operations</div></div>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown('<div class="nav-section">OVERVIEW</div>', unsafe_allow_html=True)
    st.sidebar.radio(
        "Overview navigation", ["Overview", "Incoming Tickets", "Bulk Processing"],
        index=["Overview", "Incoming Tickets", "Bulk Processing"].index(current)
        if pages[current] == "Overview" else None,
        key="nav_overview", on_change=update_page, args=("nav_overview",),
        label_visibility="collapsed",
    )
    st.sidebar.markdown('<div class="nav-section">OPERATIONS</div>', unsafe_allow_html=True)
    st.sidebar.radio(
        "Operations navigation", ["Ticket History", "Analytics"],
        index=["Ticket History", "Analytics"].index(current)
        if pages[current] == "Operations" else None,
        key="nav_operations", on_change=update_page, args=("nav_operations",),
        label_visibility="collapsed",
    )
    st.sidebar.markdown('<div class="nav-section">SYSTEM</div>', unsafe_allow_html=True)
    st.sidebar.radio(
        "System navigation", ["AI Models"], index=0 if current == "AI Models" else None,
        key="nav_system", on_change=update_page, args=("nav_system",), label_visibility="collapsed",
    )
    st.sidebar.markdown(
        '<div class="sidebar-footer"><span class="status-dot">●</span>System operational</div>',
        unsafe_allow_html=True,
    )
    return current


def _go_to_intake():
    st.session_state["active_page"] = "Incoming Tickets"
    st.session_state["nav_overview"] = "Incoming Tickets"
    st.session_state["nav_operations"] = None
    st.session_state["nav_system"] = None


def _empty_state(title, description, action_label=None):
    st.markdown(
        '<div class="empty-state"><div class="empty-title">{0}</div>'
        '<div class="muted">{1}</div></div>'.format(_safe_html(title), _safe_html(description)),
        unsafe_allow_html=True,
    )
    if action_label:
        st.button(action_label, type="primary", on_click=_go_to_intake, key="empty_cta")


def _chart(counts, color, height=255):
    if not counts:
        st.caption("No records to chart yet.")
        return
    rows = [{"Category": str(category), "Tickets": int(count)}
            for category, count in counts.items()]
    st.bar_chart(rows, x="Category", y="Tickets", color=color, height=height)


def _recent_table(tickets, height=250):
    rows = [{
        "ID": "#{0}".format(ticket["id"]),
        "Ticket": ticket["body"].replace("\n", " ").strip()[:76]
                  + ("…" if len(ticket["body"].strip()) > 76 else ""),
        "Department": ticket["predicted_department"],
        "Priority": ticket["predicted_priority"].title(),
        "SLA": ticket["sla_status"],
        "Source": ticket.get("source", "incoming").title(),
        "Created": _relative_time(ticket["created_at"]),
    } for ticket in tickets]
    st.dataframe(rows, width="stretch", hide_index=True, height=height)


def _overview(service):
    _page_header(
        "IT SUPPORT AI", "IT Support AI",
        "Automated ticket classification and operational intelligence.",
    )
    _demo_data_panel(service)
    data = service.get_analytics(reference_time=_utc_now())
    priorities = data["priority_counts"]
    cols = st.columns(4)
    for col, label, value, color in zip(
        cols,
        ["Total Tickets", "High Priority", "Medium Priority", "Low Priority"],
        [data["total_tickets"], priorities.get("high", 0),
         priorities.get("medium", 0), priorities.get("low", 0)],
        ["#FFF0E8", "#FFE8DF", "#FFF4EA", "#FFF5DF"],
    ):
        with col:
            _kpi_card(label, value, color)

    st.markdown('<div class="section-heading">Ticket activity</div>', unsafe_allow_html=True)
    if not data["total_tickets"]:
        _empty_state(
            "No tickets yet", "Incoming tickets will appear here automatically.",
            "Simulate an incoming ticket →",
        )
        return

    chart_specs = [
        ("Department distribution", data["department_counts"], "#F4A29A"),
        ("Priority distribution", priorities, "#F6B68A"),
        ("SLA status distribution", data["sla_status_counts"], "#E9BFA4"),
    ]
    st.markdown("#### " + chart_specs[0][0])
    _chart(chart_specs[0][1], chart_specs[0][2], height=230)
    chart_cols = st.columns(2)
    for col, (title, counts, color) in zip(chart_cols, chart_specs[1:]):
        with col:
            st.markdown("#### " + title)
            _chart(counts, color, height=230)
    st.markdown('<div class="section-heading">Recent tickets</div>', unsafe_allow_html=True)
    _recent_table(service.get_ticket_history(limit=5, reference_time=_utc_now()))


def _demo_data_panel(service):
    """Explicitly load/clear clearly identified samples through app service."""
    database = service.database
    demo_count = database.count_tickets_by_source("demo")
    demo_set_complete = demo_count >= len(DEMO_TICKET_BODIES)
    with st.container(border=True):
        st.markdown("### Demo Data")
        notice = st.session_state.pop("demo_notice", None)
        if notice:
            st.success(notice)
        st.caption(
            "Loads {0} sample tickets for demonstration. These are not real customer complaints. "
            "The project has no live customer support integration."
            .format(len(DEMO_TICKET_BODIES))
        )
        if demo_count:
            if demo_set_complete:
                st.info("{0} demo tickets are currently loaded.".format(demo_count))
            else:
                st.info(
                    "{0} of {1} demo tickets are loaded. Load Demo Tickets again to fill any missing samples."
                    .format(demo_count, len(DEMO_TICKET_BODIES))
                )
        load_col, clear_col = st.columns(2)
        with load_col:
            if st.button(
                "Load Demo Tickets", type="primary", disabled=demo_set_complete,
                key="load_demo_tickets",
            ):
                try:
                    result = load_demo_tickets(service)
                    st.session_state["demo_notice"] = (
                        "Loaded {0} sample tickets through the normal prediction and SLA workflow."
                        .format(result["created"])
                    )
                    st.rerun()
                except Exception as error:
                    st.error("Demo tickets could not be loaded: {0}".format(error))
        with clear_col:
            if demo_count:
                if st.button("Clear Demo Data", key="clear_demo_data"):
                    st.session_state["confirm_clear_demo_open"] = True
                    st.rerun()
            else:
                st.button("Clear Demo Data", disabled=True, key="clear_demo_disabled")
        if demo_count and st.session_state.get("confirm_clear_demo_open"):
            st.warning("This removes only records marked as demo data. Other ticket sources are kept.")
            cancel_col, confirm_col = st.columns(2)
            with cancel_col:
                if st.button("Cancel", key="cancel_demo_clear"):
                    st.session_state["confirm_clear_demo_open"] = False
                    st.rerun()
            with confirm_col:
                if st.button("Confirm Clear Demo Data", type="primary", key="confirm_demo_clear"):
                    removed = clear_demo_tickets(database)
                    st.session_state["confirm_clear_demo_open"] = False
                    st.session_state["demo_notice"] = (
                        "Removed {0} demo tickets. Other ticket sources were kept.".format(removed)
                    )
                    st.rerun()
        st.caption(
            "Demo tickets pass through the saved classifiers and configured SLA rules. "
            "Their predicted labels are model outputs, not ground truth."
        )
        st.caption(
            "Training CSV data is used only for model development. Current activity comes from "
            "incoming, API, bulk, or explicitly loaded demo tickets; no external support platform is connected."
        )


def _workflow():
    stages = ["Received", "Classified", "Prioritized", "Stored"]
    cols = st.columns([1, .24, 1, .24, 1, .24, 1])
    for index, stage in enumerate(stages):
        with cols[index * 2]:
            st.markdown('<div class="flow-step flow-done">✓ &nbsp;{0}</div>'.format(
                _safe_html(stage)), unsafe_allow_html=True)
        if index < len(stages) - 1:
            with cols[index * 2 + 1]:
                st.markdown("<div style='text-align:center;color:#D98A7B;padding:.7rem 0'>→</div>",
                            unsafe_allow_html=True)


def _display_ticket_result(result):
    ticket, prediction, sla = result["ticket"], result["prediction"], result["sla"]
    st.markdown('<div class="section-heading">Automatically processed</div>', unsafe_allow_html=True)
    st.markdown('<div class="muted" style="margin-bottom:.75rem">Ticket #{0}</div>'.format(
        ticket["id"]), unsafe_allow_html=True)
    cards = st.columns(4)
    for col, label, value, background in (
        (cards[0], "Ticket ID", "#{0}".format(ticket["id"]), "#FFF0E8"),
        (cards[1], "Department", prediction["department"], "#FFF4EA"),
        (cards[2], "Priority", prediction["priority"].upper(), "#FFE8DF"),
        (cards[3], "SLA Status", sla["status"], "#FFF5DF"),
    ):
        with col:
            st.markdown(
                '<div class="result-card" style="background:{0}">'
                '<div class="result-label">{1}</div><div class="result-value">{2}</div></div>'.format(
                    background, _safe_html(label), _safe_html(value)
                ), unsafe_allow_html=True,
            )
    st.markdown('<div class="section-heading">Intake workflow</div>', unsafe_allow_html=True)
    _workflow()
    st.caption("SLA is a configurable demonstration rule evaluated from this application's ticket timestamp.")


def _incoming_tickets(service):
    _page_header(
        "INCOMING TICKETS", "Ticket intake",
        "See how an incoming customer request is automatically classified and routed.",
    )
    with st.container(border=True):
        st.markdown("### Simulate customer ticket")
        st.caption("This simulator demonstrates the automated intake flow. The API is available for programmatic intake.")
        with st.form("ticket_intake_form", clear_on_submit=False):
            body = st.text_area(
                "Ticket description", height=190,
                placeholder="Describe the customer's issue...", max_chars=12000,
                label_visibility="collapsed",
            )
            submitted = st.form_submit_button("Process ticket", type="primary")

    if submitted:
        st.session_state.pop("latest_ticket_result", None)
        if not body or not body.strip():
            st.error("Add a ticket description to continue.")
            return
        try:
            with st.spinner("Automatically processing ticket…"):
                result = service.submit_ticket(body, source="incoming")
            st.session_state["latest_ticket_result"] = result
        except Exception as error:
            st.error("The ticket could not be processed: {0}".format(error))
            return
    result = st.session_state.get("latest_ticket_result")
    if result:
        st.markdown("<div style='height:.55rem'></div>", unsafe_allow_html=True)
        _display_ticket_result(result)


def _bulk_processing(service):
    _page_header(
        "BULK PROCESSING", "Process tickets in bulk",
        "Upload a CSV containing support tickets and automatically classify them.",
    )
    with st.container(border=True):
        st.markdown("### Upload ticket CSV")
        st.caption("A Body column is required. Other columns are preserved as metadata and are not model features.")
        uploaded = st.file_uploader("Choose a CSV file", type=["csv"], key="bulk_csv_upload")
    if uploaded is not None:
        payload = uploaded.getvalue()
        digest = hashlib.sha256(payload).hexdigest()
        records = service.database.get_bulk_batch(digest)
        if records:
            processed = bulk_frame_from_records(records)
            for index, record in enumerate(records):
                processed.loc[index, "sla_status"] = service.sla_engine.evaluate(
                    record["created_at"], record["predicted_priority"], _utc_now()
                )["status"]
        else:
            try:
                source = pd.read_csv(io.BytesIO(payload), dtype=str, keep_default_na=False)
                progress = st.progress(0, text="Validating uploaded tickets…")
                processed = process_bulk_frame(
                    source, service, batch_id=digest,
                    progress_callback=lambda value: progress.progress(
                        min(1.0, value), text="Processing tickets…"
                    ),
                )
                progress.empty()
            except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as error:
                st.error("The CSV could not be read: {0}".format(error))
                return
            except BulkInputError as error:
                st.error(str(error))
                return
            except Exception as error:
                st.error("Bulk processing stopped: {0}".format(error))
                return
    else:
        digest = service.database.get_latest_bulk_batch()
        if not digest:
            return
        records = service.database.get_bulk_batch(digest)
        processed = bulk_frame_from_records(records)
        for index, record in enumerate(records):
            processed.loc[index, "sla_status"] = service.sla_engine.evaluate(
                record["created_at"], record["predicted_priority"], _utc_now()
            )["status"]

    if not processed.empty:
        st.caption("Showing the latest processed batch saved in SQLite.")
    valid = processed["Body"].astype(str).str.strip().ne("")
    st.markdown('<div class="section-heading">Processing summary</div>', unsafe_allow_html=True)
    metrics = st.columns(3)
    metrics[0].metric("Processed tickets", int(valid.sum()))
    metrics[1].metric("Departments", processed.loc[valid, "predicted_department"].nunique())
    metrics[2].metric("Priorities", processed.loc[valid, "predicted_priority"].nunique())

    if valid.any():
        distributions = st.columns(2)
        with distributions[0]:
            st.markdown("#### Department distribution")
            _chart(processed.loc[valid, "predicted_department"].value_counts().to_dict(), "#F4A29A")
        with distributions[1]:
            st.markdown("#### Priority distribution")
            _chart(processed.loc[valid, "predicted_priority"].value_counts().to_dict(), "#F6B68A")
        display = processed.loc[valid, ["Body", "predicted_department", "predicted_priority", "sla_status"]]
        display = display.rename(columns={"Body": "Ticket Body", "predicted_department": "Department",
                                          "predicted_priority": "Priority", "sla_status": "SLA"})
        st.dataframe(display, width="stretch", hide_index=True, height=330)
    else:
        st.info("The CSV contains no nonblank Body rows, so no tickets were stored.")

    st.download_button(
        "Download processed CSV", processed.to_csv(index=False).encode("utf-8-sig"),
        file_name="processed_tickets.csv", mime="text/csv", type="primary",
    )
    st.caption("Bulk data is used for inference only. The saved classifiers are not changed.")


def _ticket_history(service):
    _page_header("OPERATIONS", "Ticket history", "Application tickets with current demonstration SLA status.")
    tickets = service.get_ticket_history(limit=200, reference_time=_utc_now())
    if not tickets:
        _empty_state("No tickets yet", "Submitted tickets will appear here.", "Simulate an incoming ticket →")
        return

    st.markdown('<div class="section-heading">Recent records</div>', unsafe_allow_html=True)
    for ticket in tickets:
        cols = st.columns([.75, 3.9, 1.9, 1.0, 1.0, 1.2, 1.15])
        with cols[0]:
            st.markdown("**#{0}**".format(ticket["id"]))
        with cols[1]:
            body = ticket["body"].replace("\n", " ").strip()
            st.markdown("**{0}**".format(_safe_html(body[:92] + ("…" if len(body) > 92 else ""))))
        with cols[2]:
            st.caption(ticket["predicted_department"])
        with cols[3]:
            st.markdown(_badge(ticket["predicted_priority"].title(),
                                _priority_kind(ticket["predicted_priority"])), unsafe_allow_html=True)
        with cols[4]:
            st.markdown(_badge(ticket["sla_status"], _sla_kind(ticket["sla_status"])), unsafe_allow_html=True)
        with cols[5]:
            st.caption(ticket.get("source", "incoming").title())
        with cols[6]:
            st.caption(_relative_time(ticket["created_at"]))

    ids = [ticket["id"] for ticket in tickets]
    selected = st.selectbox("View ticket details", ids, format_func=lambda ticket_id: "Ticket #{0}".format(ticket_id))
    detail = service.get_ticket(selected, reference_time=_utc_now())
    if detail:
        ticket, sla = detail["ticket"], detail["sla"]
        st.markdown('<div class="section-heading">Ticket #{0}</div>'.format(ticket["id"]), unsafe_allow_html=True)
        st.markdown('<div class="card">{0}</div>'.format(
            _safe_html(ticket["body"]).replace("\n", "<br>")), unsafe_allow_html=True)
        info = st.columns(4)
        info[0].caption("Department")
        info[0].write(ticket["predicted_department"])
        info[1].caption("Priority")
        info[1].markdown(_badge(ticket["predicted_priority"].title(),
                                _priority_kind(ticket["predicted_priority"])), unsafe_allow_html=True)
        info[2].caption("Current SLA")
        info[2].markdown(_badge(sla["status"], _sla_kind(sla["status"])), unsafe_allow_html=True)
        info[3].caption("Created")
        info[3].write(ticket["created_at"])
        st.caption("Data source: {0}".format(ticket.get("source", "incoming").title()))


def _analytics(service):
    _page_header("OPERATIONS", "Ticket analytics", "Operational activity generated by this application.")
    data = service.get_analytics(reference_time=_utc_now())
    if not data["total_tickets"]:
        _empty_state("No activity yet", "Analytics will appear after tickets are submitted.",
                     "Simulate an incoming ticket →")
        return
    cols = st.columns(4)
    for col, label, value, color in zip(
        cols, ["Tickets processed", "Departments", "Priority levels", "SLA statuses"],
        [data["total_tickets"], len(data["department_counts"]), len(data["priority_counts"]),
         len(data["sla_status_counts"])], ["#FFF0E8", "#FFE8DF", "#FFF4EA", "#FFF5DF"]
    ):
        with col:
            _kpi_card(label, value, color)
    st.markdown('<div class="section-heading">Activity breakdown</div>', unsafe_allow_html=True)
    chart_specs = [
        ("Department distribution", data["department_counts"], "#F4A29A"),
        ("Priority distribution", data["priority_counts"], "#F6B68A"),
        ("SLA status distribution", data["sla_status_counts"], "#E9BFA4"),
    ]
    st.markdown("#### " + chart_specs[0][0])
    _chart(chart_specs[0][1], chart_specs[0][2], height=230)
    chart_cols = st.columns(2)
    for col, (title, counts, color) in zip(chart_cols, chart_specs[1:]):
        with col:
            st.markdown("#### " + title)
            _chart(counts, color, height=230)
    st.markdown("#### Ticket origins")
    _chart(data.get("source_counts", {}), "#E9BFA4", height=190)
    st.markdown('<div class="section-heading">Recent ticket activity</div>', unsafe_allow_html=True)
    _recent_table(service.get_ticket_history(limit=8, reference_time=_utc_now()), height=300)
    st.caption("All counts use application-created SQLite records only; the training CSV is not shown as activity.")


def _models():
    _page_header("SYSTEM", "AI models", "Transparent, text-only classifiers for ticket routing.")
    cards = st.columns(2)
    specs = [
        ("Department classifier", "65.28%", "65.55%", "#F4A29A"),
        ("Priority classifier", "68.60%", "67.11%", "#F6B68A"),
    ]
    for col, (name, accuracy, macro_f1, accent) in zip(cards, specs):
        with col:
            st.markdown(
                '<div class="model-card"><div class="model-accent" style="background:{0}"></div>'
                '<div class="model-name">{1}</div><div class="model-type">TF-IDF + Linear SVM</div>'
                '<div class="model-scores"><div><div class="model-score">{2}</div>'
                '<div class="model-score-label">Accuracy</div></div>'
                '<div><div class="model-score">{3}</div>'
                '<div class="model-score-label">Macro F1</div></div></div></div>'.format(
                    accent, _safe_html(name), accuracy, macro_f1
                ), unsafe_allow_html=True,
            )
    st.caption("Scores are from the held-out test splits documented in the model reports; they do not establish production performance.")

    left, right = st.columns([1.05, 1])
    with left:
        st.markdown('<div class="section-heading">How it works</div>', unsafe_allow_html=True)
        st.markdown(
            "1. Ticket Body is transformed using the saved TF-IDF pipeline.\n"
            "2. Separate Linear SVM models predict Department and Priority.\n"
            "3. The application stores the labels in SQLite.\n"
            "4. Configurable demonstration rules evaluate ticket age."
        )
    with right:
        st.markdown('<div class="section-heading">Limitations</div>', unsafe_allow_html=True)
        st.markdown(
            "Predictions are labels, not confidence scores. Repeated generic bodies can have conflicting labels. "
            "The model scores describe held-out samples from the supplied dataset only."
        )
        st.caption("The SLA component is a configurable rule-based demonstration. The project dataset does not contain historical SLA outcomes, so the system does not claim to predict real-world SLA breaches.")


def main():
    page = _sidebar_navigation()
    try:
        service = get_application_service(str(resolve_database_path() or ""))
    except Exception as error:
        st.error("System services could not be initialized: {0}".format(error))
        return

    page_slot = st.empty()
    with page_slot.container():
        try:
            if page == "Overview":
                _overview(service)
            elif page == "Incoming Tickets":
                _incoming_tickets(service)
            elif page == "Bulk Processing":
                _bulk_processing(service)
            elif page == "Ticket History":
                _ticket_history(service)
            elif page == "Analytics":
                _analytics(service)
            else:
                _models()
        except Exception as error:
            st.error("This page could not be loaded: {0}".format(error))


if __name__ == "__main__":
    main()
