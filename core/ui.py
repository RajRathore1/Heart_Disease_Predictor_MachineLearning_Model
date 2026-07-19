"""Shared chrome used on every page - the medical disclaimer can't drift or
get forgotten on one page if every page calls the same functions. Also holds
the global CSS that gives the app its clinical product look."""

import streamlit as st

_RISK_STYLES = {
    "Low": {"color": "#15803d", "bg": "#f0fdf4", "border": "#bbf7d0", "icon": "✓"},
    "Moderate": {"color": "#b45309", "bg": "#fffbeb", "border": "#fde68a", "icon": "!"},
    "High": {"color": "#b91c1c", "bg": "#fef2f2", "border": "#fecaca", "icon": "▲"},
}

_GLOBAL_CSS = """
<style>
/* Typography and page rhythm */
.block-container { padding-top: 2.2rem; max-width: 1150px; }
h1 { font-weight: 700; letter-spacing: -0.02em; }
h2, h3 { font-weight: 650; letter-spacing: -0.01em; }

/* Card-like containers */
div[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
}
div[data-testid="stMetric"] label { color: #64748b; }

/* Forms and expanders read as panels */
div[data-testid="stForm"], details[data-testid="stExpander"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
}

/* Primary button: full-width, confident */
div[data-testid="stForm"] button[kind="primaryFormSubmit"],
button[kind="primary"] {
    width: 100%;
    border-radius: 8px;
    font-weight: 600;
    padding: 0.6rem 1rem;
}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background: #0f172a;
    border-right: 1px solid #1e293b;
}
section[data-testid="stSidebar"] * { color: #e2e8f0; }

/* Nav links (st.page_link) as pill buttons */
section[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"] {
    border-radius: 8px;
    padding: 0.55rem 0.75rem;
    margin: 0.1rem 0;
    transition: background 0.15s ease;
}
section[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"]:hover {
    background: #1e293b;
}
section[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"][aria-current="page"] {
    background: #0e7490;
}
section[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"] p {
    font-weight: 600;
    font-size: 0.95rem;
}

/* Sidebar inputs: dark surfaces with readable light text */
section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    background: #1e293b;
    border-color: #334155;
}
section[data-testid="stSidebar"] div[data-baseweb="select"] * {
    color: #e2e8f0 !important;
    fill: #94a3b8 !important;
}
section[data-testid="stSidebar"] input {
    background: #1e293b !important;
    color: #e2e8f0 !important;
    border-color: #334155 !important;
}
section[data-testid="stSidebar"] hr { border-color: #1e293b; }
</style>
"""


def apply_global_styles() -> None:
    st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)


def sidebar_nav(pages: list) -> None:
    """Branded sidebar: product identity on top, then custom nav links.
    Used with st.navigation(position='hidden') so the default nav (which
    labels the home page 'app' after its filename) never renders.
    """
    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 0.5rem 0 1.1rem 0;">
                <div style="display: flex; align-items: center; gap: 0.55rem;">
                    <div style="background: #0e7490; border-radius: 9px; width: 2.3rem; height: 2.3rem;
                                display: flex; align-items: center; justify-content: center;
                                font-size: 1.25rem; color: #ffffff; font-weight: 700;">♥</div>
                    <div>
                        <div style="font-size: 1.18rem; font-weight: 700; letter-spacing: -0.01em;">CardioCheck</div>
                        <div style="font-size: 0.74rem; color: #94a3b8; margin-top: -0.1rem;">
                            Heart Disease Risk Assessment
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        for page in pages:
            st.page_link(page)
        st.markdown("<hr style='margin: 1rem 0 0.8rem 0;'>", unsafe_allow_html=True)


def page_header(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div style="border-left: 4px solid #0e7490; padding: 0.2rem 0 0.2rem 1rem; margin-bottom: 1.2rem;">
            <div style="font-size: 1.9rem; font-weight: 700; letter-spacing: -0.02em;">{title}</div>
            <div style="color: #64748b; font-size: 1rem; margin-top: 0.15rem;">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_disclaimer() -> None:
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 0.75rem; background: #ffffff;
                    border: 1px solid #e2e8f0; border-left: 4px solid #f59e0b; border-radius: 10px;
                    padding: 0.65rem 1rem; margin-bottom: 1.3rem;
                    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);">
            <div style="font-size: 1.05rem;">⚕️</div>
            <div style="font-size: 0.86rem; color: #475569; line-height: 1.45;">
                <strong style="color: #0f172a;">For screening support only.</strong>
                This is a statistical risk estimate from a machine learning model — not a diagnosis.
                Always confirm with a qualified healthcare professional.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_risk_badge(risk_bucket: str, probability: float) -> None:
    style = _RISK_STYLES.get(risk_bucket, _RISK_STYLES["Moderate"])
    st.markdown(
        f"""
        <div style="background: {style['bg']}; border: 1px solid {style['border']};
                    border-radius: 12px; padding: 1.3rem 1.5rem; margin: 0.6rem 0 1rem 0;">
            <div style="display: flex; align-items: center; gap: 1rem;">
                <div style="background: {style['color']}; color: white; border-radius: 50%;
                            width: 2.6rem; height: 2.6rem; display: flex; align-items: center;
                            justify-content: center; font-size: 1.2rem; font-weight: 700;">
                    {style['icon']}
                </div>
                <div>
                    <div style="font-size: 1.25rem; font-weight: 700; color: {style['color']};">
                        {risk_bucket} risk of heart disease
                    </div>
                    <div style="color: #475569; font-size: 0.92rem;">
                        Predicted probability: <strong>{probability:.1%}</strong>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(min(max(probability, 0.0), 1.0))
