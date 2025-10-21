# pages/results.py
# -*- coding: utf-8 -*-
import io
import numpy as np
import pandas as pd
import streamlit as st
from state import ensure_defaults

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# --- Global tilgjengelighets-CSS (tydelig fokus) ---
st.markdown("""
<style>
:focus { outline: 3px solid #ffbf47 !important; outline-offset: 2px; }
.stButton>button:focus { box-shadow: 0 0 0 3px #ffbf47 !important; }
button[disabled], .stButton>button[disabled] { opacity: 0.6 !important; }
</style>
""", unsafe_allow_html=True)

ensure_defaults()
st.title("Resultater")

if "freqs" not in st.session_state:
    st.warning("Ingen simulering funnet. Kjør først en simulering.")
    st.stop()

freqs = st.session_state["freqs"]          # (T, P)
genotypes = st.session_state["genotypes"]  # (T eller T-1, P, 3)
num_pops = int(freqs.shape[1])

# ---------- DataFrame med sikker lengdeutjevning ----------
def make_dataframe() -> pd.DataFrame:
    freqs_arr = st.session_state["freqs"]
    genos_arr = st.session_state["genotypes"]

    T = freqs_arr.shape[0]
    P = freqs_arr.shape[1]
    G = genos_arr.shape[0]

    # Hvis genotypes mangler siste rad, pad med HW-frekvenser fra siste p
    if G == T - 1:
        last_rows = []
        for p_idx in range(P):
            p = float(freqs_arr[-1, p_idx])
            p2 = p * p
            pq = 2 * p * (1 - p)
            q2 = (1 - p) * (1 - p)
            last_rows.append([p2, pq, q2])
        genos_arr = np.vstack([genos_arr, np.array([last_rows], dtype=float)])
        G = genos_arr.shape[0]

    L = min(T, G)
    freqs_use = freqs_arr[:L]
    genos_use = genos_arr[:L]

    rows = {"generation": np.arange(L, dtype=int)}
    for p in range(P):
        rows[f"p_A1_pop{p+1}"] = freqs_use[:, p]
    for p in range(P):
        rows[f"AA_pop{p+1}"] = genos_use[:, p, 0]
        rows[f"Aa_pop{p+1}"] = genos_use[:, p, 1]
        rows[f"aa_pop{p+1}"] = genos_use[:, p, 2]
    return pd.DataFrame(rows)

df = make_dataframe()

# ---------- Plot oppsett ----------
choice = st.radio("Velg visning:", ["Allelfrekvenser", "Genotypefrekvenser"], horizontal=True)

# Fargeblind-vennlig palett (Okabe–Ito)
GENO_COLORS = {"AA": "#0072B2", "Aa": "#E69F00", "aa": "#009E73"}  # blå, oransje, grønn
ALLELE_COLOR = "#0072B2"
LINE_WIDTH = 2.0

def apply_common_layout(fig: go.Figure, title: str, two_panels: bool = False) -> go.Figure:
    YMIN, YMAX = -0.02, 1.02
    if two_panels:
        fig.update_yaxes(range=[YMIN, YMAX], fixedrange=True, title_text="Frekvens", row=1, col=1)
        fig.update_yaxes(range=[YMIN, YMAX], fixedrange=True, row=1, col=2)
        fig.update_xaxes(title_text="Generasjon", fixedrange=True, row=1, col=1)
        fig.update_xaxes(title_text="Generasjon", fixedrange=True, row=1, col=2)
    else:
        fig.update_yaxes(range=[YMIN, YMAX], fixedrange=True, title_text="Frekvens")
        fig.update_xaxes(title_text="Generasjon", fixedrange=True)
    fig.update_layout(title=title, margin=dict(l=40, r=20, t=60, b=40), legend_title_text=None)
    return fig

def fig_allele_1pop_plotly(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["generation"], y=df["p_A1_pop1"], mode="lines", name="A₁",
        hovertemplate="Generasjon %{x}<br>Frekvens A₁: %{y:.3f}<extra></extra>",
        line=dict(color=ALLELE_COLOR, width=LINE_WIDTH),
    ))
    return apply_common_layout(fig, "Allelfrekvens (A₁)")

def fig_genotypes_1pop_plotly(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["generation"], y=df["AA_pop1"], mode="lines", name="A₁A₁",
                             hovertemplate="Generasjon %{x}<br>Frekvens A₁A₁: %{y:.3f}<extra></extra>",
                             line=dict(color=GENO_COLORS["AA"], width=LINE_WIDTH)))
    fig.add_trace(go.Scatter(x=df["generation"], y=df["Aa_pop1"], mode="lines", name="A₁A₂",
                             hovertemplate="Generasjon %{x}<br>Frekvens A₁A₂: %{y:.3f}<extra></extra>",
                             line=dict(color=GENO_COLORS["Aa"], width=LINE_WIDTH)))
    fig.add_trace(go.Scatter(x=df["generation"], y=df["aa_pop1"], mode="lines", name="A₂A₂",
                             hovertemplate="Generasjon %{x}<br>Frekvens A₂A₂: %{y:.3f}<extra></extra>",
                             line=dict(color=GENO_COLORS["aa"], width=LINE_WIDTH)))
    return apply_common_layout(fig, "Genotypefrekvenser")

def fig_allele_2pops_plotly(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Populasjon 1", "Populasjon 2"), shared_yaxes=False)
    fig.add_trace(go.Scatter(x=df["generation"], y=df["p_A1_pop1"], mode="lines", name="A₁",
                             hovertemplate="Generasjon %{x}<br>Frekvens A₁: %{y:.3f}<extra></extra>",
                             line=dict(color=ALLELE_COLOR, width=LINE_WIDTH), showlegend=True), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["generation"], y=df["p_A1_pop2"], mode="lines", name="A₁",
                             hovertemplate="Generasjon %{x}<br>Frekvens A₁: %{y:.3f}<extra></extra>",
                             line=dict(color=ALLELE_COLOR, width=LINE_WIDTH), showlegend=False), row=1, col=2)
    return apply_common_layout(fig, "Allelfrekvens (A₁) per populasjon", two_panels=True)

def fig_genotypes_2pops_plotly(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Populasjon 1", "Populasjon 2"), shared_yaxes=False)
    # Pop 1
    fig.add_trace(go.Scatter(x=df["generation"], y=df["AA_pop1"], mode="lines", name="A₁A₁",
                             hovertemplate="Generasjon %{x}<br>Genotype A₁A₁: %{y:.3f}<extra></extra>",
                             line=dict(color=GENO_COLORS["AA"], width=LINE_WIDTH), showlegend=True), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["generation"], y=df["Aa_pop1"], mode="lines", name="A₁A₂",
                             hovertemplate="Generasjon %{x}<br>Genotype A₁A₂: %{y:.3f}<extra></extra>",
                             line=dict(color=GENO_COLORS["Aa"], width=LINE_WIDTH), showlegend=True), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["generation"], y=df["aa_pop1"], mode="lines", name="A₂A₂",
                             hovertemplate="Generasjon %{x}<br>Genotype A₂A₂: %{y:.3f}<extra></extra>",
                             line=dict(color=GENO_COLORS["aa"], width=LINE_WIDTH), showlegend=True), row=1, col=1)
    # Pop 2 – samme farger
    fig.add_trace(go.Scatter(x=df["generation"], y=df["AA_pop2"], mode="lines", name="A₁A₁",
                             hovertemplate="Generasjon %{x}<br>Genotype A₁A₁: %{y:.3f}<extra></extra>",
                             line=dict(color=GENO_COLORS["AA"], width=LINE_WIDTH), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=df["generation"], y=df["Aa_pop2"], mode="lines", name="A₁A₂",
                             hovertemplate="Generasjon %{x}<br>Genotype A₁A₂: %{y:.3f}<extra></extra>",
                             line=dict(color=GENO_COLORS["Aa"], width=LINE_WIDTH), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=df["generation"], y=df["aa_pop2"], mode="lines", name="A₂A₂",
                             hovertemplate="Generasjon %{x}<br>Genotype A₂A₂: %{y:.3f}<extra></extra>",
                             line=dict(color=GENO_COLORS["aa"], width=LINE_WIDTH), showlegend=False), row=1, col=2)
    return apply_common_layout(fig, "Genotypefrekvenser per populasjon", two_panels=True)

# Velg figur
fig = (fig_allele_1pop_plotly(df) if num_pops == 1 and choice == "Allelfrekvenser" else
       fig_genotypes_1pop_plotly(df) if num_pops == 1 else
       fig_allele_2pops_plotly(df) if choice == "Allelfrekvenser" else
       fig_genotypes_2pops_plotly(df))

plot_config = {"displayModeBar": False, "scrollZoom": False, "doubleClick": False, "editable": False}
st.plotly_chart(fig, use_container_width=True, config=plot_config)

# --- ARIA live-region for forlengelse ---
st.markdown('<div id="sr-extend" role="status" aria-live="polite"></div>', unsafe_allow_html=True)

# --- Forleng simuleringen (under grafen) ---
colA, colSpacer = st.columns([1, 4])
with colA:
    more = st.button("➕ Kjør 100 generasjoner til")

if more:
    params = st.session_state.get("last_run_params", {})
    if not params:
        st.warning("Fant ingen sist brukte parametre – kjør først en simulering.")
        st.stop()

    extra = 100
    last_p = st.session_state["freqs"][-1, :]
    P = st.session_state["freqs"].shape[1]

    if P == 1:
        from utils import simulate_one_pop, Fitness
        fit = Fitness(params["wAA_1"], params["wAa_1"], params["waa_1"])
        extra_freqs, extra_genos = simulate_one_pop(
            p0=float(last_p[0]),
            N=params["N"] if params.get("use_drift", False) else None,
            fitness=fit, mu=params["mu"], nu=params["nu"],
            generations=extra,
            bottleneck_start=None, bottleneck_duration=None, bottleneck_size=None,
        )
    else:
        from utils import simulate_two_pops, Fitness
        fit1 = Fitness(params["wAA_1"], params["wAa_1"], params["waa_1"])
        fit2 = Fitness(params["wAA_2"], params["wAa_2"], params["waa_2"])
        extra_freqs, extra_genos = simulate_two_pops(
            p0_1=float(last_p[0]), p0_2=float(last_p[1]),
            N=params["N"] if params.get("use_drift", False) else None,
            fitness1=fit1, fitness2=fit2,
            mu=params["mu"], nu=params["nu"],
            generations=extra,
            migrate=params.get("migrate", False),
            m12=params.get("m12", 0.0), m21=params.get("m21", 0.0),
        )

    # dropp første rad (identisk med eksisterende slutt)
    extra_freqs = extra_freqs[1:, :]
    extra_genos = extra_genos[1:, :, :]

    st.session_state["freqs"] = np.vstack([st.session_state["freqs"], extra_freqs])
    st.session_state["genotypes"] = np.vstack([st.session_state["genotypes"], extra_genos])

    # Oppdater lokalt og DF
    freqs = st.session_state["freqs"]
    genotypes = st.session_state["genotypes"]
    df = make_dataframe()
    st.success(f"La til {extra} generasjoner.")
    st.markdown('<div role="status" aria-live="polite">La til 100 generasjoner.</div>', unsafe_allow_html=True)

# ---------- Parametre brukt ----------
st.markdown("### Parametre brukt i simuleringen")
params = st.session_state.get("last_run_params", {})
if not params:
    st.info("Fant ikke parametere fra siste kjøring.")
else:
    if params["num_pops"] == 1:
        st.markdown(f"""
- **Startfrekvens A₁:** {params['p0']:.2f}
- **Seleksjon (wAA, wAa, waa):** ({params['wAA_1']:.2f}, {params['wAa_1']:.2f}, {params['waa_1']:.2f})
- **Mutasjon (μ, ν):** ({params['mu']:.4f}, {params['nu']:.4f})
- **Endelig populasjonsstørrelse:** {"På" if params['use_drift'] else "Av"}{f", N={params['N']}" if params['use_drift'] else ""}
- **Flaskehals:** {"På" if params['use_bottleneck'] else "Av"}{(f", start={params['bottleneck_start']}, varighet={params['bottleneck_duration']}, N_fl={params['bottleneck_size']}") if params['use_bottleneck'] else ""}
""")
    else:
        st.markdown(f"""
- **Startfrekvens A₁:** pop1={params['p0_1']:.2f}, pop2={params['p0_2']:.2f}
- **Seleksjon pop1 (wAA, wAa, waa):** ({params['wAA_1']:.2f}, {params['wAa_1']:.2f}, {params['waa_1']:.2f})
- **Seleksjon pop2 (wAA, wAa, waa):** ({params['wAA_2']:.2f}, {params['wAa_2']:.2f}, {params['waa_2']:.2f})
- **Mutasjon (μ, ν):** ({params['mu']:.4f}, {params['nu']:.4f})
- **Genetisk drift:** {"På" if params['use_drift'] else "Av"}{f", N={params['N']}" if params['use_drift'] else ""}
- **Migrasjon:** {"På" if params['migrate'] else "Av"}{(f", m12={params['m12']:.2f}, m21={params['m21']:.2f}") if params['migrate'] else ""}
""")

# ---------- Matplotlib: bygg PNG for nedlasting ----------
def build_png_matplotlib(df: pd.DataFrame, choice: str, num_pops: int) -> bytes:
    plt.close("all")
    if num_pops == 1:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.set_ylim(0, 1)
        ax.set_xlabel("Generasjon")
        ax.set_ylabel("Frekvens")
        ax.grid(True, alpha=0.3)
        if choice == "Allelfrekvenser":
            ax.plot(df["generation"], df["p_A1_pop1"], label="A₁",
                    color="#0072B2", linewidth=2.0)
            ax.set_title("Allelfrekvens (A₁)")
            ax.legend()
        else:
            ax.plot(df["generation"], df["AA_pop1"], label="A₁A₁", color="#0072B2", linewidth=2.0)
            ax.plot(df["generation"], df["Aa_pop1"], label="A₁A₂", color="#E69F00", linewidth=2.0)
            ax.plot(df["generation"], df["aa_pop1"], label="A₂A₂", color="#009E73", linewidth=2.0)
            ax.set_title("Genotypefrekvenser")
            ax.legend()
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
        for ax, title in ((ax1, "Populasjon 1"), (ax2, "Populasjon 2")):
            ax.set_ylim(0, 1)
            ax.set_xlabel("Generasjon")
            ax.set_ylabel("Frekvens")
            ax.grid(True, alpha=0.3)
            ax.set_title(title)
        if choice == "Allelfrekvenser":
            ax1.plot(df["generation"], df["p_A1_pop1"], label="A₁", color="#0072B2", linewidth=2.0)
            ax2.plot(df["generation"], df["p_A1_pop2"], label="A₁", color="#0072B2", linewidth=2.0)
            fig.suptitle("Allelfrekvens (A₁) per populasjon")
            ax1.legend(); ax2.legend()
        else:
            ax1.plot(df["generation"], df["AA_pop1"], label="A₁A₁", color="#0072B2", linewidth=2.0)
            ax1.plot(df["generation"], df["Aa_pop1"], label="A₁A₂", color="#E69F00", linewidth=2.0)
            ax1.plot(df["generation"], df["aa_pop1"], label="A₂A₂", color="#009E73", linewidth=2.0)

            ax2.plot(df["generation"], df["AA_pop2"], label="A₁A₁", color="#0072B2", linewidth=2.0)
            ax2.plot(df["generation"], df["Aa_pop2"], label="A₁A₂", color="#E69F00", linewidth=2.0)
            ax2.plot(df["generation"], df["aa_pop2"], label="A₂A₂", color="#009E73", linewidth=2.0)

            fig.suptitle("Genotypefrekvenser per populasjon")
            ax1.legend(); ax2.legend()
        fig.tight_layout(rect=[0, 0, 1, 0.95])

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()

# ---------- Nedlasting: CSV + PNG ----------
st.markdown("---")
csv = df.to_csv(index=False).encode("utf-8")
c1, c2 = st.columns(2)
with c1:
    st.download_button("💾 Last ned data (CSV)", data=csv, file_name="popgen_results.csv", mime="text/csv")
with c2:
    png_bytes = build_png_matplotlib(df, choice, num_pops)
    st.download_button("🖼️ Last ned plott (PNG)", data=png_bytes, file_name="plot.png", mime="image/png")

st.caption("Hold pekeren over grafen for å lese av verdier.")
