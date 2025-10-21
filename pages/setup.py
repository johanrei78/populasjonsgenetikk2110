# pages/setup.py
# -*- coding: utf-8 -*-
import streamlit as st
from constants import UI
from state import ensure_defaults, reset_results, reset_to_defaults
from utils import simulate_one_pop, simulate_two_pops, Fitness

# --- Global tilgjengelighets-CSS (tydelig fokus) ---
st.markdown("""
<style>
:focus { outline: 3px solid #ffbf47 !important; outline-offset: 2px; }
.stButton>button:focus { box-shadow: 0 0 0 3px #ffbf47 !important; }
button[disabled], .stButton>button[disabled] { opacity: 0.6 !important; }
</style>
""", unsafe_allow_html=True)

# --- 1) Defaults før vi lager widgets ---
ensure_defaults()

def _clamp01(x):
    try:
        return min(1.0, max(0.0, float(x)))
    except Exception:
        return None

# Startfrekvenser (0.5)
for k in ("p0", "p0_1", "p0_2"):
    v = st.session_state.get(k)
    if v is None:
        st.session_state[k] = 0.5
    else:
        vv = _clamp01(v)
        st.session_state[k] = 0.5 if vv is None else vv

# Fitness (1.0)
for k in ("wAA_1","wAa_1","waa_1","wAA_2","wAa_2","waa_2"):
    v = st.session_state.get(k)
    if v is None:
        st.session_state[k] = 1.0
    else:
        vv = _clamp01(v)
        st.session_state[k] = 1.0 if vv is None else vv

# Mutasjoner (0.0)
for k in ("mu","nu"):
    v = st.session_state.get(k)
    if v is None:
        st.session_state[k] = 0.0
    else:
        vv = _clamp01(v)
        st.session_state[k] = 0.0 if vv is None else vv

# Bool-flagg
for k, default in (("use_drift", False), ("use_bottleneck", False), ("migrate", False)):
    if st.session_state.get(k) is None:
        st.session_state[k] = default

# N default
if st.session_state.get("N") is None:
    st.session_state["N"] = 100

# Stabil default for generations via state
if st.session_state.get("generations") is None:
    st.session_state["generations"] = 100

st.title("Oppsett & evolusjonsmekanismer")

# --- 2) Tid (hotfix mot “hopper til 10” + uten varsel) ---
with st.expander("Tid ⏳", expanded=True):
    g_val = st.session_state.get("generations", 100)
    try:
        g_val = int(g_val)
    except Exception:
        g_val = 100
    g_val = max(UI["GEN_MIN"], min(UI["GEN_MAX"], g_val))
    st.session_state.pop("generations", None)
    st.slider(
        "Antall generasjoner",
        UI["GEN_MIN"], UI["GEN_MAX"],
        value=g_val,
        key="generations",
        on_change=reset_results
    )

# --- 3) Antall populasjoner ---
with st.expander("Antall populasjoner 👥", expanded=True):
    st.radio(
        "Antall populasjoner:", [1, 2],
        horizontal=True,
        key="num_pops",
        on_change=reset_results
    )
try:
    npops = int(st.session_state.get("num_pops", 1))
except Exception:
    npops = 1

# Sikre at pop 2-fitness og frekvens finnes når npops==2
if npops == 2:
    for k in ("p0_1","p0_2"):
        if st.session_state.get(k) is None:
            st.session_state[k] = 0.5
    for k in ("wAA_2","wAa_2","waa_2"):
        if st.session_state.get(k) is None:
            st.session_state[k] = 1.0

# --- 4) Startfrekvenser ---
with st.expander("Startfrekvens A₁ 🌱", expanded=True):
    if npops == 1:
        st.slider("Allelfrekvens A₁", 0.0, 1.0, step=UI["P_STEP"],
                  key="p0", on_change=reset_results)
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.slider("Populasjon 1", 0.0, 1.0, step=UI["P_STEP"],
                      key="p0_1", on_change=reset_results)
        with c2:
            st.slider("Populasjon 2", 0.0, 1.0, step=UI["P_STEP"],
                      key="p0_2", on_change=reset_results)

# --- 5) Naturlig seleksjon ---
with st.expander("Naturlig seleksjon 🧬", expanded=True):
    helper = "Sannsynligheten for at et individ med genotypen får avkom, sammenliknet med individer med andre genotyper (relativ fitness)."
    if npops == 1:
        st.slider("Fitness A₁A₁", 0.0, 1.0, step=UI["FIT_STEP"], key="wAA_1", on_change=reset_results, help=helper)
        st.slider("Fitness A₁A₂", 0.0, 1.0, step=UI["FIT_STEP"], key="wAa_1", on_change=reset_results, help=helper)
        st.slider("Fitness A₂A₂", 0.0, 1.0, step=UI["FIT_STEP"], key="waa_1", on_change=reset_results, help=helper)
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Populasjon 1")
            st.slider("Fitness A₁A₁", 0.0, 1.0, step=UI["FIT_STEP"], key="wAA_1", on_change=reset_results, help=helper)
            st.slider("Fitness A₁A₂", 0.0, 1.0, step=UI["FIT_STEP"], key="wAa_1", on_change=reset_results, help=helper)
            st.slider("Fitness A₂A₂", 0.0, 1.0, step=UI["FIT_STEP"], key="waa_1", on_change=reset_results, help=helper)
        with c2:
            st.subheader("Populasjon 2")
            st.slider("Fitness A₁A₁", 0.0, 1.0, step=UI["FIT_STEP"], key="wAA_2", on_change=reset_results, help=helper)
            st.slider("Fitness A₁A₂", 0.0, 1.0, step=UI["FIT_STEP"], key="wAa_2", on_change=reset_results, help=helper)
            st.slider("Fitness A₂A₂", 0.0, 1.0, step=UI["FIT_STEP"], key="waa_2", on_change=reset_results, help=helper)

# --- 6) Mutasjoner ---
with st.expander("Mutasjoner 🧪", expanded=True):
    st.slider("Mutasjonsrate A₁ → A₂",
              0.0, UI["MUT_MAX"], step=UI["MUT_STEP"],
              format="%.6f",
              key="mu", on_change=reset_results,
              help=f"Sannsynlighet per generasjon for at A₁ muterer til A₂ (0–{UI['MUT_MAX']}).")

    st.slider("Mutasjonsrate A₂ → A₁",
              0.0, UI["MUT_MAX"], step=UI["MUT_STEP"],
              format="%.6f",
              key="nu", on_change=reset_results,
              help=f"Sannsynlighet per generasjon for at A₂ muterer til A₁ (0–{UI['MUT_MAX']}).")

# --- 7) Genetisk drift ---
with st.expander("Genetisk drift 🎲", expanded=True):
    st.checkbox("Inkluder genetisk drift (endelig populasjonsstørrelse)",
                key="use_drift", on_change=reset_results,
                help="Hvis på: populasjonen simuleres med endelig størrelse N.")
    if st.session_state.get("use_drift", False):
        N_val = st.session_state.get("N", 100)
        if isinstance(N_val, (tuple, list)):
            N_val = int(N_val[0])
        else:
            N_val = int(N_val)
        st.session_state["N"] = N_val

        st.slider("Populasjonsstørrelse (N)", UI["N_MIN"], UI["N_MAX"],
                  step=1, key="N", on_change=reset_results,
                  help="Antall individer i populasjonen.")

# --- 8) Flaskehals (kun én pop + drift aktiv) ---
if npops == 1 and st.session_state.get("use_drift", False):
    with st.expander("Flaskehals ⏬"):
        st.checkbox("Inkluder flaskehalshendelse", key="use_bottleneck",
                    on_change=reset_results,
                    help="Kortvarig reduksjon i populasjonsstørrelse som øker genetisk drift.")

        if st.session_state.get("use_bottleneck", False):
            gens = int(st.session_state["generations"])
            max_start = max(gens - 1, 0)
            N_now = int(st.session_state["N"])

            if st.session_state.get("bottleneck_start") is None:
                st.session_state["bottleneck_start"] = min(20, max_start)
            if st.session_state.get("bottleneck_duration") is None:
                st.session_state["bottleneck_duration"] = min(10, max(gens - st.session_state["bottleneck_start"], 1))
            if st.session_state.get("bottleneck_size") is None:
                st.session_state["bottleneck_size"] = min(10, N_now)

            st.slider("Startgenerasjon", 0, max_start, step=1,
                      key="bottleneck_start", on_change=reset_results)
            max_duration = max(gens - int(st.session_state["bottleneck_start"]), 1)
            st.slider("Varighet (generasjoner)", 1, max_duration, step=1,
                      key="bottleneck_duration", on_change=reset_results)
            st.slider("Populasjonsstørrelse under flaskehals", 2, N_now, step=1,
                      key="bottleneck_size", on_change=reset_results)

# --- 9) Genflyt (kun ved to pop) ---
if npops == 2:
    with st.expander("Genflyt (migrasjon) 🔁"):
        st.checkbox("Inkluder migrasjon", key="migrate", on_change=reset_results,
                    help="Utveksling av gener mellom populasjonene.")
        if st.session_state["migrate"]:
            st.slider("Migrasjonsrate fra pop 1 → 2", 0.0, 1.0, step=UI["MIG_STEP"],
                      key="m12", on_change=reset_results,
                      help="Andelen av genene i populasjon 2 som kommer fra populasjon 1 i hver generasjon (m ∈ [0,1]).")
            st.slider("Migrasjonsrate fra pop 2 → 1", 0.0, 1.0, step=UI["MIG_STEP"],
                      key="m21", on_change=reset_results,
                      help="Andelen av genene i populasjon 1 som kommer fra populasjon 2 i hver generasjon (m ∈ [0,1]).")

# --- 10) Status-region for skjermleser ---
st.markdown('<div id="sr-status" role="status" aria-live="polite"></div>', unsafe_allow_html=True)

# --- 11) Kjør simulering (med spinner + status) ---
st.markdown("---")
if st.button("🚀 Kjør simulering"):
    status = st.empty()
    status.markdown('<div role="status" aria-live="polite">Kjører simulering …</div>', unsafe_allow_html=True)

    with st.spinner("Kjører simulering …"):
        N_arg = st.session_state["N"] if st.session_state.get("use_drift", False) else None

        bn = bool(st.session_state.get("use_bottleneck", False))
        b_start = st.session_state.get("bottleneck_start") if bn else None
        b_dur   = st.session_state.get("bottleneck_duration") if bn else None
        b_size  = st.session_state.get("bottleneck_size") if bn else None

        migr = bool(st.session_state.get("migrate", False))
        m12  = st.session_state.get("m12", 0.0)
        m21  = st.session_state.get("m21", 0.0)

        if npops == 1:
            freqs, genotypes = simulate_one_pop(
                p0=st.session_state["p0"],
                N=N_arg,
                fitness=Fitness(st.session_state["wAA_1"], st.session_state["wAa_1"], st.session_state["waa_1"]),
                mu=st.session_state["mu"],
                nu=st.session_state["nu"],
                generations=st.session_state["generations"],
                bottleneck_start=b_start, bottleneck_duration=b_dur, bottleneck_size=b_size,
            )
        else:
            freqs, genotypes = simulate_two_pops(
                p0_1=st.session_state["p0_1"], p0_2=st.session_state["p0_2"],
                N=N_arg,
                fitness1=Fitness(st.session_state["wAA_1"], st.session_state["wAa_1"], st.session_state["waa_1"]),
                fitness2=Fitness(st.session_state["wAA_2"], st.session_state["wAa_2"], st.session_state["waa_2"]),
                mu=st.session_state["mu"], nu=st.session_state["nu"],
                generations=st.session_state["generations"],
                migrate=migr, m12=m12, m21=m21,
            )

        st.session_state["freqs"] = freqs
        st.session_state["genotypes"] = genotypes

        st.session_state["last_run_params"] = dict(
            num_pops=npops,
            generations=int(st.session_state["generations"]),
            p0=float(st.session_state.get("p0", 0.5)),
            p0_1=float(st.session_state.get("p0_1", 0.5)),
            p0_2=float(st.session_state.get("p0_2", 0.5)),
            wAA_1=float(st.session_state["wAA_1"]), wAa_1=float(st.session_state["wAa_1"]), waa_1=float(st.session_state["waa_1"]),
            wAA_2=float(st.session_state.get("wAA_2", 1.0)), wAa_2=float(st.session_state.get("wAa_2", 1.0)), waa_2=float(st.session_state.get("waa_2", 1.0)),
            mu=float(st.session_state["mu"]), nu=float(st.session_state["nu"]),
            use_drift=bool(st.session_state.get("use_drift", False)),
            N=None if not st.session_state.get("use_drift", False) else int(st.session_state["N"]),
            use_bottleneck=bool(st.session_state.get("use_bottleneck", False)),
            bottleneck_start=st.session_state.get("bottleneck_start"),
            bottleneck_duration=st.session_state.get("bottleneck_duration"),
            bottleneck_size=st.session_state.get("bottleneck_size"),
            migrate=bool(st.session_state.get("migrate", False)),
            m12=float(st.session_state.get("m12", 0.0)),
            m21=float(st.session_state.get("m21", 0.0)),
        )

    status.markdown('<div role="status" aria-live="polite">Simulering ferdig.</div>', unsafe_allow_html=True)
    st.success("Simulering ferdig!")
    st.switch_page("pages/results.py")

# --- 12) Tilbakestill-knapp ---
st.markdown("---")
if st.button("🔄 Tilbakestill alle parametere"):
    reset_to_defaults()
    st.rerun()
