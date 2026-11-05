"""Dashboard Garmin Forerunner 965 + Coach de IA local (Ollama).

Ejecutar:
    source .venv/bin/activate
    shiny run --launch-browser app.py
"""

from __future__ import annotations

import asyncio
import threading
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from faicons import icon_svg  # noqa: E402
from shiny import App, reactive, render, ui  # noqa: E402

import coach as coach_mod  # noqa: E402
from garmin_data import (  # noqa: E402
    GarminServicio,
    NecesitaMFA,
    Snapshot,
    modo_demo,
    snapshot_demo,
)

# --- Estado compartido (app personal: una sola cuenta de Garmin) -------------

servicio = GarminServicio()
_cache: dict = {"snap": None, "t": 0.0}
CACHE_SEG = 600  # no volver a pedir datos a Garmin más de una vez cada 10 min


def descargar(forzar: bool = False) -> Snapshot:
    if modo_demo():
        return snapshot_demo()
    if not forzar and _cache["snap"] and time.time() - _cache["t"] < CACHE_SEG:
        return _cache["snap"]
    snap = servicio.obtener()
    _cache.update(snap=snap, t=time.time())
    return snap


# Carga el modelo de Ollama en memoria al arrancar (en segundo plano)
threading.Thread(target=lambda: asyncio.run(coach_mod.precargar()), daemon=True).start()

# --- UI ----------------------------------------------------------------------

SALUDO = f"""
### 👋 Hola, soy tu Coach
Tengo tus datos de hoy del Forerunner 965. Pregúntame sobre **ciclismo o carrera**:

- <span class="suggestion">¿Hoy me conviene entrenar fuerte o descansar?</span>
- <span class="suggestion">¿Qué intensidad y zonas me recomiendas para hoy?</span>
- <span class="suggestion">Analiza mis últimos entrenamientos y mi recuperación</span>
- <span class="suggestion">Hazme un plan de 12 semanas para un maratón</span>
- <span class="suggestion">Quiero preparar un gran fondo de 120 km, ¿cómo empiezo?</span>

<small class="text-muted">Modelo local: {coach_mod.MODELO} (Ollama)</small>
"""

CSS = """
.bslib-value-box .value-box-title { font-size: .85rem; opacity: .85; }
.bslib-value-box .value-box-value { font-size: 2rem; }
.small-kpi { display:flex; justify-content:space-between; padding:.35rem 0;
             border-bottom:1px solid var(--bs-border-color-translucent); }
.small-kpi:last-child { border-bottom:none; }
.small-kpi span:last-child { font-weight:600; }
.estado-pill { font-size:.8rem; }
"""

panel_dashboard = ui.nav_panel(
    "Dashboard",
    ui.output_ui("aviso"),
    ui.output_ui("kpis"),
    ui.layout_columns(
        ui.card(
            ui.card_header(
                ui.div(
                    "Tendencia 14 días",
                    ui.input_radio_buttons(
                        "metrica",
                        None,
                        {
                            "sueno": "Sueño",
                            "vfc": "VFC",
                            "bb": "Body Battery",
                            "estres": "Estrés",
                            "fc_reposo": "FC reposo",
                        },
                        inline=True,
                    ),
                    class_="d-flex justify-content-between align-items-center flex-wrap gap-2",
                )
            ),
            ui.output_plot("grafica", height="270px"),
            full_screen=True,
            min_height="340px",
        ),
        ui.card(
            ui.card_header("Recuperación y carga"),
            ui.output_ui("recuperacion"),
            min_height="340px",
        ),
        col_widths=(8, 4),
        fill=False,
    ),
    ui.card(
        ui.card_header("Últimos entrenamientos"),
        ui.output_data_frame("tabla_actividades"),
        fill=False,
    ),
)

panel_chat = ui.nav_panel(
    "Coach IA",
    ui.layout_columns(
        ui.card(
            ui.chat_ui(
                "chat",
                greeting=SALUDO,
                placeholder="Pregunta sobre tu entrenamiento de ciclismo o carrera...",
                width="100%",
            ),
            class_="h-100",
        ),
        ui.card(
            ui.card_header("Lo que ve el Coach"),
            ui.output_ui("contexto_chat"),
            ui.input_action_button(
                "nueva_conv", "Nueva conversación", icon=icon_svg("rotate-left"),
                class_="btn-outline-secondary btn-sm mt-2",
            ),
        ),
        col_widths=(8, 4),
        height="calc(100vh - 110px)",
    ),
)

app_ui = ui.page_navbar(
    panel_dashboard,
    panel_chat,
    ui.nav_spacer(),
    ui.nav_control(ui.output_ui("estado_conexion")),
    ui.nav_control(
        ui.input_action_button(
            "actualizar", "Actualizar", icon=icon_svg("arrows-rotate"),
            class_="btn-sm btn-primary",
        )
    ),
    title=ui.span(icon_svg("person-running"), " Mi Garmin · FR965"),
    header=ui.tags.style(CSS),
    fillable="Coach IA",
    window_title="Mi Garmin · Coach",
)


# --- Utilidades de presentación ---------------------------------------------


def _fmt(v, sufijo="", nd="—"):
    return nd if v is None else f"{v}{sufijo}"


def _tema_score(v, alto=70, medio=40):
    if v is None:
        return "light"
    return "success" if v >= alto else ("warning" if v >= medio else "danger")


def _fila(etiqueta, valor):
    return ui.div(ui.span(etiqueta), ui.span(valor), class_="small-kpi")


# --- Servidor ----------------------------------------------------------------


def server(input, output, session):
    snap = reactive.value(None)
    error = reactive.value(None)
    mfa_estado = reactive.value(None)
    coach = coach_mod.Coach()
    chat = ui.Chat(id="chat", history=False)

    @reactive.extended_task
    async def tarea_descarga(forzar: bool):
        return await asyncio.to_thread(descargar, forzar)

    @reactive.extended_task
    async def tarea_mfa(estado, codigo: str):
        await asyncio.to_thread(servicio.completar_mfa, estado, codigo)
        return await asyncio.to_thread(descargar, True)

    @reactive.effect
    def _inicio():
        with reactive.isolate():
            tarea_descarga.invoke(False)

    @reactive.effect
    @reactive.event(input.actualizar)
    def _refrescar():
        error.set(None)
        tarea_descarga.invoke(True)

    def _procesar(tarea):
        st = tarea.status()
        if st == "success":
            s = tarea.result()
            snap.set(s)
            error.set(None)
            coach.actualizar_contexto(s.resumen_texto())
        elif st == "error":
            try:
                tarea.result()
            except NecesitaMFA as e:
                mfa_estado.set(e.estado)
                ui.modal_show(
                    ui.modal(
                        ui.p("Garmin envió un código de verificación a tu correo/teléfono."),
                        ui.input_text("mfa_codigo", "Código MFA", placeholder="123456"),
                        title="Verificación de Garmin",
                        footer=ui.input_action_button("mfa_enviar", "Verificar", class_="btn-primary"),
                        easy_close=False,
                    )
                )
            except Exception as e:  # noqa: BLE001
                error.set(f"{type(e).__name__}: {e}")

    @reactive.effect
    def _resultado_descarga():
        _procesar(tarea_descarga)

    @reactive.effect
    def _resultado_mfa():
        _procesar(tarea_mfa)

    @reactive.effect
    @reactive.event(input.mfa_enviar)
    def _enviar_mfa():
        codigo = (input.mfa_codigo() or "").strip()
        if not codigo:
            return
        ui.modal_remove()
        tarea_mfa.invoke(mfa_estado(), codigo)

    def cargando() -> bool:
        return tarea_descarga.status() == "running" or tarea_mfa.status() == "running"

    # ---- Navbar ----
    @render.ui
    def estado_conexion():
        if cargando():
            return ui.span(icon_svg("spinner"), " Sincronizando…", class_="badge text-bg-secondary estado-pill me-2")
        s = snap()
        if error():
            return ui.span("Error de conexión", class_="badge text-bg-danger estado-pill me-2")
        if s is None:
            return ui.span()
        if s.demo:
            return ui.span("Modo demo", class_="badge text-bg-warning estado-pill me-2")
        return ui.span(f"Conectado · {s.nombre or 'Garmin'}", class_="badge text-bg-success estado-pill me-2")

    # ---- Avisos ----
    @render.ui
    def aviso():
        if error():
            return ui.div(
                ui.strong("No se pudieron descargar los datos de Garmin. "),
                error(),
                ui.br(),
                ui.small("Revisa GARMIN_EMAIL / GARMIN_PASSWORD en .env y vuelve a pulsar Actualizar."),
                class_="alert alert-danger py-2",
            )
        s = snap()
        if s is None:
            return ui.div("Cargando datos del reloj…", class_="alert alert-secondary py-2")
        if s.demo:
            return ui.div(
                ui.strong("Datos de demostración. "),
                "Para ver tus datos reales agrega GARMIN_EMAIL y GARMIN_PASSWORD en el archivo ",
                ui.code(".env"), " y reinicia la app.",
                class_="alert alert-warning py-2",
            )
        if s.errores:
            return ui.div(
                "Algunos datos no están disponibles hoy: " + ", ".join(s.errores),
                class_="alert alert-light py-1 small",
            )
        return None

    # ---- KPIs ----
    @render.ui
    def kpis():
        s = snap()
        h = s.hoy if s else {}
        return ui.layout_column_wrap(
            ui.value_box(
                "Disposición para entrenar",
                _fmt(h.get("disposicion")),
                _fmt(h.get("disposicion_nivel"), nd=""),
                ui.span(h.get("disposicion_feedback") or "", class_="small"),
                showcase=icon_svg("gauge-high"),
                theme=_tema_score(h.get("disposicion"), 60, 30),
            ),
            ui.value_box(
                "Estado de VFC",
                _fmt(h.get("vfc_noche"), " ms"),
                _fmt(h.get("vfc_estado"), nd=""),
                ui.span(f"Promedio 7 d: {_fmt(h.get('vfc_semana'), ' ms')}", class_="small"),
                showcase=icon_svg("heart-pulse"),
                theme="success" if h.get("vfc_estado") == "Equilibrado" else _tema_score(None),
            ),
            ui.value_box(
                "Sueño",
                _fmt(h.get("sueno_score")),
                f"{_fmt(h.get('sueno_horas'), ' h')} · {_fmt(h.get('sueno_calidad'), nd='')}",
                showcase=icon_svg("moon"),
                theme=_tema_score(h.get("sueno_score"), 80, 60),
            ),
            ui.value_box(
                "Body Battery",
                _fmt(h.get("bb_actual")),
                f"Máx {_fmt(h.get('bb_max'))} · Mín {_fmt(h.get('bb_min'))}",
                showcase=icon_svg("battery-three-quarters"),
                theme=_tema_score(h.get("bb_actual"), 60, 30),
            ),
            width=1 / 4,
            fill=False,
        )

    @render.ui
    def recuperacion():
        s = snap()
        h = s.hoy if s else {}
        return ui.div(
            _fila("Tiempo de recuperación", _fmt(h.get("recuperacion_h"), " h")),
            _fila("Estado de entrenamiento", _fmt(h.get("estado_entreno"))),
            _fila("Carga aguda / crónica", f"{_fmt(h.get('carga_aguda'))} / {_fmt(h.get('carga_cronica'))}"),
            _fila("Estrés promedio", _fmt(h.get("estres"))),
            _fila("FC en reposo", _fmt(h.get("fc_reposo"), " lpm")),
            _fila("VO2 máx carrera / bici", f"{_fmt(h.get('vo2_carrera'))} / {_fmt(h.get('vo2_ciclismo'))}"),
            _fila(
                "Sueño profundo / REM",
                f"{_fmt(h.get('sueno_profundo_h'), ' h')} / {_fmt(h.get('sueno_rem_h'), ' h')}",
            ),
        )

    # ---- Gráfica ----
    @render.plot(alt="Tendencia de la métrica seleccionada")
    def grafica():
        s = snap()
        fig, ax = plt.subplots(figsize=(8, 3))
        for lado in ("top", "right"):
            ax.spines[lado].set_visible(False)
        if s is None or s.tendencia.empty:
            ax.text(0.5, 0.5, "Sin datos", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            return fig

        t = s.tendencia
        m = input.metrica()
        color = "#2563eb"
        if m == "bb" and "bb_max" in t:
            ax.bar(t.index, t["bb_max"] - t.get("bb_min", 0), bottom=t.get("bb_min", 0),
                   color="#60a5fa", width=0.6, label="Rango del día")
            ax.set_ylim(0, 100)
            ax.set_ylabel("Body Battery")
        else:
            col, etiqueta = {
                "sueno": ("sueno", "Puntuación de sueño"),
                "vfc": ("vfc", "VFC (ms)"),
                "estres": ("estres", "Estrés medio"),
                "fc_reposo": ("fc_reposo", "FC reposo (lpm)"),
            }[m]
            if col not in t or t[col].isna().all():
                ax.text(0.5, 0.5, "Sin datos para esta métrica", ha="center", va="center",
                        transform=ax.transAxes)
                return fig
            if m == "vfc" and "vfc_base_min" in t and t["vfc_base_min"].notna().any():
                ax.fill_between(t.index, t["vfc_base_min"], t["vfc_base_max"],
                                color="#22c55e", alpha=0.15, label="Rango base")
                ax.legend(frameon=False, fontsize=8, loc="upper left")
            ax.plot(t.index, t[col], marker="o", color=color, linewidth=2)
            ax.set_ylabel(etiqueta)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        return fig

    # ---- Tabla ----
    @render.data_frame
    def tabla_actividades():
        s = snap()
        if s is None or s.actividades.empty:
            return None
        return render.DataGrid(s.actividades.fillna("—"), height="320px")

    # ---- Chat ----
    @render.ui
    def contexto_chat():
        s = snap()
        if s is None:
            return ui.p("Esperando datos…", class_="text-muted")
        h = s.hoy
        return ui.div(
            _fila("Disposición", _fmt(h.get("disposicion"))),
            _fila("VFC", f"{_fmt(h.get('vfc_noche'), ' ms')} ({_fmt(h.get('vfc_estado'), nd='')})"),
            _fila("Sueño", f"{_fmt(h.get('sueno_score'))} · {_fmt(h.get('sueno_horas'), ' h')}"),
            _fila("Body Battery", _fmt(h.get("bb_actual"))),
            _fila("Recuperación", _fmt(h.get("recuperacion_h"), " h")),
            _fila("Entrenamientos", f"{len(s.actividades)} recientes"),
            ui.p(
                "El Coach solo responde sobre ciclismo, carrera y tu entrenamiento.",
                class_="small text-muted mt-3 mb-0",
            ),
        )

    @chat.on_user_submit
    async def _responder(mensaje: str):
        await chat.append_message_stream(coach.stream(mensaje))

    @reactive.effect
    @reactive.event(input.nueva_conv)
    async def _nueva_conversacion():
        coach.reiniciar()
        await chat.clear_messages(greeting=True)
        await chat.set_greeting(SALUDO)


app = App(app_ui, server)
