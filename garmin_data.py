"""Conexión con Garmin Connect y normalización de los datos del reloj.

Usa la librería `garminconnect`. Las credenciales se leen del archivo .env:

    GARMIN_EMAIL=tu_correo
    GARMIN_PASSWORD=tu_contraseña

Después del primer inicio de sesión los tokens se guardan en GARMIN_TOKENS
(por defecto ./.garmin_tokens), así que no se vuelve a pedir usuario/contraseña
ni código MFA en cada arranque.

Si no hay credenciales (o GARMIN_DEMO=1) se usan datos de demostración para
poder probar el dashboard y el chat.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

CARPETA = Path(__file__).parent
TOKENS = os.getenv("GARMIN_TOKENS") or str(CARPETA / ".garmin_tokens")
DIAS_TENDENCIA = 14

# --- Traducciones de los códigos que regresa Garmin --------------------------

NIVELES_DISPOSICION = {
    "PRIME": "Óptima",
    "HIGH": "Alta",
    "MODERATE": "Moderada",
    "LOW": "Baja",
    "POOR": "Muy baja",
}
FEEDBACK_DISPOSICION = {
    "WELL_RECOVERED": "Bien recuperado",
    "RECOVERED_AND_READY": "Recuperado y listo",
    "READY_TO_GO": "Listo para entrenar",
    "ENERGIZED_BY_GOOD_SLEEP": "Con energía tras buen sueño",
    "GOOD_SLEEP": "Buen sueño",
    "POOR_SLEEP": "Sueño deficiente",
    "HIGH_TRAINING_LOAD": "Carga de entrenamiento alta",
    "HIGH_LOAD": "Carga alta",
    "RECOVERING": "En recuperación",
    "RECOVERY_IN_PROGRESS": "Recuperación en progreso",
    "HRV_LOW": "VFC baja",
    "LOW_HRV": "VFC baja",
    "TAKE_IT_EASY": "Tómalo con calma",
    "REST": "Descansa",
}
ESTADOS_VFC = {
    "BALANCED": "Equilibrado",
    "UNBALANCED": "Desequilibrado",
    "LOW": "Bajo",
    "POOR": "Bajo",
    "NONE": "Sin datos",
}
CALIDAD_SUENO = {
    "EXCELLENT": "Excelente",
    "GOOD": "Bueno",
    "FAIR": "Regular",
    "POOR": "Malo",
}
ESTADOS_ENTRENAMIENTO = {
    "PRODUCTIVE": "Productivo",
    "MAINTAINING": "Mantenimiento",
    "RECOVERY": "Recuperación",
    "PEAKING": "Pico de forma",
    "UNPRODUCTIVE": "Improductivo",
    "DETRAINING": "Pérdida de forma",
    "OVERREACHING": "Sobrecarga",
    "STRAINED": "Forzado",
    "NO_STATUS": "Sin estado",
}
TIPOS_ACTIVIDAD = {
    "running": "Carrera",
    "trail_running": "Trail",
    "treadmill_running": "Cinta",
    "track_running": "Pista",
    "cycling": "Ciclismo",
    "road_biking": "Ruta",
    "mountain_biking": "MTB",
    "gravel_cycling": "Gravel",
    "indoor_cycling": "Rodillo",
    "virtual_ride": "Rodillo virtual",
    "strength_training": "Fuerza",
    "walking": "Caminata",
    "hiking": "Senderismo",
    "lap_swimming": "Natación",
    "open_water_swimming": "Aguas abiertas",
    "yoga": "Yoga",
}


def _get(d: Any, *ruta, default=None):
    """Lee un valor anidado sin romperse si falta alguna llave."""
    actual = d
    for paso in ruta:
        if isinstance(actual, dict):
            actual = actual.get(paso)
        elif isinstance(actual, list) and isinstance(paso, int) and len(actual) > paso:
            actual = actual[paso]
        else:
            return default
        if actual is None:
            return default
    return actual


def _traducir(tabla: dict, valor: str | None) -> str | None:
    if not valor:
        return None
    clave = str(valor).upper()
    for k, v in tabla.items():
        if clave.startswith(k):
            return v
    return str(valor).replace("_", " ").capitalize()


# --- Estructura de datos que consume la app ----------------------------------


@dataclass
class Snapshot:
    """Todo lo que muestra el dashboard (y lo que se le pasa a la IA)."""

    fecha: date
    demo: bool = False
    nombre: str | None = None
    hoy: dict[str, Any] = field(default_factory=dict)
    tendencia: pd.DataFrame = field(default_factory=pd.DataFrame)
    actividades: pd.DataFrame = field(default_factory=pd.DataFrame)
    errores: list[str] = field(default_factory=list)

    def resumen_texto(self) -> str:
        """Resumen compacto en texto para darle contexto al modelo de IA."""
        h = self.hoy
        lineas = [f"Fecha de los datos: {self.fecha.isoformat()}"]

        def agrega(etiqueta, valor, sufijo=""):
            if valor is not None and valor != "":
                lineas.append(f"- {etiqueta}: {valor}{sufijo}")

        agrega("Disposición para entrenar", h.get("disposicion"), "/100")
        agrega("Nivel de disposición", h.get("disposicion_nivel"))
        agrega("Comentario de Garmin", h.get("disposicion_feedback"))
        agrega("Tiempo de recuperación restante", h.get("recuperacion_h"), " h")
        agrega("VFC última noche", h.get("vfc_noche"), " ms")
        agrega("VFC promedio 7 días", h.get("vfc_semana"), " ms")
        if h.get("vfc_base_min") and h.get("vfc_base_max"):
            agrega("Rango base de VFC", f"{h['vfc_base_min']}-{h['vfc_base_max']}", " ms")
        agrega("Estado de VFC", h.get("vfc_estado"))
        agrega("Puntuación de sueño", h.get("sueno_score"), "/100")
        agrega("Calidad de sueño", h.get("sueno_calidad"))
        agrega("Horas dormidas", h.get("sueno_horas"), " h")
        agrega("Sueño profundo", h.get("sueno_profundo_h"), " h")
        agrega("Sueño REM", h.get("sueno_rem_h"), " h")
        agrega("Body Battery actual", h.get("bb_actual"))
        agrega("Body Battery máximo hoy", h.get("bb_max"))
        agrega("Body Battery mínimo hoy", h.get("bb_min"))
        agrega("Estrés promedio hoy", h.get("estres"), " (0-100)")
        agrega("FC en reposo", h.get("fc_reposo"), " lpm")
        agrega("Estado de entrenamiento", h.get("estado_entreno"))
        agrega("Carga aguda (7 días)", h.get("carga_aguda"))
        agrega("Carga crónica (28 días)", h.get("carga_cronica"))
        agrega("VO2 máx carrera", h.get("vo2_carrera"))
        agrega("VO2 máx ciclismo", h.get("vo2_ciclismo"))

        t = self.tendencia
        if not t.empty:
            lineas.append(f"\nTendencia últimos {len(t)} días (más antiguo → más reciente):")
            cols = {
                "sueno": "Sueño",
                "vfc": "VFC",
                "bb_max": "BB máx",
                "estres": "Estrés",
                "fc_reposo": "FC reposo",
            }
            for c, nombre in cols.items():
                if c in t and t[c].notna().any():
                    vals = ", ".join("—" if pd.isna(v) else f"{v:.0f}" for v in t[c])
                    lineas.append(f"- {nombre}: {vals}")

        a = self.actividades
        if not a.empty:
            lineas.append("\nÚltimos entrenamientos:")
            for _, r in a.head(8).iterrows():
                partes = [f"{r['Fecha']} {r['Tipo']} \"{r['Nombre']}\""]
                if r.get("Distancia (km)"):
                    partes.append(f"{r['Distancia (km)']} km")
                partes.append(f"{r['Duración']}")
                if r.get("Ritmo / Vel."):
                    partes.append(r["Ritmo / Vel."])
                if r.get("FC media"):
                    partes.append(f"FC media {r['FC media']}")
                if r.get("TE aeróbico"):
                    partes.append(f"TE aeróbico {r['TE aeróbico']}")
                if r.get("Carga"):
                    partes.append(f"carga {r['Carga']}")
                lineas.append("- " + ", ".join(partes))

        return "\n".join(lineas)


# --- Cliente Garmin ----------------------------------------------------------


class NecesitaMFA(Exception):
    """Garmin pidió un código de verificación (MFA)."""

    def __init__(self, estado):
        super().__init__("Se requiere código MFA")
        self.estado = estado


def credenciales_configuradas() -> bool:
    return bool(os.getenv("GARMIN_EMAIL") and os.getenv("GARMIN_PASSWORD"))


def tokens_guardados() -> bool:
    p = Path(TOKENS).expanduser()
    return p.exists() and any(p.iterdir()) if p.is_dir() else p.exists()


def modo_demo() -> bool:
    if os.getenv("GARMIN_DEMO", "").strip() in {"1", "true", "True", "si", "sí"}:
        return True
    return not (credenciales_configuradas() or tokens_guardados())


class GarminServicio:
    """Envuelve garminconnect.Garmin con login persistente y descarga de datos."""

    def __init__(self):
        self.api = None
        self.listo = False  # True cuando el login terminó (incluido el MFA)

    # ---- Autenticación ----
    def conectar(self, email: str | None = None, password: str | None = None):
        """Inicia sesión. Lanza NecesitaMFA si Garmin pide el código."""
        from garminconnect import Garmin

        email = email or os.getenv("GARMIN_EMAIL")
        password = password or os.getenv("GARMIN_PASSWORD")
        api = Garmin(email=email, password=password, return_on_mfa=True)
        estado, extra = api.login(TOKENS)
        self.api = api
        if estado == "needs_mfa":
            self.listo = False
            raise NecesitaMFA(extra)
        self.listo = True
        self._guardar_tokens()

    def completar_mfa(self, estado, codigo: str):
        self.api.resume_login(estado, codigo.strip())
        self.listo = True
        self._guardar_tokens()

    def _guardar_tokens(self):
        try:
            self.api.client.dump(TOKENS)
        except Exception as e:  # no es grave: solo se volverá a pedir login
            print(f"[Garmin] No se pudieron guardar los tokens: {e}")

    # ---- Descarga ----
    def _llamar(self, errores: list[str], nombre: str, fn, *args):
        try:
            return fn(*args)
        except Exception as e:
            errores.append(f"{nombre}: {type(e).__name__}")
            print(f"[Garmin] Error en {nombre}: {e}")
            return None

    def obtener(self, dia: date | None = None) -> Snapshot:
        if not self.listo:
            self.conectar()
        api = self.api
        dia = dia or date.today()
        hoy_s = dia.isoformat()
        inicio_s = (dia - timedelta(days=DIAS_TENDENCIA - 1)).isoformat()
        err: list[str] = []

        resumen = self._llamar(err, "resumen diario", api.get_user_summary, hoy_s) or {}
        disp = self._llamar(err, "disposición", api.get_training_readiness, hoy_s) or []
        vfc = self._llamar(err, "VFC", api.get_hrv_data, hoy_s) or {}
        sueno = self._llamar(err, "sueño", api.get_sleep_data, hoy_s) or {}
        estado = self._llamar(err, "estado de entrenamiento", api.get_training_status, hoy_s) or {}
        acts = self._llamar(err, "actividades", api.get_activities, 0, 15) or []

        # Tendencias (una llamada por métrica para todo el rango)
        diarios = self._llamar(err, "resúmenes", api.get_daily_summaries, inicio_s, hoy_s) or {}
        sueno_rango = self._llamar(err, "sueño (rango)", api.get_sleep_daily, inicio_s, hoy_s) or []
        vfc_rango = self._llamar(err, "VFC (rango)", api.get_hrv_data_range, inicio_s, hoy_s) or {}

        snap = Snapshot(fecha=dia, errores=err)
        try:
            snap.nombre = api.get_full_name()
        except Exception:
            pass
        snap.hoy = _normalizar_hoy(resumen, disp, vfc, sueno, estado)
        snap.tendencia = _normalizar_tendencia(dia, diarios, sueno_rango, vfc_rango)
        snap.actividades = _normalizar_actividades(acts)
        return snap


# --- Normalización -----------------------------------------------------------


def _horas(segundos) -> float | None:
    return round(segundos / 3600, 1) if segundos else None


def _normalizar_hoy(resumen, disp, vfc, sueno, estado) -> dict[str, Any]:
    h: dict[str, Any] = {}

    # Disposición para entrenar (puede venir varias veces en el día; la 1a es la más reciente)
    tr = disp[0] if isinstance(disp, list) and disp else (disp if isinstance(disp, dict) else {})
    h["disposicion"] = _get(tr, "score")
    h["disposicion_nivel"] = _traducir(NIVELES_DISPOSICION, _get(tr, "level"))
    fb = _get(tr, "feedbackShort") or _get(tr, "feedbackLong")
    h["disposicion_feedback"] = _traducir(FEEDBACK_DISPOSICION, fb)
    rec_min = _get(tr, "recoveryTime")
    h["recuperacion_h"] = round(rec_min / 60) if rec_min is not None else None

    # VFC
    s = _get(vfc, "hrvSummary", default={})
    h["vfc_noche"] = _get(s, "lastNightAvg")
    h["vfc_semana"] = _get(s, "weeklyAvg")
    h["vfc_estado"] = _traducir(ESTADOS_VFC, _get(s, "status"))
    h["vfc_base_min"] = _get(s, "baseline", "balancedLow")
    h["vfc_base_max"] = _get(s, "baseline", "balancedUpper")

    # Sueño
    d = _get(sueno, "dailySleepDTO", default={})
    h["sueno_score"] = _get(d, "sleepScores", "overall", "value")
    h["sueno_calidad"] = _traducir(CALIDAD_SUENO, _get(d, "sleepScores", "overall", "qualifierKey"))
    h["sueno_horas"] = _horas(_get(d, "sleepTimeSeconds"))
    h["sueno_profundo_h"] = _horas(_get(d, "deepSleepSeconds"))
    h["sueno_ligero_h"] = _horas(_get(d, "lightSleepSeconds"))
    h["sueno_rem_h"] = _horas(_get(d, "remSleepSeconds"))
    h["sueno_despierto_h"] = _horas(_get(d, "awakeSleepSeconds"))

    # Resumen del día
    h["bb_actual"] = _get(resumen, "bodyBatteryMostRecentValue")
    h["bb_max"] = _get(resumen, "bodyBatteryHighestValue")
    h["bb_min"] = _get(resumen, "bodyBatteryLowestValue")
    estres = _get(resumen, "averageStressLevel")
    h["estres"] = estres if estres is not None and estres >= 0 else None
    h["fc_reposo"] = _get(resumen, "restingHeartRate")
    h["pasos"] = _get(resumen, "totalSteps")

    # Estado de entrenamiento (viene indexado por id de dispositivo)
    datos = _get(estado, "mostRecentTrainingStatus", "latestTrainingStatusData", default={})
    disp_estado = next(iter(datos.values()), {}) if isinstance(datos, dict) and datos else {}
    h["estado_entreno"] = _traducir(
        ESTADOS_ENTRENAMIENTO,
        _get(disp_estado, "trainingStatusFeedbackPhrase") or _get(disp_estado, "trainingStatus"),
    )
    h["carga_aguda"] = _get(disp_estado, "acuteTrainingLoadDTO", "dailyTrainingLoadAcute")
    h["carga_cronica"] = _get(disp_estado, "acuteTrainingLoadDTO", "dailyTrainingLoadChronic")
    v_run = _get(estado, "mostRecentVO2Max", "generic", "vo2MaxPreciseValue")
    v_bike = _get(estado, "mostRecentVO2Max", "cycling", "vo2MaxPreciseValue")
    h["vo2_carrera"] = round(v_run, 1) if v_run else None
    h["vo2_ciclismo"] = round(v_bike, 1) if v_bike else None
    return h


def _normalizar_tendencia(dia, diarios, sueno_rango, vfc_rango) -> pd.DataFrame:
    fechas = [(dia - timedelta(days=i)).isoformat() for i in range(DIAS_TENDENCIA - 1, -1, -1)]
    df = pd.DataFrame(index=fechas)
    df.index.name = "fecha"

    for r in (_get(diarios, "data", default=[]) if isinstance(diarios, dict) else diarios or []):
        f = r.get("calendarDate")
        if f in df.index:
            df.loc[f, "bb_max"] = r.get("bodyBatteryHighestValue")
            df.loc[f, "bb_min"] = r.get("bodyBatteryLowestValue")
            est = r.get("averageStressLevel")
            df.loc[f, "estres"] = est if est is not None and est >= 0 else None
            df.loc[f, "fc_reposo"] = r.get("restingHeartRate")

    for r in sueno_rango or []:
        f = r.get("calendarDate")
        if f in df.index:
            v = r.get("values") or {}
            df.loc[f, "sueno"] = v.get("sleepScore") or v.get("overallScore")
            secs = v.get("totalSleepTimeInSeconds")
            df.loc[f, "sueno_horas"] = secs / 3600 if secs else None

    for r in _get(vfc_rango, "hrvSummaries", default=[]) or []:
        f = r.get("calendarDate")
        if f in df.index:
            df.loc[f, "vfc"] = r.get("lastNightAvg")
            df.loc[f, "vfc_base_min"] = _get(r, "baseline", "balancedLow")
            df.loc[f, "vfc_base_max"] = _get(r, "baseline", "balancedUpper")

    df.index = pd.to_datetime(df.index)
    return df.apply(pd.to_numeric, errors="coerce")


def _formatea_duracion(segundos) -> str:
    if not segundos:
        return "—"
    segundos = int(segundos)
    h, rem = divmod(segundos, 3600)
    m, _ = divmod(rem, 60)
    return f"{h}h {m:02d}m" if h else f"{m} min"


def _ritmo_o_velocidad(tipo: str, distancia_m, duracion_s, vel_ms) -> str | None:
    if not distancia_m or not duracion_s:
        return None
    if "run" in tipo or tipo in {"walking", "hiking"}:
        seg_km = duracion_s / (distancia_m / 1000)
        return f"{int(seg_km // 60)}:{int(seg_km % 60):02d} /km"
    vel = (vel_ms * 3.6) if vel_ms else (distancia_m / 1000) / (duracion_s / 3600)
    return f"{vel:.1f} km/h"


def _normalizar_actividades(acts) -> pd.DataFrame:
    filas = []
    for a in acts or []:
        tipo = _get(a, "activityType", "typeKey", default="")
        dist = a.get("distance")
        dur = a.get("duration") or a.get("movingDuration")
        inicio = a.get("startTimeLocal") or ""
        try:
            fecha = datetime.fromisoformat(inicio).strftime("%d/%m %H:%M")
        except ValueError:
            fecha = inicio[:16]
        te = a.get("aerobicTrainingEffect")
        carga = a.get("activityTrainingLoad")
        filas.append(
            {
                "Fecha": fecha,
                "Tipo": TIPOS_ACTIVIDAD.get(tipo, tipo.replace("_", " ").capitalize()),
                "Nombre": a.get("activityName") or "",
                "Distancia (km)": round(dist / 1000, 2) if dist else None,
                "Duración": _formatea_duracion(dur),
                "Ritmo / Vel.": _ritmo_o_velocidad(tipo, dist, dur, a.get("averageSpeed")),
                "FC media": int(a["averageHR"]) if a.get("averageHR") else None,
                "TE aeróbico": round(te, 1) if te else None,
                "Carga": int(carga) if carga else None,
            }
        )
    return pd.DataFrame(filas)


# --- Datos de demostración ---------------------------------------------------


def snapshot_demo(dia: date | None = None) -> Snapshot:
    """Datos inventados pero realistas para probar la app sin cuenta de Garmin."""
    dia = dia or date.today()
    rnd = random.Random(dia.toordinal())
    fechas = pd.date_range(end=pd.Timestamp(dia), periods=DIAS_TENDENCIA)
    n = len(fechas)
    tendencia = pd.DataFrame(
        {
            "sueno": [rnd.randint(62, 90) for _ in range(n)],
            "sueno_horas": [round(rnd.uniform(6.0, 8.3), 1) for _ in range(n)],
            "vfc": [rnd.randint(52, 70) for _ in range(n)],
            "vfc_base_min": [55] * n,
            "vfc_base_max": [68] * n,
            "bb_max": [rnd.randint(65, 98) for _ in range(n)],
            "bb_min": [rnd.randint(8, 30) for _ in range(n)],
            "estres": [rnd.randint(22, 45) for _ in range(n)],
            "fc_reposo": [rnd.randint(46, 52) for _ in range(n)],
        },
        index=fechas,
    )
    tendencia.index.name = "fecha"
    u = tendencia.iloc[-1]
    hoy = {
        "disposicion": 68,
        "disposicion_nivel": "Moderada",
        "disposicion_feedback": "Recuperación en progreso, carga alta reciente",
        "recuperacion_h": 14,
        "vfc_noche": int(u["vfc"]),
        "vfc_semana": int(tendencia["vfc"].tail(7).mean()),
        "vfc_estado": "Equilibrado",
        "vfc_base_min": 55,
        "vfc_base_max": 68,
        "sueno_score": int(u["sueno"]),
        "sueno_calidad": "Bueno",
        "sueno_horas": float(u["sueno_horas"]),
        "sueno_profundo_h": 1.4,
        "sueno_ligero_h": 4.1,
        "sueno_rem_h": 1.6,
        "sueno_despierto_h": 0.3,
        "bb_actual": 71,
        "bb_max": int(u["bb_max"]),
        "bb_min": int(u["bb_min"]),
        "estres": int(u["estres"]),
        "fc_reposo": int(u["fc_reposo"]),
        "pasos": 8420,
        "estado_entreno": "Productivo",
        "carga_aguda": 612,
        "carga_cronica": 540,
        "vo2_carrera": 54.0,
        "vo2_ciclismo": 56.0,
    }
    acts = [
        ("Rodaje suave", "running", 10200, 3300, 142, 2.8, 95),
        ("Fondo gran fondo", "road_biking", 92400, 11400, 138, 4.1, 260),
        ("Series 6x1000", "running", 11800, 3600, 158, 3.9, 180),
        ("Rodillo Z2", "indoor_cycling", 35000, 3900, 128, 2.6, 70),
        ("Tirada larga", "running", 21100, 6900, 148, 3.6, 210),
        ("Ruta con puertos", "road_biking", 74000, 10200, 145, 4.4, 290),
    ]
    filas = []
    for i, (nombre, tipo, dist, dur, fc, te, carga) in enumerate(acts):
        filas.append(
            {
                "activityName": nombre,
                "activityType": {"typeKey": tipo},
                "distance": dist,
                "duration": dur,
                "averageHR": fc,
                "aerobicTrainingEffect": te,
                "activityTrainingLoad": carga,
                "startTimeLocal": (datetime.combine(dia, datetime.min.time()) - timedelta(days=i + 1, hours=-7)).isoformat(),
            }
        )
    return Snapshot(
        fecha=dia,
        demo=True,
        nombre="Atleta demo",
        hoy=hoy,
        tendencia=tendencia,
        actividades=_normalizar_actividades(filas),
    )
