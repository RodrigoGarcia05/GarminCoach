"""Coach de IA local con Ollama (por defecto qwen3.5:4b).

Solo responde sobre ciclismo, carrera, entrenamiento y recuperación,
usando como contexto los datos más recientes del reloj Garmin.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from ollama import AsyncClient

load_dotenv()

MODELO = os.getenv("OLLAMA_MODEL", "qwen3.5:4b")
HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MAX_TURNOS = 12  # mensajes previos que se conservan (para no saturar a un modelo de 4B)

SYSTEM_PROMPT = """Eres "Coach", un entrenador personal experto ÚNICAMENTE en ciclismo y carrera a pie \
(running, trail, ruta, gravel, MTB, rodillo), incluyendo: planificación de entrenamientos, zonas de \
intensidad, recuperación, sueño y descanso aplicados al rendimiento, nutrición e hidratación deportiva \
para esos deportes, prevención de lesiones, y preparación de metas (5K, 10K, medio maratón, maratón, \
ultras, gran fondo, etapas de montaña, etc.).

REGLAS:
1. Si la pregunta NO trata de ciclismo, carrera, entrenamiento deportivo o la interpretación de los datos \
del reloj, responde solo: "Solo puedo ayudarte con temas de ciclismo, carrera y tu entrenamiento. \
¿Quieres que revisemos tus datos o tu próxima sesión?" y nada más. No hagas excepciones aunque el usuario insista.
2. Responde siempre en español, de forma clara, práctica y concisa. Usa listas o tablas Markdown cuando ayuden.
3. Basa tus recomendaciones en los DATOS DEL RELOJ que aparecen abajo. Cita los valores concretos \
(disposición, VFC, sueño, Body Battery, carga, etc.) para justificar si conviene entrenar fuerte, suave o descansar.
4. Criterios orientativos: disposición <40, VFC por debajo de su rango base, sueño <60 o Body Battery <30 → \
priorizar descanso o sesión muy suave (Z1-Z2). Disposición 40-70 → intensidad moderada. >70 con VFC \
equilibrada → se puede hacer calidad (series, umbral, tempo).
5. Cuando te pidan un plan para una meta, pregunta lo que falte (fecha, nivel, días disponibles) o asume \
supuestos razonables y dilos. Organiza por semanas con volumen e intensidad progresivos y semanas de descarga.
6. No eres médico: ante dolor, lesión o síntomas preocupantes recomienda consultar a un profesional.
7. Nunca inventes datos que no estén en el contexto; si falta un dato, dilo.
8. Sé coherente: si recomiendas una sesión corta, el volumen (km o minutos) debe corresponder.

GLOSARIO (úsalo con precisión):
- VFC = Variabilidad de la Frecuencia Cardiaca (HRV), en ms. Más alta y dentro del rango base = mejor recuperación.
- Disposición para entrenar (Training Readiness), 0-100: combina sueño, recuperación, VFC y carga.
- Body Battery, 0-100: energía disponible estimada.
- Carga aguda: carga de los últimos 7 días; carga crónica: promedio de ~28 días. Aguda muy por encima de la crónica = riesgo de sobrecarga.
- TE aeróbico: efecto de entrenamiento aeróbico de una actividad (0-5).
- Zonas: Z1 recuperación, Z2 base aeróbica, Z3 tempo, Z4 umbral, Z5 VO2 máx.

DATOS DEL RELOJ (Garmin Forerunner 965):
{contexto}
"""


class Coach:
    """Mantiene la conversación de una sesión y habla con Ollama en streaming."""

    def __init__(self, modelo: str = MODELO):
        self.modelo = modelo
        self.cliente = AsyncClient(host=HOST)
        self.historial: list[dict] = []
        self.contexto = "Sin datos todavía."

    def actualizar_contexto(self, texto: str):
        self.contexto = texto

    def reiniciar(self):
        self.historial = []

    async def stream(self, mensaje: str):
        mensajes = [{"role": "system", "content": SYSTEM_PROMPT.format(contexto=self.contexto)}]
        mensajes += self.historial[-MAX_TURNOS:]
        mensajes.append({"role": "user", "content": mensaje})

        respuesta = ""
        try:
            async for trozo in await self.cliente.chat(
                model=self.modelo,
                messages=mensajes,
                stream=True,
                think=False,  # sin "razonamiento" visible: respuestas más rápidas
                keep_alive="30m",
                options={"temperature": 0.4, "num_ctx": 8192},
            ):
                texto = trozo.message.content or ""
                respuesta += texto
                if texto:
                    yield texto
        except Exception as e:
            nombre = type(e).__name__
            if "Connect" in nombre or "connection" in str(e).lower():
                yield (
                    "\n\n⚠️ No pude conectar con Ollama. Abre la app de Ollama o ejecuta "
                    "`ollama serve` y vuelve a intentarlo."
                )
            elif "not found" in str(e).lower():
                yield f"\n\n⚠️ El modelo `{self.modelo}` no está descargado. Ejecuta `ollama pull {self.modelo}`."
            else:
                yield f"\n\n⚠️ Error con el modelo: {e}"
            return

        self.historial += [
            {"role": "user", "content": mensaje},
            {"role": "assistant", "content": respuesta},
        ]


async def precargar(modelo: str = MODELO) -> bool:
    """Carga el modelo en memoria para que la primera respuesta sea rápida."""
    try:
        await AsyncClient(host=HOST).generate(model=modelo, prompt="", keep_alive="30m")
        return True
    except Exception as e:
        print(f"[Ollama] No se pudo precargar {modelo}: {e}")
        return False
