# GarminCoach


# Mi Garmin · Coach IA

Dashboard en [Shiny for Python](https://shiny.posit.co/py/) con los datos de tu **Garmin Forerunner 965**
(disposición para entrenar, VFC, sueño, Body Battery, estrés, recuperación, carga y últimos entrenamientos)
y un **chat con un modelo local de Ollama** (`qwen3.5:4b`) que solo responde sobre **ciclismo y carrera**
usando tus datos como contexto.

## Archivos

| Archivo | Qué hace |
|---|---|
| `app.py` | La app Shiny (pestañas *Dashboard* y *Coach IA*) |
| `garmin_data.py` | Login en Garmin Connect (`garminconnect`), descarga y limpieza de datos, modo demo |
| `coach.py` | Cliente de Ollama con el prompt que limita el chat a ciclismo/carrera |

## Configuración (solo la primera vez)

1. Instala dependencias en el entorno virtual:

   ```bash
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Ten Ollama abierto con el modelo descargado:

   ```bash
   ollama pull qwen3.5:4b
   ```

3. En `.env` pon tu cuenta de Garmin Connect (ver `.env.example`):

   ```
   GARMIN_EMAIL=tu_correo@ejemplo.com
   GARMIN_PASSWORD=tu_contraseña
   ```

   Si dejas esos campos vacíos la app arranca en **modo demo** con datos de ejemplo.

## Ejecutar

```bash
source .venv/bin/activate
python -m shiny run --launch-browser app.py
```

(Se usa `python -m shiny` porque este `.venv` se copió de `basic-chat2` y el ejecutable `shiny` apunta a la
ruta vieja. Si prefieres, borra `.venv` y créalo de nuevo con `python3 -m venv .venv`.)

## Notas

- **Primer login / MFA:** si la cuenta tiene verificación en dos pasos, la app abre una ventana para escribir
  el código que te manda Garmin. Después los tokens se guardan en `.garmin_tokens/` y ya no se vuelve a pedir
  usuario, contraseña ni código (duran ~1 año). Puedes borrar la contraseña del `.env` después del primer login.
- Los datos se guardan en caché 10 minutos; el botón **Actualizar** fuerza una nueva descarga.
- Si Garmin responde "Too many login attempts", espera unos minutos antes de reintentar.
- El modelo se precarga al iniciar la app; la primera respuesta puede tardar unos segundos más.
- `.env` y `.garmin_tokens/` están en `.gitignore`: **no los subas a GitHub**.

