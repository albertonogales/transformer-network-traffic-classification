# Protocolo de Captura — Campaña IJIS Revisión (2026-11-17)

Objetivo: maximizar el número de semi-flujos por ventana para que el modelo tenga suficiente
tráfico que clasificar, y garantizar que los resultados sean defendibles ante los revisores.

---

## 0. Antes de empezar (hacer una sola vez)

1. **Instalar dependencias**
   ```bash
   tshark -v        # debe aparecer ≥ 4.0
   python3 -c "import numpy, pandas; print('OK')"
   ```

2. **Copiar la carpeta al Mac**
   ```bash
   cp -r <ruta_dropbox>/revision_IJIS/tiempo_real_mac ~/ijis_rt
   ```

3. **Verificar que los pesos existen**
   ```bash
   ls ~/ijis_rt/*.pkl   # debe listar pesos.pkl y pesos_articulo2023.pkl
   ```

4. **Configurar las variables de entorno** (una vez por sesión de terminal)
   ```bash
   cd ~/ijis_rt
   export RT_PESOS="$PWD/pesos.pkl:$PWD/pesos_articulo2023.pkl"
   export RT_DIR="$HOME/ijis_rt_resultados"
   ```

5. **Cerrar todo lo que consuma red**: Dropbox, iCloud Drive, Time Machine, actualizaciones
   de sistema, Chrome con sincronización, Slack, Teams, cualquier app en segundo plano.
   Comprobar con `nettop` o Activity Monitor → Network que solo queda la app objetivo.

6. **Preparar el diario de capturas** (fichero de texto o papel):
   ```
   Clase | Run | App y versión | Hora inicio | Actividad durante los 80 s | Incidencias
   ```

---

## 1. Lógica del script

`run_experiment_v2.sh <Clase> <Run>` captura **10 ventanas de 8 s** (= 80 s en total).
Cada ventana produce 4 filas en `results_log.csv`: 2 modelos × 2 preprocesamientos.
La clasificación sale en pantalla al terminar cada ventana — puedes ver `n_semiflows` en vivo.

**La aplicación objetivo debe estar generando tráfico ANTES de lanzar el script.**
Si arrancas la app durante la primera ventana, esa ventana puede salir con 0 semi-flujos.

---

## 2. Protocolo por clase — Campaña in-distribution (Runs 1–3)

### Chat
**App recomendada:** Telegram (escritorio) o cualquier app de mensajería sobre TLS.  
**Qué hacer durante los 80 s:**
- Mantén una conversación activa: envía mensajes de texto largos, audios, imágenes.
- Si tienes un interlocutor disponible, mejor; si no, envíate mensajes a ti mismo desde
  el móvil respondiendo rápido.
- El flujo TLS/WebSocket de chat genera ~5–15 paquetes por mensaje; necesitas flujos con
  ≥ 20 paquetes. Envía al menos 3–4 mensajes por ventana de 8 s, o mejor: mantén una
  llamada de audio en paralelo al chat (la llamada genera tráfico continuo).

**Señal de que va bien:** `n_semiflows ≥ 2` por ventana.

---

### Email
**App recomendada:** cliente de correo nativo (Mail.app) con cuenta IMAP + SMTP.  
Esta es la clase más difícil — el modelo tiene solo 168 semi-flujos de entrenamiento.

**Qué preparar ANTES del run:**
- Compón 3–5 emails con adjuntos grandes (PDF de ≥ 5 MB cada uno). Déjalos en borradores.
- Suscríbete temporalmente a una newsletter que mande emails con imágenes grandes, para
  tener descarga IMAP durante el run.

**Qué hacer durante los 80 s:**
- Envía uno de los borradores pesados al principio del run (tráfico SMTP de subida continuo).
- Mientras sube, descarga la bandeja de entrada con emails con adjuntos (tráfico IMAP).
- Lo que genera semi-flujos es la **transferencia continua de un adjunto grande** (un único
  flujo TCP con ≥ 20 paquetes). Un email de texto de 1 KB no genera semi-flujos.

**Alternativa más fiable:** conecta Mail.app a una cuenta que tenga 500+ emails sin descargar
(IMAP sin sincronización previa) y fuerza la sincronización durante el run.

**OOD — ProtonMail (web, Chrome):**
- Envía un email con adjunto de ≥ 5 MB desde proton.me en el navegador.
- El flujo HTTPS de subida genera tráfico continuo y semi-flujos.

**Señal de que va bien:** `n_semiflows ≥ 1` por ventana. Para Email, incluso 1 ya es útil.

---

### FileTransfer
**Esta es la clase más fácil — genera el mayor número de semi-flujos.**  
**App recomendada:** SCP o SFTP a un servidor propio (el mismo que usaste en 2026).

**Qué hacer durante los 80 s:**
- Transfiere un fichero de ≥ 500 MB: `scp archivo_grande.bin usuario@servidor:/tmp/`
- La transferencia debe estar ya en curso cuando lances el script (no empieces durante
  la primera ventana).
- Si no tienes servidor, `rsync` a un disco externo también genera SFTP local.

**Alternativa OOD — BitTorrent:**
- Abre un torrent de contenido libre (distribuciones Linux en archive.org).
- El tráfico P2P es muy diferente del SFTP/SCP de entrenamiento: múltiples conexiones
  cortas → puede generar menos semi-flujos que el SFTP. Deja el cliente corriendo con
  varios peers activos para compensar.

**Señal de que va bien:** `n_semiflows ≥ 20` por ventana.

---

### Streaming
**App recomendada in-distribution:** Spotify (escritorio) o YouTube en Chrome a 1080p.  
*(Spotify y YouTube SÍ están en ISCXVPN2016; Netflix también, pero la usaremos para OOD → ver abajo.)*

**Qué hacer durante los 80 s:**
- Reproduce audio/vídeo de forma continua. No pauses.
- Para Spotify: playlist de música (flujo continuo de datos de audio).
- Para YouTube: vídeo a 1080p o 4K (mayor bitrate = más paquetes = más semi-flujos).
- La reproducción debe haber empezado antes del script para evitar la descarga inicial
  del buffer (que es un burst corto, no flujo continuo).

**OOD — Amazon Prime Video:**
- Abre Prime Video en el navegador, reproduce una serie a la máxima calidad disponible.
- El flujo HTTPS adaptativo (DASH) genera tráfico muy continuo. Desactiva el límite de
  calidad si existe.

**Señal de que va bien:** `n_semiflows ≥ 15` por ventana.

---

### VoIP
**App recomendada in-distribution:** Skype (videollamada) o FaceTime.  
**Importante: usa videollamada, NO solo audio** — la cámara multiplica el bitrate y
garantiza más paquetes por segundo.

**Qué hacer durante los 80 s:**
- Llama a alguien (o usa una segunda cuenta/dispositivo).
- Mantén la cámara encendida y habla: el silencio activa el comfort noise y reduce
  el bitrate; habla o reproduce música frente al micro.

**OOD — Google Meet:**
- Exactamente igual: videollamada con cámara activa a través de meet.google.com.

**Señal de que va bien:** `n_semiflows ≥ 10` por ventana.

---

## 3. Orden recomendado de captura

Haz primero los 3 runs in-distribution de cada clase (runs 1–3), luego los 3 runs OOD
(runs 7–9) para separarlos claramente. Dentro de cada clase, haz los 3 runs seguidos
sin cambiar de app, para que las condiciones sean homogéneas.

```
In-distribution:
  FileTransfer runs 1,2,3   ← empieza aquí (fácil, para verificar que el script funciona)
  Streaming    runs 1,2,3
  VoIP         runs 1,2,3
  Chat         runs 1,2,3
  Email        runs 1,2,3   ← deja para el final, es la más difícil

OOD (después de un descanso):
  FileTransfer runs 7,8,9   (BitTorrent)
  Streaming    runs 7,8,9   (Amazon Prime Video)
  VoIP         runs 7,8,9   (Google Meet)
  Chat         runs 7,8,9   (WhatsApp Web)
  Email        runs 7,8,9   (ProtonMail)
```

Tiempo estimado: 15 runs × 80 s + 5 min de cambio de app × 10 cambios = ~35 min in-dist
                + 15 runs × 80 s OOD + cambios ≈ 35 min OOD
                = ~70 min en total.

---

## 4. Cómo lanzar cada run

```bash
cd ~/ijis_rt
# Asegúrate de que la app ya está generando tráfico
bash run_experiment_v2.sh FileTransfer 1
# Espera los 80 s, lee la salida en pantalla
# Anota en el diario: app, versión, hora, actividad, n_semiflows típico
bash run_experiment_v2.sh FileTransfer 2
bash run_experiment_v2.sh FileTransfer 3
```

Para los runs OOD, mismos comandos con run 7, 8, 9:
```bash
bash run_experiment_v2.sh Streaming 7   # Amazon Prime Video corriendo
bash run_experiment_v2.sh Streaming 8
bash run_experiment_v2.sh Streaming 9
```

---

## 5. Qué hacer si un run sale mal

| Problema | Acción |
|---|---|
| `n_semiflows == 0` en todas las ventanas | App no estaba generando tráfico. Anotar run como inválido en el diario. Usar el siguiente número de run. |
| `n_semiflows == 0` solo en 1–2 ventanas | Normal (pausa de buffering, etc.). El run es válido; esas ventanas se excluyen por el criterio pre-declarado. |
| TShark falla / permiso denegado | `sudo chmod +r /dev/bpf*` o reinstalar ChmodBPF desde Wireshark. |
| App generó tráfico mezclado (descarga de actualizaciones, etc.) | Anotar en el diario. El run sigue siendo válido (tráfico de fondo es ruido realista). |
| El run se interrumpe a mitad | No reutilizar ese número. Usar el siguiente run libre. |

**Regla de oro:** nunca borres un run aunque haya salido mal. El script se niega a
sobrescribir; si lo borras manualmente y repites con el mismo número, los revisores
no podrán verificar que el protocolo se siguió.

---

## 6. Tras la campaña

```bash
# Copiar resultados de vuelta a Dropbox
cp -r ~/ijis_rt_resultados <ruta_dropbox>/revision_IJIS/resultados/campana_nueva/
```

Verificar que el CSV tiene filas para todos los runs esperados:
```bash
python3 -c "
import pandas as pd
df = pd.read_csv('$HOME/ijis_rt_resultados/results_log.csv')
print(df.groupby(['class','run','preproc','capas','pe'])['n_semiflows'].agg(['count','mean','min']))
"
```

---

## 7. Diario de capturas (plantilla)

| Clase | Run | App | Versión | Hora inicio | Actividad | n_semiflows/ventana | Incidencias |
|---|---|---|---|---|---|---|---|
| FileTransfer | 1 | SCP | OpenSSH 9.x | 10:02 | scp 600 MB a servidor | ~25 | — |
| Streaming | 7 | Amazon Prime Video | Web (Chrome 130) | 10:30 | Serie en 1080p continua | ~18 | — |
| ... | | | | | | | |
