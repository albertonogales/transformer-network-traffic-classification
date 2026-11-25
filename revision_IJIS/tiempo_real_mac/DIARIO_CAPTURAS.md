# Diario de Capturas — Campaña IJIS Revisión

Fecha: _____________  Máquina: MacBook _____________ (modelo, RAM, macOS)  
TShark versión: _____________  Red: WiFi / Ethernet (rodear)

---

## PREPARACIÓN (rellenar antes de empezar)

- [ ] Dropbox / iCloud Drive pausados
- [ ] Time Machine desactivado
- [ ] Actualizaciones de sistema desactivadas
- [ ] Slack / Teams / Mail en segundo plano cerrados
- [ ] Chrome con sincronización desactivada (o cerrado excepto cuando se use)
- [ ] `nettop` abierto en otra ventana para vigilar tráfico de fondo
- [ ] Variables de entorno configuradas:
  ```bash
  export RT_PESOS="$HOME/ijis_rt/pesos.pkl:$HOME/ijis_rt/pesos_articulo2023.pkl"
  export RT_DIR="$HOME/ijis_rt_resultados"
  ```
- [ ] Test rápido (un run de prueba, no cuenta):
  ```bash
  bash ~/ijis_rt/run_experiment_v2.sh FileTransfer 0
  ```
  n_semiflows obtenidos: _______ → si > 0, el sistema funciona.

---

## CAMPAÑA IN-DISTRIBUTION

### FileTransfer — Run 1

**Antes de lanzar:**
- [ ] `scp` o descarga SFTP ya corriendo (fichero ≥ 500 MB en transferencia)
- [ ] Verificar en `nettop` que solo hay tráfico SCP/SFTP

```bash
bash ~/ijis_rt/run_experiment_v2.sh FileTransfer 1
```

App: _____________ Versión: _____________  
Hora inicio: _____________  
Actividad: _____________________________________________  
n_semiflows por ventana (leer de la salida): ___________________________  
Incidencias: _____________________________________________

---

### FileTransfer — Run 2

- [ ] Nueva transferencia en curso (misma app)

```bash
bash ~/ijis_rt/run_experiment_v2.sh FileTransfer 2
```

Hora inicio: _____________  
n_semiflows: ___________________________  
Incidencias: _____________________________________________

---

### FileTransfer — Run 3

```bash
bash ~/ijis_rt/run_experiment_v2.sh FileTransfer 3
```

Hora inicio: _____________  
n_semiflows: ___________________________  
Incidencias: _____________________________________________

---

### Streaming — Run 1

**Antes de lanzar:**
- [ ] Spotify / YouTube a 1080p ya reproduciendo (buffer precargado, no en pausa)
- [ ] Calidad forzada al máximo

```bash
bash ~/ijis_rt/run_experiment_v2.sh Streaming 1
```

App: _____________ Versión: _____________  
Hora inicio: _____________  
Actividad: _____________________________________________  
n_semiflows: ___________________________  
Incidencias: _____________________________________________

---

### Streaming — Run 2

```bash
bash ~/ijis_rt/run_experiment_v2.sh Streaming 2
```

Hora inicio: _____________  
n_semiflows: ___________________________  
Incidencias: _____________________________________________

---

### Streaming — Run 3

```bash
bash ~/ijis_rt/run_experiment_v2.sh Streaming 3
```

Hora inicio: _____________  
n_semiflows: ___________________________  
Incidencias: _____________________________________________

---

### VoIP — Run 1

**Antes de lanzar:**
- [ ] Videollamada Skype / FaceTime activa (cámara ON, hablando o música frente al micro)

```bash
bash ~/ijis_rt/run_experiment_v2.sh VoIP 1
```

App: _____________ Versión: _____________  
Hora inicio: _____________  
Actividad: _____________________________________________  
n_semiflows: ___________________________  
Incidencias: _____________________________________________

---

### VoIP — Run 2

```bash
bash ~/ijis_rt/run_experiment_v2.sh VoIP 2
```

Hora inicio: _____________  
n_semiflows: ___________________________

---

### VoIP — Run 3

```bash
bash ~/ijis_rt/run_experiment_v2.sh VoIP 3
```

Hora inicio: _____________  
n_semiflows: ___________________________

---

### Chat — Run 1

**Antes de lanzar:**
- [ ] Conversación activa abierta (Telegram / WhatsApp Web / similar)
- [ ] Interlocutor disponible O segunda cuenta lista para auto-respuesta

```bash
bash ~/ijis_rt/run_experiment_v2.sh Chat 1
```

App: _____________ Versión: _____________  
Hora inicio: _____________  
Actividad: _____________________________________________  
n_semiflows: ___________________________  
Incidencias: _____________________________________________

---

### Chat — Run 2

```bash
bash ~/ijis_rt/run_experiment_v2.sh Chat 2
```

Hora inicio: _____________  
n_semiflows: ___________________________

---

### Chat — Run 3

```bash
bash ~/ijis_rt/run_experiment_v2.sh Chat 3
```

Hora inicio: _____________  
n_semiflows: ___________________________

---

### Email — Run 1

**Antes de lanzar:**
- [ ] 3–5 borradores con adjuntos de ≥ 5 MB listos para enviar
- [ ] Mail.app con cuenta IMAP sin sincronizar (o bandeja con emails nuevos con adjuntos)
- [ ] Envía el primer borrador UNA VEZ que el script ya haya capturado la ventana 1

```bash
bash ~/ijis_rt/run_experiment_v2.sh Email 1
```
*(Nada más aparecer "Ventana 1", empieza a enviar el adjunto grande)*

App: _____________ Versión: _____________  
Hora inicio: _____________  
Actividad: _____________________________________________  
n_semiflows: ___________________________  
Incidencias: _____________________________________________

---

### Email — Run 2

```bash
bash ~/ijis_rt/run_experiment_v2.sh Email 2
```

Hora inicio: _____________  
n_semiflows: ___________________________

---

### Email — Run 3

```bash
bash ~/ijis_rt/run_experiment_v2.sh Email 3
```

Hora inicio: _____________  
n_semiflows: ___________________________

---

## DESCANSO ENTRE CAMPAÑAS

Verificación rápida antes de continuar con OOD:
```bash
python3 -c "
import pandas as pd
df = pd.read_csv('$HOME/ijis_rt_resultados/results_log.csv')
print(df[df['preproc']=='articulo'].groupby(['class','run'])['n_semiflows'].agg(['count','mean','min']))
"
```
Resultado: anotar clases con media < 2 semi-flujos/ventana y pensar si hay que repetir algún run.

---

## CAMPAÑA OOD (Runs 7–9)

> Para cada clase, usa la app OOD indicada. El protocolo de actividad es idéntico al
> in-distribution, pero con distinta aplicación. Rellena solo los campos que difieran.

### FileTransfer OOD — BitTorrent

**Antes de lanzar:**
- [ ] Cliente BitTorrent (qBittorrent) con torrent de distribución Linux descargando
- [ ] Al menos 5 peers activos (comprobar en la UI del cliente)

```bash
bash ~/ijis_rt/run_experiment_v2.sh FileTransfer 7
bash ~/ijis_rt/run_experiment_v2.sh FileTransfer 8
bash ~/ijis_rt/run_experiment_v2.sh FileTransfer 9
```

n_semiflows run 7: _____ run 8: _____ run 9: _____  
Incidencias: _____________________________________________

---

### Streaming OOD — Amazon Prime Video

**Antes de lanzar:**
- [ ] Prime Video en Chrome, reproduciendo a máxima calidad (serie o película activa)
- [ ] Calidad forzada a 1080p si la opción está disponible

```bash
bash ~/ijis_rt/run_experiment_v2.sh Streaming 7
bash ~/ijis_rt/run_experiment_v2.sh Streaming 8
bash ~/ijis_rt/run_experiment_v2.sh Streaming 9
```

n_semiflows run 7: _____ run 8: _____ run 9: _____  
Incidencias: _____________________________________________

---

### VoIP OOD — Google Meet

**Antes de lanzar:**
- [ ] Sala de Google Meet abierta con al menos un participante (o cuenta secundaria)
- [ ] Cámara y micrófono activos

```bash
bash ~/ijis_rt/run_experiment_v2.sh VoIP 7
bash ~/ijis_rt/run_experiment_v2.sh VoIP 8
bash ~/ijis_rt/run_experiment_v2.sh VoIP 9
```

n_semiflows run 7: _____ run 8: _____ run 9: _____  
Incidencias: _____________________________________________

---

### Chat OOD — WhatsApp Web

**Antes de lanzar:**
- [ ] web.whatsapp.com abierto y conectado en Chrome
- [ ] Conversación activa con interlocutor o cuenta secundaria en el móvil

```bash
bash ~/ijis_rt/run_experiment_v2.sh Chat 7
bash ~/ijis_rt/run_experiment_v2.sh Chat 8
bash ~/ijis_rt/run_experiment_v2.sh Chat 9
```

n_semiflows run 7: _____ run 8: _____ run 9: _____  
Incidencias: _____________________________________________

---

### Email OOD — ProtonMail

**Antes de lanzar:**
- [ ] proton.me abierto en Chrome, sesión iniciada
- [ ] Borradores con adjuntos de ≥ 5 MB listos para enviar

```bash
bash ~/ijis_rt/run_experiment_v2.sh Email 7
bash ~/ijis_rt/run_experiment_v2.sh Email 8
bash ~/ijis_rt/run_experiment_v2.sh Email 9
```

n_semiflows run 7: _____ run 8: _____ run 9: _____  
Incidencias: _____________________________________________

---

## CIERRE

```bash
# Copiar resultados a Dropbox
cp -r ~/ijis_rt_resultados <ruta_dropbox>/revision_IJIS/resultados/campana_nueva/
```

Verificación final:
```bash
python3 -c "
import pandas as pd
df = pd.read_csv('$HOME/ijis_rt_resultados/results_log.csv')
print('Total filas:', len(df))
print(df[df['preproc']=='articulo'].groupby(['class','run'])['n_semiflows'].agg(['count','mean','min','max']))
"
```

Resultado pegado aquí: ___________________________________________

Hora de finalización: _____________  
Observaciones generales: _____________________________________________
