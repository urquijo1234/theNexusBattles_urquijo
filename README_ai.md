# IA Client 2 y Servicio de Acción

Pipeline de entrenamiento y servicio de inferencia para seleccionar acciones del héroe usando aprendizaje profundo (Keras) y cliente TypeScript que pelea en el servidor Socket.IO.

## Comandos rápidos

```bash
# Entrenar rápido (usa data/pve_turns2.ndjson)
python -m ai.train_action_selector --data AIclient/AIClient\(2\)/data/pve_turns2.ndjson --out ai/models

# Servir FastAPI
uvicorn ai.server:app --host 0.0.0.0 --port 9000

# Ejecutar AIclient2
npm run aiclient2

# Demo PvE (server + IA + clientes)
bash scripts/demo_pve.sh
```
