import asyncio
import numpy as np
from fastapi import FastAPI
from fastapi.responses import JSONResponse, HTMLResponse
from contextlib import asynccontextmanager
import requests, threading, time, re, statistics, json


# --- Globális változók ---
RSSI_VALUES = [999, 999, 999]       
ANCHORS = [(0, 0), (3, 0), (2.5, 3)]
latest_position = {"x": 0, "y": 0, "distances": [0, 0, 0], "rssi": [999, 999, 999], "rssi_raw": [0, 0, 0]}
HISTORY = []           # az utolsó mérések listája (pl. 20 elem)
STOP_FLAG = False

# --- Paraméterek ---
URL = "http://192.168.43.50/disp"
INTERVAL = 2           # másodpercenkénti frissítés
HISTORY_LEN = 5      # hány értéket tartson meg a stabilitás méréséhez
MIN_VALID = 0          # Jel erőssége: 0-100 tartomány
MAX_VALID = 100

def rssi_to_distance(rssi, rssi_min=25, rssi_max=65, d_min=0.1, d_max=5):
    """
    Jelerőssségből (0-100) távolságra konvertálás.
    A: referencia érték 1 méternél (alapértelmezés: 56)
    n: útcsillapítás faktor (alapértelmezés: 2)
    
    Képlet: távolság = (A / érték)^(1/n)
    """

    # clamp
    rssi = max(rssi_min, min(rssi, rssi_max))

    # normalizálás (0 → közel, 1 → távol)
    norm = (rssi - rssi_min) / (rssi_max - rssi_min)

    # nemlineáris finomhangolás (opcionális)
    norm = norm ** 2

    distance = d_min + norm * (d_max - d_min)
    return distance

def trilateration(p1, p2, p3, r1, r2, r3):
    A = 2 * (p2[0] - p1[0])
    B = 2 * (p2[1] - p1[1])
    C = r1**2 - r2**2 - p1[0]**2 + p2[0]**2 - p1[1]**2 + p2[1]**2
    D = 2 * (p3[0] - p2[0])
    E = 2 * (p3[1] - p2[1])
    F = r2**2 - r3**2 - p2[0]**2 + p3[0]**2 - p2[1]**2 + p3[1]**2
    
    denom = E * A - B * D
    if abs(denom) < 0.0001:
        return 0, 0  # szingularis matrix
    
    x = (C * E - F * B) / denom
    y = (C * D - A * F) / denom
    
    # Negativitás kezelése - abszolút értékek
    x = abs(x)
    y = abs(y)
   
    return x, y

def fetch_rssi_loop():
    """Háttérfolyamat: folyamatos RSSI lekérés és dinamikus szűrés"""
    global RSSI_VALUES, STOP_FLAG, HISTORY, latest_position

    while not STOP_FLAG:
        try:
            resp = requests.get(URL, timeout=5)
            resp.raise_for_status()
            
            # JSON feldolgozás
            data = resp.json()
            
            # Értékek kinyerése: ertek1, ertek2, ertek3
            if all(key in data for key in ["ertek1", "ertek2", "ertek3"]):
                new_values = [int(data["ertek1"]), int(data["ertek2"]), int(data["ertek3"])]                
                # Nyers adatok tárolása
                RSSI_VALUES = new_values.copy()              
                
                # Nyomtatás terminálba
                print(f"[NYERS]  ertek1={RSSI_VALUES[0]:3d}, ertek2={RSSI_VALUES[1]:3d}, ertek3={RSSI_VALUES[2]:3d}")                

                distances = [rssi_to_distance(v) for v in RSSI_VALUES]

                x, y = trilateration(ANCHORS[0], ANCHORS[1], ANCHORS[2], *distances)
                print(f"[POZICIO] x={x:.2f}, y={y:.2f}, tavolsagok={[round(d, 2) for d in distances]}")
                print("-" * 60)
                
                latest_position = {
                    "x": round(x, 2),
                    "y": round(y, 2),
                    "distances": distances, 
                    "rssi": RSSI_VALUES,
                    "rssi_raw": RSSI_VALUES
                }
            else:
                print("[HIBA] Hianyzo mezok: ertek1, ertek2, ertek3")

        except json.JSONDecodeError as e:
            print(f"[HIBA] JSON feldolgozas: {e}")
        except Exception as e:
            print(f"[HIBA] Lekeresben: {e}")
        time.sleep(INTERVAL)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI életcikluskezelés"""
    global STOP_FLAG
    STOP_FLAG = False
    thread = threading.Thread(target=fetch_rssi_loop, daemon=True)
    thread.start()
    print("[OK] Haatterfolyamat elindult...")
    try:
        yield
    finally:
        STOP_FLAG = True
        print("[STOP] Haatterfolyamat leallt.")


app = FastAPI(title="Dinamikus RSSI API", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def index():
    with open("webpage.html", "r", encoding="utf-8") as f:
        html = f.read()
    return HTMLResponse(content=html)

@app.get("/data")
async def get_data():
    """Aktuális pozíció, távolságok és RSSI értékek"""
    return JSONResponse(latest_position)

@app.get("/rssi")
def get_rssi():
    """Aktuális szűrt RSSI értékek"""
    return JSONResponse({"rssi": RSSI_VALUES})





if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port="5002")