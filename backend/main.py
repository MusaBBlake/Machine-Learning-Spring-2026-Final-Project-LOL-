from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import pickle
import os
import re

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model + champion data at startup ────────────────────────────────────
with open("model.pkl", "rb") as f:
    data = pickle.load(f)

model     = data["model"]
fcols     = data["fcols"]        # ordered list of champion IDs used as feature columns
name_to_id = data["name_to_id"]  # normalized_name -> champion_id
id_to_name = data["id_to_name"]  # champion_id -> display_name
all_ids   = data["all_ids"]      # list of all champion IDs

ROLES = {
    'top': [
        'Aatrox','Camille','Darius','Fiora','Gangplank','Garen','Gnar',
        'Gragas','Gwen','Illaoi','Irelia','Jax','Jayce','Kayle','Kennen',
        'Kled','Malphite','Maokai','Mordekaiser','Nasus','Ornn','Pantheon',
        'Poppy','Quinn','Renekton','Riven','Rumble','Sett','Shen','Shyvana',
        'Singed','Sion','Teemo','Tryndamere','Urgot','Vayne','Vladimir',
        'Volibear','Warwick','Yorick','Ambessa','KSante','Olaf'
    ],
    'jungle': [
        'Amumu','Belveth','Briar','Darius','Diana','Ekko','Elise',
        'Evelynn','Fiddlesticks','Gragas','Graves','Hecarim','Ivern',
        'Jarvan IV','Kayn',"Kha'Zix",'Kindred','Kled','LeeSin','Lillia',
        'MasterYi','MonkeyKing','Mordekaiser','Nidalee','Nocturne','Nunu',
        'Olaf','Pantheon','Poppy','Rammus','RekSai','Rengar','Sejuani',
        'Shaco','Shyvana','Skarner','Sylas','Taliyah','Trundle','Udyr',
        'Vi','Viego','Volibear','Warwick','XinZhao','Zac'
    ],
    'mid': [
        'Ahri','Akali','Akshan','Anivia','Annie','Aurelion Sol','Aurora',
        'Azir','Cassiopeia','Corki','Diana','Ekko','Fizz','Galio',
        'Gragas','Hwei','Irelia','Jayce','Kassadin','Katarina','Leblanc',
        'Lissandra','Lux','Malzahar','Naafiri','Neeko','Orianna','Pantheon',
        'Qiyana','Ryze','Sylas','Syndra','Taliyah','Talon','TwistedFate',
        'Veigar','Vex','Viktor','Vladimir','Xerath','Yasuo','Yone',
        'Zed','Ziggs','Zoe','Mel'
    ],
    'bot': [
        'Aphelios','Ashe','Caitlyn','Corki','Draven','Ezreal','Jhin',
        'Jinx','Kaisa','Kalista','KogMaw','Lucian','MissFortune','Nilah',
        'Quinn','Samira','Senna','Sivir','Smolder','Tristana','Twitch',
        'Varus','Vayne','Xayah','Yasuo','Yone','Zeri','Ziggs'
    ],
    'support': [
        'Alistar','Bard','Blitzcrank','Brand','Braum','Janna','Karma',
        'Lulu','Lux','Maokai','Milio','Morgana','Nami','Nautilus',
        'Neeko','Pyke','Rakan','Rell','Renata','Senna','Seraphine',
        'Shaco','Sona','Soraka','Swain','Tahm Kench','Taric','Thresh',
        "Vel'Koz",'Xerath','Yuumi','Zilean','Zyra','Zoe','Leona',
        'Galio','Heimerdinger'
    ]
}

def normalize(name: str) -> str:
    return re.sub(r"[ '\.]", "", name)

def make_vec(blue_team: List[str], red_team: List[str]) -> pd.DataFrame:
    vec = {col: 0 for col in fcols}
    for name in blue_team:
        cid = name_to_id.get(normalize(name))
        if cid is not None and cid in vec:
            vec[cid] = 1
    for name in red_team:
        cid = name_to_id.get(normalize(name))
        if cid is not None and cid in vec:
            vec[cid] = -1
    return pd.DataFrame([vec])[fcols]

def get_win_prob(blue_team, red_team):
    vec = make_vec(blue_team, red_team)
    prob = model.predict_proba(vec)[0]
    classes = list(model.classes_)
    return float(prob[classes.index(1)]) if 1 in classes else float(prob[1])


# ── Request/Response models ──────────────────────────────────────────────────
class PredictRequest(BaseModel):
    blue_team: List[str]
    red_team: List[str]
    team: str          # "blue" or "red"
    role: Optional[str] = None  # "top","jungle","mid","bot","support" or None for any

class WinProbRequest(BaseModel):
    blue_team: List[str]
    red_team: List[str]


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/champions")
def get_champions():
    """Return all champion names (display) for the frontend."""
    return {"champions": sorted(id_to_name.values())}

@app.post("/predict")
def predict_best_pick(req: PredictRequest):
    already_picked = set(
        name_to_id.get(normalize(n)) for n in req.blue_team + req.red_team
    )

    pool = ROLES[req.role] if req.role and req.role in ROLES else list(id_to_name.values())

    results = []
    for champ_name in pool:
        cid = name_to_id.get(normalize(champ_name))
        if cid is None or cid in already_picked:
            continue
        vec = make_vec(req.blue_team, req.red_team)
        vec[cid] = 1 if req.team == "blue" else -1
        prob = model.predict_proba(vec)[0]
        classes = list(model.classes_)
        p_blue = float(prob[classes.index(1)]) if 1 in classes else float(prob[1])
        p_team = p_blue if req.team == "blue" else 1 - p_blue
        results.append({"champion": champ_name, "win_prob": round(p_team, 4)})

    results.sort(key=lambda x: x["win_prob"], reverse=True)
    return {"recommendations": results[:10]}

@app.post("/winprob")
def win_probability(req: WinProbRequest):
    p = get_win_prob(req.blue_team, req.red_team)
    return {"blue_win_prob": round(p, 4), "red_win_prob": round(1 - p, 4)}

@app.get("/health")
def health():
    return {"status": "ok"}
