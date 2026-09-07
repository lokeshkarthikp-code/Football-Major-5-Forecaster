"""
Canonical team registries for La Liga, Serie A, Bundesliga and Ligue 1.

Kept separate from names.py purely for size -- these are merged into the
single CANONICAL_TEAMS registry at import time, and resolution works
exactly as it does for the Premier League.

NAMING CONVENTIONS THIS HAS TO ABSORB
-------------------------------------
Football-Data.co.uk abbreviates aggressively and inconsistently:

    "Ath Madrid"    Atletico Madrid
    "Ath Bilbao"    Athletic Bilbao          <- different club, similar string
    "Espanol"       Espanyol                 <- their spelling, not a typo
    "Sociedad"      Real Sociedad
    "M'gladbach"    Borussia Monchengladbach
    "Ein Frankfurt" Eintracht Frankfurt
    "Paris SG"      Paris Saint-Germain
    "Inter"         Internazionale
    "Milan"         AC Milan                 <- and "Inter" is NOT Milan

football-data.org, by contrast, uses full legal names:
"Club Atletico de Madrid", "FC Bayern Munchen", "Paris Saint-Germain FC".

Both must resolve to the same canonical id or every cross-source join
silently drops rows.

THE DANGEROUS PAIRS
-------------------
Some clubs differ by one token and are genuinely distinct:
    Atletico Madrid  vs  Athletic Bilbao     ("Ath Madrid" / "Ath Bilbao")
    AC Milan         vs  Inter Milan          (both "Milan" in casual use)
    Real Sociedad    vs  Real Madrid
    Real Betis       vs  Real Madrid
    Borussia Dortmund vs Borussia Monchengladbach

The resolver never fuzzy-matches automatically, which is what keeps these
apart. Do not add a bare "Real" or "Borussia" alias to any of them.

VERIFY BEFORE TRUSTING
----------------------
This registry is written from known conventions, not scraped from the
live feed. Run `python scripts/audit_leagues.py` after the first
multi-league ingest -- it reports every unresolved spelling so gaps get
fixed against real data rather than assumed away.
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# Spain
# --------------------------------------------------------------------------
LA_LIGA: dict[str, list[str]] = {
    "Real Madrid": ["Real Madrid", "Real Madrid CF", "R Madrid"],
    "Barcelona": ["Barcelona", "FC Barcelona", "Barca", "FC Barcelona B"],
    "Atletico Madrid": [
        "Atletico Madrid", "Ath Madrid", "Atl Madrid", "Atletico",
        "Club Atletico de Madrid", "Atlético Madrid", "Atletico de Madrid",
    ],
    "Athletic Bilbao": [
        "Athletic Bilbao", "Ath Bilbao", "Athletic Club", "Athletic",
        "Bilbao", "Athletic Club de Bilbao",
    ],
    "Real Sociedad": ["Real Sociedad", "Sociedad", "Real Sociedad de Futbol"],
    "Sevilla": ["Sevilla", "Sevilla FC"],
    "Valencia": ["Valencia", "Valencia CF"],
    "Real Oviedo": ["Real Oviedo", "Oviedo"],
    "Villarreal": ["Villarreal", "Villarreal CF"],
    "Real Betis": ["Real Betis", "Betis", "Real Betis Balompie"],
    "Celta Vigo": ["Celta Vigo", "Celta", "RC Celta de Vigo", "Celta de Vigo"],
    "Espanyol": ["Espanyol", "Espanol", "RCD Espanyol", "RCD Espanyol de Barcelona"],
    "Getafe": ["Getafe", "Getafe CF"],
    "Osasuna": ["Osasuna", "CA Osasuna"],
    "Rayo Vallecano": ["Rayo Vallecano", "Vallecano", "Rayo", "Rayo Vallecano de Madrid"],
    "Girona": ["Girona", "Girona FC"],
    "Alaves": ["Alaves", "Deportivo Alaves", "Alavés", "Deportivo Alavés"],
    "Mallorca": ["Mallorca", "RCD Mallorca"],
    "Valladolid": ["Valladolid", "Real Valladolid", "Real Valladolid CF"],
    "Las Palmas": ["Las Palmas", "UD Las Palmas"],
    "Cadiz": ["Cadiz", "Cádiz", "Cadiz CF"],
    "Granada": ["Granada", "Granada CF"],
    "Elche": ["Elche", "Elche CF"],
    "Almeria": ["Almeria", "Almería", "UD Almeria", "UD Almería"],
    "Levante": ["Levante", "Levante UD"],
    "Eibar": ["Eibar", "SD Eibar"],
    "Leganes": ["Leganes", "Leganés", "CD Leganes", "CD Leganés"],
    "Huesca": ["Huesca", "SD Huesca"],
    "Malaga": ["Malaga", "Málaga", "Malaga CF"],
    "Deportivo La Coruna": [
        "Deportivo La Coruna", "La Coruna", "Deportivo", "RC Deportivo",
        "Deportivo de La Coruna", "RC Deportivo La Coruña"
    ],
    "Sporting Gijon": ["Sporting Gijon", "Sp Gijon", "Sporting de Gijon"],
    "Real Zaragoza": ["Real Zaragoza", "Zaragoza"],
    "Racing Santander": ["Racing Santander", "Santander", "Racing de Santander", "Real Racing Club de Santander"],
    "Cordoba": ["Cordoba", "Córdoba", "Cordoba CF"],
    "Tenerife": ["Tenerife", "CD Tenerife"],
}

# --------------------------------------------------------------------------
# Italy
# --------------------------------------------------------------------------
SERIE_A: dict[str, list[str]] = {
    # "Milan" alone means AC Milan in this feed. Inter is always "Inter".
    "AC Milan": ["AC Milan", "Milan", "A.C. Milan", "Associazione Calcio Milan"],
    "Inter Milan": [
        "Inter Milan", "Inter", "Internazionale", "FC Internazionale Milano",
        "Inter Milano",
    ],
    "Juventus": ["Juventus", "Juve", "Juventus FC"],
    "Napoli": ["Napoli", "SSC Napoli", "Naples"],
    "Roma": ["Roma", "AS Roma", "A.S. Roma"],
    "Lazio": ["Lazio", "SS Lazio", "S.S. Lazio"],
    "Atalanta": ["Atalanta", "Atalanta BC"],
    "Fiorentina": ["Fiorentina", "ACF Fiorentina"],
    "Torino": ["Torino", "Torino FC"],
    "Bologna": ["Bologna", "Bologna FC 1909"],
    "Udinese": ["Udinese", "Udinese Calcio"],
    "Sassuolo": ["Sassuolo", "US Sassuolo Calcio", "US Sassuolo"],
    "Sampdoria": ["Sampdoria", "UC Sampdoria"],
    "Genoa": ["Genoa", "Genoa CFC"],
    "Cagliari": ["Cagliari", "Cagliari Calcio"],
    "Empoli": ["Empoli", "Empoli FC"],
    "Verona": ["Verona", "Hellas Verona", "Hellas Verona FC"],
    "Lecce": ["Lecce", "US Lecce"],
    "Monza": ["Monza", "AC Monza"],
    "Salernitana": ["Salernitana", "US Salernitana 1919"],
    "Spezia": ["Spezia", "Spezia Calcio"],
    "Venezia": ["Venezia", "Venezia FC"],
    "Cremonese": ["Cremonese", "US Cremonese"],
    "Frosinone": ["Frosinone", "Frosinone Calcio"],
    "Como": ["Como", "Como 1907"],
    "Parma": ["Parma", "Parma Calcio 1913"],
    "Brescia": ["Brescia", "Brescia Calcio"],
    "Crotone": ["Crotone", "FC Crotone"],
    "Benevento": ["Benevento", "Benevento Calcio"],
    "SPAL": ["SPAL", "Spal"],
    "Chievo": ["Chievo", "Chievo Verona", "ChievoVerona"],
    "Palermo": ["Palermo", "US Citta di Palermo"],
    "Pescara": ["Pescara", "Delfino Pescara 1936"],
    "Carpi": ["Carpi", "Carpi FC 1909"],
    "Pisa": ["Pisa", "Pisa SC"],
}

# --------------------------------------------------------------------------
# Germany
# --------------------------------------------------------------------------
BUNDESLIGA: dict[str, list[str]] = {
    "Bayern Munich": [
        "Bayern Munich", "Bayern", "FC Bayern Munchen", "FC Bayern München",
        "Bayern Munchen", "Bayern München", "FC Bayern",
    ],
    # Never add a bare "Borussia" alias -- it is ambiguous with Gladbach.
    "Borussia Dortmund": [
        "Borussia Dortmund", "Dortmund", "BVB", "Ballspielverein Borussia Dortmund",
    ],
    "Bayer Leverkusen": [
        "Bayer Leverkusen", "Leverkusen", "Bayer 04 Leverkusen", "Bayer 04",
    ],
    "RB Leipzig": ["RB Leipzig", "Leipzig", "RasenBallsport Leipzig"],
    "Borussia Monchengladbach": [
        "Borussia Monchengladbach", "M'gladbach", "Mgladbach", "Gladbach",
        "Monchengladbach", "Mönchengladbach", "Borussia Mönchengladbach",
        "Bor. Monchengladbach",
    ],
    "Eintracht Frankfurt": [
        "Eintracht Frankfurt", "Ein Frankfurt", "Frankfurt", "Eintracht",
    ],
    "Wolfsburg": ["Wolfsburg", "VfL Wolfsburg"],
    "Hoffenheim": ["Hoffenheim", "TSG Hoffenheim", "TSG 1899 Hoffenheim", "TSG 1899"],
    "Elversberg": ["Elversberg", "SV Elversberg", "SV 07 Elversberg"],
    "Werder Bremen": ["Werder Bremen", "Werder", "SV Werder Bremen", "Bremen"],
    "Freiburg": ["Freiburg", "SC Freiburg"],
    "Mainz": ["Mainz", "Mainz 05", "1. FSV Mainz 05", "FSV Mainz 05"],
    "Union Berlin": ["Union Berlin", "Union", "1. FC Union Berlin"],
    "Stuttgart": ["Stuttgart", "VfB Stuttgart"],
    "Augsburg": ["Augsburg", "FC Augsburg"],
    "Koln": ["Koln", "Köln", "FC Koln", "1. FC Koln", "1. FC Köln", "Cologne"],
    "Hertha Berlin": ["Hertha Berlin", "Hertha", "Hertha BSC"],
    "Schalke 04": ["Schalke 04", "Schalke", "FC Schalke 04"],
    "Bochum": ["Bochum", "VfL Bochum", "VfL Bochum 1848"],
    "Heidenheim": ["Heidenheim", "1. FC Heidenheim 1846", "FC Heidenheim"],
    "Darmstadt": ["Darmstadt", "SV Darmstadt 98", "Darmstadt 98"],
    "St Pauli": ["St Pauli", "St. Pauli", "FC St. Pauli", "FC St Pauli"],
    "Holstein Kiel": ["Holstein Kiel", "Kiel", "Holstein"],
    "Hamburg": ["Hamburg", "Hamburger SV", "HSV", "Hamburg SV"],
    "Fortuna Dusseldorf": [
        "Fortuna Dusseldorf", "Dusseldorf", "Düsseldorf",
        "Fortuna Düsseldorf", "F Dusseldorf",
    ],
    "Arminia Bielefeld": ["Arminia Bielefeld", "Bielefeld", "Arminia"],
    "Greuther Furth": [
        "Greuther Furth", "Greuther Fürth", "Furth", "Fürth",
        "SpVgg Greuther Furth",
    ],
    "Paderborn": ["Paderborn", "SC Paderborn 07", "Paderborn 07"],
    "Hannover": ["Hannover", "Hannover 96"],
    "Nurnberg": ["Nurnberg", "Nürnberg", "1. FC Nurnberg", "1. FC Nürnberg"],
    "Ingolstadt": ["Ingolstadt", "FC Ingolstadt 04"],
    "Hamburger SV II": ["Hamburger SV II"],
}

# --------------------------------------------------------------------------
# France
# --------------------------------------------------------------------------
LIGUE_1: dict[str, list[str]] = {
    "Paris Saint-Germain": [
        "Paris Saint-Germain", "Paris SG", "PSG", "Paris Saint Germain",
        "Paris Saint-Germain FC", "Paris",
    ],
    "Marseille": ["Marseille", "Olympique de Marseille", "OM", "Olympique Marseille"],
    "Lyon": ["Lyon", "Olympique Lyonnais", "OL", "Olympique Lyon"],
    "Monaco": ["Monaco", "AS Monaco", "AS Monaco FC"],
    "Lille": ["Lille", "LOSC Lille", "LOSC", "Lille OSC"],
    "Nice": ["Nice", "OGC Nice", "OGC Nice Cote d'Azur"],
    "Rennes": ["Rennes", "Stade Rennais", "Stade Rennais FC", "Stade Rennais FC 1901"],
    "Lens": ["Lens", "RC Lens", "Racing Club de Lens"],
    "Saint-Etienne": [
        "Saint-Etienne", "St Etienne", "Saint Etienne", "Saint-Étienne",
        "AS Saint-Etienne", "AS Saint-Étienne", "ASSE",
    ],
    "Nantes": ["Nantes", "FC Nantes"],
    "Montpellier": ["Montpellier", "Montpellier HSC"],
    "Strasbourg": ["Strasbourg", "RC Strasbourg", "RC Strasbourg Alsace"],
    "Bordeaux": ["Bordeaux", "Girondins de Bordeaux", "FC Girondins de Bordeaux"],
    "Reims": ["Reims", "Stade de Reims"],
    "Brest": ["Brest", "Stade Brestois", "Stade Brestois 29"],
    "Le Mans": ["Le Mans", "Le Mans FC"],
    "Toulouse": ["Toulouse", "Toulouse FC"],
    "Angers": ["Angers", "Angers SCO", "SCO Angers"],
    "Lorient": ["Lorient", "FC Lorient"],
    "Metz": ["Metz", "FC Metz"],
    "Auxerre": ["Auxerre", "AJ Auxerre"],
    "Le Havre": ["Le Havre", "Le Havre AC", "Havre"],
    "Clermont": ["Clermont", "Clermont Foot", "Clermont Foot 63"],
    "Troyes": ["Troyes", "ES Troyes AC", "Troyes AC"],
    "Nimes": ["Nimes", "Nîmes", "Nimes Olympique"],
    "Dijon": ["Dijon", "Dijon FCO"],
    "Amiens": ["Amiens", "Amiens SC"],
    "Caen": ["Caen", "SM Caen", "Stade Malherbe Caen"],
    "Guingamp": ["Guingamp", "EA Guingamp"],
    "Ajaccio": ["Ajaccio", "AC Ajaccio"],
    "Bastia": ["Bastia", "SC Bastia"],
    "Nancy": ["Nancy", "AS Nancy Lorraine"],
    "Evian": ["Evian", "Evian Thonon Gaillard", "Evian TG"],
    "Gazelec Ajaccio": ["Gazelec Ajaccio", "GFC Ajaccio", "Gazelec"],
    "Paris FC": ["Paris FC"],
}


INTERNATIONAL_LEAGUES: dict[str, list[str]] = {
    **LA_LIGA, **SERIE_A, **BUNDESLIGA, **LIGUE_1,
}

__all__ = [
    "LA_LIGA", "SERIE_A", "BUNDESLIGA", "LIGUE_1", "INTERNATIONAL_LEAGUES",
]