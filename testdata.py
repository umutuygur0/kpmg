# data.py - Genişletilmiş test
import os

import requests

headers = {"key": os.environ.get("EVDS_API_KEY", ""), "Accept": "application/json"}

testler = {
    "USD/TRY":       "https://evds3.tcmb.gov.tr/igmevdsms-dis/series=TP.DK.USD.A.YTL&startDate=01-01-2024&endDate=01-03-2024&type=json&frequency=1",
    "EUR/TRY":       "https://evds3.tcmb.gov.tr/igmevdsms-dis/series=TP.DK.EUR.A.YTL&startDate=01-01-2024&endDate=01-03-2024&type=json&frequency=1",
    "CPI (TP.FG.J0)":"https://evds3.tcmb.gov.tr/igmevdsms-dis/series=TP.FG.J0&startDate=01-01-2024&endDate=01-03-2024&type=json&frequency=5",
    "Rezerv (TP.AB.B1)":"https://evds3.tcmb.gov.tr/igmevdsms-dis/series=TP.AB.B1&startDate=01-01-2024&endDate=01-03-2024&type=json&frequency=5",
    # Mevduat faiz alternatifleri
    "Mevduat KTF10": "https://evds3.tcmb.gov.tr/igmevdsms-dis/series=TP.KTF10&startDate=01-01-2024&endDate=01-03-2024&type=json&frequency=5",
    "Mevduat KTF15": "https://evds3.tcmb.gov.tr/igmevdsms-dis/series=TP.KTF15&startDate=01-01-2024&endDate=01-03-2024&type=json&frequency=5",
    "Mevduat MKOFA": "https://evds3.tcmb.gov.tr/igmevdsms-dis/series=TP.MKOFA.T3&startDate=01-01-2024&endDate=01-03-2024&type=json&frequency=5",
    # Politika faiz alternatifleri
    "Politika TF1":  "https://evds3.tcmb.gov.tr/igmevdsms-dis/series=TP.PPONREPO.G1&startDate=01-01-2024&endDate=01-03-2024&type=json&frequency=1",
    "Politika TF2":  "https://evds3.tcmb.gov.tr/igmevdsms-dis/series=TP.PPIOFX.G1&startDate=01-01-2024&endDate=01-03-2024&type=json&frequency=1",
    "Politika TF3":  "https://evds3.tcmb.gov.tr/igmevdsms-dis/series=TP.O.N3&startDate=01-01-2024&endDate=01-03-2024&type=json&frequency=1",
}

for ad, url in testler.items():
    r = requests.get(url, headers=headers, timeout=20)
    preview = r.text[:120].replace("\n", " ")
    print(f"{ad:25s} → {r.status_code}: {preview}")