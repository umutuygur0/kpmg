# Portfolio Decision Engine

Türkiye makro/piyasa verisine (TCMB EVDS, FRED, Yahoo Finance, Investing.com)
dayanan, risk profiline göre dinamik varlık dağılımı öneren bir karar motoru.
FastAPI backend + tek sayfalık dashboard.

## Yapı

| Dosya | Görev |
|---|---|
| `portfolio_data_pipeline.py` | Dış kaynaklardan (EVDS, FRED, Yahoo, Investing.com) veri çeker, `portfolio_data/` altına yazar. |
| `decision_engine.py` | Ana karar motoru: makro veriden varlık skorları, ayarlamalar ve nihai dağılım hesaplar. CLI olarak da çalışır. |
| `backend.py` | `decision_engine.py`'yi HTTP API olarak açan FastAPI sunucusu; dashboard'u da servis eder. |
| `portfolio_dashboard.html` | Tek sayfalık frontend (canlı analiz + backtest). Tüm hesaplamayı backend'den çeker. |
| `stat_report.py` | Bir profil için motorun tüm ara adımlarını gösteren PDF şeffaflık raporu üretir. |
| `generate_decision_schema.py` | Motorun akış şemasını (`decision_schema.png`) üretir. |
| `testdata.py` | EVDS seri kodlarını hızlıca denemek için scratch script. |

## Kurulum

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt

cp .env.example .env          # sonra .env içine EVDS_API_KEY'i gir
```

EVDS API anahtarı ücretsiz: https://evds2.tcmb.gov.tr/index.php?/evds/login

`.env` dosyasını okumak için ortam değişkenlerini shell'de export etmeniz ya da
`python-dotenv` gibi bir araçla yüklemeniz gerekir; pipeline şu an sadece
`os.environ` okuyor.

## Veriyi çek

```bash
python portfolio_data_pipeline.py
```

`portfolio_data/processed/merged_aligned_daily.csv` dosyasını üretir —
motor bunu okur. Bu adım olmadan motor/backend çalışmaz.

## Backend + dashboard'u başlat

```bash
python backend.py
```

http://127.0.0.1:8000/ → dashboard
http://127.0.0.1:8000/docs → API dokümantasyonu (Swagger)

## CLI'dan doğrudan motor

```bash
python decision_engine.py --profile orta_riskli
python decision_engine.py --profile cok_riskli --date 2025-06-01 --save
```

## PDF rapor

```bash
python stat_report.py --profile orta_riskli
```

## Bilinen sınırlamalar

- **CDS 5Y verisi (Investing.com)**: Site scraping'e karşı koruma uyguluyor;
  şu an bu kaynak sık sık başarısız oluyor ve motor `cds_level`'ı `None`
  olarak işleyip devam ediyor (risk skoru diğer 4 bileşenden hesaplanır).
- **`turkey_10y_bond` (EVDS)**: Seri kodu (`TP.DT.TRY.10`) şu an EVDS'ten
  400 dönüyor; motor bu alanı kullanmıyor, sadece pipeline logunda başarısız
  görünüyor.
- Pipeline'daki EVDS/FRED/Yahoo çağrıları canlı dış servislere bağımlı;
  ağ kesintisinde ilgili seri o run için atlanır.
