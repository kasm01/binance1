# LONG1 + SHORT1: iki botlu üretim tamamlama kapıları
13 Kasım 2026 UTC, yalnız iki coin: LONG1=NEARUSDT, SHORT1=INJUSDT.
LONG2-4 ve SHORT2-4 eski aday env/model dosyaları korunur. Güncel eğitim veya otomatik çoğaltma yapılmaz.

## Denetlenen mevcut durum
LONG1 supervisor, Algo native stop sidecar ve SHORT1 shadow bus aktif.
SHORT1 DRY_RUN=1, ARMED=0; ana executable SHORT işlem süreci çalışmıyor.
Binance read-only pozisyon sorgusunda NEARUSDT/INJUSDT boş ve iki coin için Algo emir yok.
LONG1 stop attach kodu başarısız olunca CRITICAL log yazıyor; otomatik acil güvenli kapanış ve kalıcı yeniden giriş bloke zinciri yok.
SHORT1 native stop script dosyası var ama devamlı systemd servisi yok; merkezi Risk Master ENFORCE_DRY_RUN, production_authorized=False. SHORT1 canlıya geçişi kodla bloke.
SHORT1 V5/V6 canlı araştırma kayıtları geliyor; izole shadow:short1:inj:signal_bus:v1 büyüyor. Normal SHORT production sinyalleri ~11-13 gün eski; gerçek uçtan uca short intent akışı kapalı.

## Kapı 1: LONG1 native stop hata güvenliği — P0
Başarı şartları:
1. Pozisyon fill teyit edilir; native STOP_MARKET Binance Algo Order oluşturulup GET ile doğrulanır.
2. POST timeout belirsizliği önce exchange order sorgusuyla çözülür, ikinci stopu körlemesine göndermeyiz.
3. Stop doğrulanamazsa kalıcı yeni giriş kilidi (Redis + process) oluşturulur.
4. Mevcut gerçek LONG pozisyonu exchange API üzerinden doğrulanmadan MARKET CLOSE gönderilmez; doğrulanmışsa tek fiziksel close gateway ile kapatılır.
5. MARKET CLOSE sonrası Binance positionRisk LONG miktarı 0 doğrulanır. Sonuç bilinmiyorsa otomatik tekrar MARKET emri gönderilmez; kırmızı alarm ve kilit korunur.
6. Servis yeniden başladıktan sonra kilit kendiliğinden kaldırılmaz; yalnız exchange/Algo/telemetry doğrulaması sonrası kontrollü açılır.
7. Flat senaryosunda hiçbir emir oluşturulmadan fake Redis/mocked exchange testleri ve dry-run simülasyonları; ancak sonra pozisyonsuz bakım penceresinde canlı kod restart.
Mevcut bu turda LONG1 kodunda değişiklik YAPILMADI; acil kapanış taslağı uzak kod yazma güvenlik engeline takıldı. Risk hâlâ açık.

## Kapı 2: SHORT1 shadow -> production DRY-RUN audit — P1
- V6 source ID + completed 3m close + score + model threshold açık taşınmalı; V5 kalite danışmanlığı farklı hedef.
- Fresh PENDING intent gerçek yeni kaynaktan gelmeli, origin tekrar kullanılamaz.
- Yeni model olayı tekrar model score, sniper, 1m Entry Confirmation, market context, Execution Risk, Final Safety'yi geçmeli.
- Dry-run Signal Bus production akışlarından farklı stream isimleri ve günlük/haftalık metrikler.
- Shadow READY sayısı canlı ORDER yetkisi sayılmaz; her RET/REJECT reason telemetry.

## Kapı 3: SHORT1 native stop / lifecycle / boot
- Gerçek SHORT fill entryPrice & qty doğrula; doğru BUY STOP_MARKET closePosition, algoId ile exchange GET verification.
- Binance hesabı FLAT ise yalnız SHORT1 tagged stale orders temizle.
- Stop POST başarısızlığında güvenli tek close / belirsiz durum kilidi testleri.
- VM boot/read-only reconcile; canlı yetki verilene kadar --apply çalışan SHORT stop servisi oluşturma.
- Pozisyon kapandıktan sonra exchange SHORT qty=0 doğrulamadan native stop temizlenmez.

## Kapı 4: İki botlu merkezi risk
- İki farklı coin olsa da aynı equity: max total notional, directional gross exposure, DD governor, fee/funding.
- Redis transaction + idempotent reservation/confirm/release; botlardan birinin çökmesi diğerinin sermaye rezervasyonunu bozmamalı.
- Merkezi canlı authorize LIVE profile mevcut değildir; dry-run merkez REDUCED modunda 0.5 market multiplier ile INJ SHORT notional 10 USDT minimum altına düşebiliyor: sırf işlem açılsın diye risk veya min notional zorlamayacağız.
- LONG1'in mevcut canlı emir hattında ortak production risk gate tamamlanmadan SHORT1 canlıya açılmayacak.

## Kapı 5: izolasyon, performans
- Ayırılmış logs, PID, service and Redis state; ortak market collector, ortak Risk Master, ayrı model skorları.
- Fee/slippage/funding ve stop/TP fiyat yolunu 1m high/low advers-first ile test et; LONG1 yeni native stop sonrası ilk canlı fill proof ve SHORT1 ilk shadow->executor dry run proof.
- Görev tamamlanma ölçütü: LONG1+SHORT1 bağımsız coinlerde aynı merkezden risk yetkisi, tek pozisyon, doğrulanmış borsa stopu, güvenli kapanış, outcome feedback.
