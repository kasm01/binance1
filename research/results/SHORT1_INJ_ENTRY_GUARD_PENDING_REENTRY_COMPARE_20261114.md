# SHORT1 INJUSDT — Giriş savunması / PENDING / stop sonrası dönüş denetimi
14 Kasım 2026. Yalnız araştırma. SHORT1 DRY_RUN=1 ve ARMED=0 kaldı. LONG1 değiştirilmedi.

## Kaynak ve metodoloji
- Gerçek eski loglardan 60 SHORT1 INJ işlemi: 21 Temmuz–2 Eylül, 23 ROI zarar, 37 kazanç.
- Giriş 1m/3m/5m/15m/60m özellikleri yalnız girişten ÖNCE tamamlanmış mumlar üzerinden alındı. Sonraki 1m fiyat verileri yalnız geriye dönük çıktı incelemesi için kullanıldı.
- Kaynak: research/results/SHORT1_INJ_REAL_TRADES_ENTRY_1M3M5M_REBOUND_20261114.csv
- Eski log pozisyon stopları mevcut SHORT1 stop ile AYNI DEĞİL. Orijinal net8 bir getiri oranı ve USDT karşılığı notional ile yaklaşıklandı.
- Yeni karşılaştırmada varsayımsal 3x SHORT; TP +1.25% brüt margin ROI; SL +2.99% aleyhte fiyat; 60m süre sonunda kapanış; 0.08% notional çift yönlü komisyon + 0.06% slip, funding yok, aynı 1m mumda stop ve TP birlikteyse STOP önceliği.
- 60m süre kuralı gerçek SHORT1 lifecycle içinde yok. Bu, gerçek Binance işlemleriyle eşdeğer backtest değildir. Tarihsel girişler kısmen örtüşüyor; toplam ROI yüzdeleri hesap getirisi değildir.

## Girişte yön farkları
- Son 1m yeşil 60/60 eski işlemde mevcut: zarar 23, kazanç 37. Tek başına ret filtresi olmamalı.
- 15m ve 60m birlikte yükselişte 20 işlem; 10 zarar, 10 kazanç (%50 zarar). Diğer 40 işlemde 13 zarar (%32.5).
- Son üç adet 3m yükseliş: 27 işlemde 11 zarar; son üç adet 5m yükseliş: 29 işlemde 10 zarar. Sadece 5m yükselişe ret uygulamak yararlı görünmüyor.

## Eski işlemleri 15m+60m UP ile eleme
- Loglardan tahmini eski net8 USDT toplam -0.8504; 20 yükseliş girişini engellemek, 10 zararla birlikte 10 kazananı da kaldırır ve tarihsel yaklaşık net sonucu +0.1763 USDT yapar. Bu geçmişe uyarlanmış tarama, ileri performans kanıtı değil.
- Aynı yeni synthetic çıkış kurallarında hemen 60 giriş ROI toplamı -11.4549 yüzde puanı, yükseliş sert veto uygulanırsa -3.0723 yüzde puanı; ikisi de negatif.
- Aylık ayrım: Temmuz veto kötüleşiyor (-4.019 yerine -4.806 ROI toplam); Ağustos +5.810 yerine +4.980; Eylül -13.246 yerine -3.247. Dolayısıyla rejim değişiminde filtre kararsız, HARD REJECT önerilmez.

## 1m+3m+5m satış onaylı PENDING
- Son tamamlanmış üç timeframe bearish + aşağı close-to-close. Yalnız onaydan sonraki 1m OPEN varsayımsal giriş. Opsiyonel 3m toplam taker SELL flow şartı.
- PENDING 5m, flow yok: 60'tan 29 giriş kabul; 16 eski kaybeden ve 15 eski kazananı engelledi; kabul edilenlerde yeni synthetic ortalama net ROI -0.0019%; medyan giriş gecikmesi 165sn.
- PENDING 5m, negatif flow şartı: 23 giriş; ortalama -0.0072%.
- PENDING 10m, flow yok: 38 giriş; ortalama -0.1733%.
- PENDING 15m, flow yok: 46 giriş; ortalama -0.1289%.
- EŞLEŞMİŞ KARŞILAŞTIRMA: aynı 29 fırsat gecikmesiz açılsa yeni exit ile ortalama +0.1639%, gecikmeli -0.0019%. Giriş gecikmesi -0.1658 yüzde puan daha kötü; görünür iyileşme çoğunlukla az işlem açma etkisidir.
- Kaynak: research/results/SHORT1_INJ_ENTRY_GUARDS_SAME_EXIT_BASELINE_20261114.json

## Stop sonrası geri dönüş ve yeni SHORT giriş ihtimali
- Eski 23 stop sonrası 60m içinde 14 kez eski SHORT giriş fiyatına, 240m içinde 21 kez döndü. 240m içinde 15 kez eski girişten +1.25% ROI fiyatını gördü.
- Eski girişe dönenlerde medyan bekleme 37m; hedefe ulaşanlarda medyan 82m. Geri dönüşten önce stop kapanışından ekstra yukarı fiyat: medyan 0.319%, maksimum 1.345%. Stop kaldırmak kanıtlanmış avantaj değil.
- Eski stop sonrası 1m+3m+5m yeni bearish doğrulaması 15m içinde 13/23; yeni varsayımsal 60m net ROI ortalama -0.0239%, 7 pozitif. 30m içinde 18/23, net -0.0593%; 60m içinde 23/23, net -0.1637%.
- Bu olaylar yeni bağımsız MODEL sinyali değildir. Asla gerçek SHORT1 emir yetkisi sayılmaz.
- Kaynak: research/results/SHORT1_INJ_POST_STOP_REENTRY_PROXY_SUMMARY_20261114.json

## Tek başına stop mesafesi çözmüyor
- Sabit TP+1.25 ROI ve 60m horizon ile stop +0.30%, +0.50%, +1%, +1.50%, +2%, +2.99% fiyat değerlerinin tamamında 60 fırsatın net ROI ortalaması NEGATİF.
- En iyi görünen +1.50% stop da -0.1458% ortalama net ROI; bu seçimi mevcut araştırma örnekleminden üretime taşımak metodolojik olarak hatalı.
- Kaynak: research/results/SHORT1_INJ_STOP_TP125_1M_OHLC_SENSITIVITY_20261114.csv

## Karar
1. LONG1/SHORT1 canlı parametrelerinde değişiklik yok. SHORT1 dry-run.
2. 15m+60m yukarı yönü SHADOW CAUTION/PENDING etiketi olarak deneyelim; şimdilik HARD RET gerekmez.
3. Candle-only stop sonrası reentry canlıya açılmaz; gerçek yeni V5/V6 ya da yeni hedefe uygun MODEL kaynak ID, tamamlanmış mum, Master, Sniper, Entry Confirmation ve Risk Master onayı gerekir.
4. Çıkış modelini gerçek lifecycle'a uygun olarak daha geniş, bağımsız güncel dönemlerde 1m adverse-first replay ile ölçelim.
