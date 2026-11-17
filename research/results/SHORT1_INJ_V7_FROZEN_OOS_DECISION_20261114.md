# SHORT1 INJUSDT — V7 donmuş model, yeni 13 Ekim–14 Kasım dönemi testi
14 Kasım 2026 UTC. ARAŞTIRMA, canlı terfi YOK. LONG1 ve SHORT1 production config değiştirilmedi.

## Denetim
- Model: research/short/models/short_model_INJUSDT_3m_tp125_sl299_v7_research_only.joblib
- Model eşik: 0.7123981411511489, önceki validation sırasında kilitlenmiş. Bu testte model yeniden EĞİTİLMEDİ, threshold değiştirilmedi.
- Veri: 13 Ekim 2026 19:54 UTC – 14 Kasım 10:29 UTC arası 15,172 tane 3m Binance Futures public mum, 11 GET sayfası. CSV cache: research/short/data/INJUSDT_3m_20261013_20261114_FROZEN_OOS.csv
- Özelliklerin EMA/rolling warm-up verisi için orijinal araştırma datasının son 310 mumu geçici bellekte eklendi. İlk OOS sinyal eğitime dahil değil; eğitim verisine ekleme yapılmadı.
- Toplam 15,151 OOS sinyal barı, 1,033 skor eşiği geçişi. Model AUC = 0.58563. Modelin yeni dönem kalite kaybı gözlemlendi.
- Etiket: tamamlanmış 3m sinyalinden SONRA ilk 3m bar OPEN, hedef brüt ROI +%1.25 (3x), native SL +%2.99 fiyat, 60 dakika sonrası zorunlu sanal kapanış, 3m OHLC stop-first belirsizlik kuralı. Varsayımsal toplam notional fee+slippage %0.14; funding yok.
- Gerçek SHORT1'in 1m Sniper, gerçek emir fill, native Binance Algo Stop, merkezi risk ve zaman sınırı bu araştırma etiketinde çalıştırılmadı. Üretimde aynı 60m timeout yok. Bu gerçek bot performansı DEĞİL.

## Donmuş model performansı
| OOS dönemi | Eşikten geçen sinyal | 60m aralığı örtüşmeyen senaryo | Örtüşmeyen kazanan | Örtüşmeyen ort. varsayımsal net ROI |
|---|---:|---:|---:|---:|
| 13 Eki–14 Kas | 1033 | 83 | 69 | -0.3493% |
| 13–31 Eki | 696 | 54 | 44 | -0.6339% |
| 1–14 Kas | 337 | 29 | 25 | +0.1806% |
- 83 ayrık senaryoda 6 normal stop, 69 TP, diğerleri 60m sanal süre sonu. 1,033 birbirine yüksek ölçüde örtüşen aday bağımsız 1,033 işlem sayısı değildir. Ayrık 83 satır aynı equity sizing veya gerçek fill PnL değildir.
- Her 1,033 eşik geçişinde net ROI ortalaması -0.3728%. Ekim ~-0.5989%, Kasım ~+0.0942%.
- Eğitim+validation ve önceki testlerde seçilen cutoff kullanıldı. Bu yeni dönemde hiçbir cutoff ayarı yapılmadı.

## İleriye bakmadan sonradan rejim analizi (yalnız tanısal)
- OOS seçilmiş her aday için aynı 3m veri kümesinden geçmiş 15m/60m fiyat getirileri ve taker-buy flow computed. Yeni regime seçimleri bu OOS'de incelendiği için OOS performans sayıları TANISALDIR, üretim kuralı seçmek için yeni bağımsız dönem gereklidir.
- 15m+60m birlikte UP:
  - Ekim 308 skor adayında varsayımsal 3m stop-first net ROI ortalama -1.2528%, stop olay oranı %16.56.
  - Kasım 111 skor adayında ortalama +0.3422%, stop olay oranı %0.90.
- Aynı uptrend koşulu farklı dönemlerde ters davranıyor. Sert CONTRA-TREND REJECT güvenilir değildir. Kullanılacak ilk sonraki adım sadece ayrı SHADOW CAUTION raporu olabilir.
- Alt gruplarda ayrı ayrı 60m nonoverlap hesapları başka seçim pencereleri üretir; bu alt-grup sayıları ana 83 gözlemin ayrıştırılmış alt toplamı gibi toplanmamalı.

## Güvenlik kapısı kararı
- V7: NOT PROMOTED, research only. V5/V6 mevcut shadow modelleri değiştirilmedi.
- SHORT1 canlı açılmadı, DRY_RUN=1 ARMED=0; SHORT1 native stop servisinin LIVE kullanımı henüz yetkili değil.
- Model hedefinin TP/SL ile uyumu V7'de kısmen sağlansa da negatif dönemsel risk, yeni veri değişkenliği, gerçek 1m giriş ve 60m timeout uyuşmazlığı sebebiyle kanıt yetersiz.
- Normal MasterExecutor eski score threshold 0.555 kullanır; V7 0.7124 kaynak model doğrulama eşiğiyle farklıdır. Kaynak skoru başka skora dönüştürmek veya MasterContext boşken sahte READY oluşturmak yasak.

## Sonraki iş akışı
1. SHORT1 1m Entry Confirmation ve native stop + giveback lifecycle eşleşen fiyat-yolu replay; gerçek 1m emir fill ve komisyon tahmini. Ağustos–Kasım dönemi model events farklı olabilir; kaynak event ID olmadan gerçek intent sayılmayacak.
2. V5/V6 shadow ile V7 research skorlarını ayrı tut; 15m+60m UP ve flow yalnız telemetri/CAUTION etiketi, hard veto değil.
3. Model/stop beklenen getirisi net pozitif bağımsız dönemle kanıtlanana kadar yeni model ana SHORT1 emir hattına geçmez.
4. Yalnız LONG1+SHORT1 merkezileştirme: atomic Risk Master, stop reconciler, Redis/state izolasyonu. LONG2–4 ve SHORT2–4 aynen bekler.

## Teknik dosyalar
- research/short/audit_inj_v7_frozen_oos_oct_nov_v1.py
- research/short/audit_inj_v7_frozen_oos_regimes_v1.py
- research/results/SHORT1_INJ_V7_FROZEN_OOS_OCT_NOV_20261114.json
- research/results/SHORT1_INJ_V7_FROZEN_OOS_REGIME_DIAGNOSTICS_20261114.json
- research/results/SHORT1_INJ_V7_FROZEN_OOS_SELECTED_CONTEXT_20261114.csv
- research/results/SHORT1_INJ_V7_FROZEN_OOS_DAILY_20261114.csv
