# SHORT1 INJUSDT — Aşama 2 güvenlik ve mimari refactor
Tarih: 13 Kasım 2026. LONG1 canlı kodu ve servisleri bu aşamada değiştirilmedi.

## Coin sahipliği
- LONG1 = NEARUSDT; SHORT1 = INJUSDT. Diğer 6 stratejinin coinleri henüz belirlenmedi.
- core/strategy_symbol_policy.py strateji/coin eşleşmesini ve 8 ayrı coin atamasını doğrular.
- SHORT1 yanlış sembolde açılış işlemi hazırlamaz; 8 bot ayrı piyasa fakat ortak merkezi Risk Master.

## Merkezi risk
- Merkez profil ENFORCE_DRY_RUN, yalnız LONG1/SHORT1 doğrulanmış, max_open_positions=2, production_authorized=False.
- SHORT açılış kodunda daha önce sadece dry-run modunda merkezden risk rezervasyonu vardı: yanlışlıkla canlıya geçerse merkez baypas edilebiliyordu.
- Açılışa central_risk_live_not_authorized kapısı eklendi. Mevcut üretim profili onaylanmadan DRY_RUN=0 yaparak SHORT açılması mümkün değil.
- Rezervasyonsuz merkezi risk sorgusu: context multiplier 1.0 -> READY, ~14.38 USDT; multiplier 0.5 -> REJECT, ~7.19 USDT (10 USDT minimum altında). Mode REDUCED. Anlık bakiye değişebilir.
- Kaldıraç, risk ayarı veya fon aktarımı değiştirilmedi.

## Binance koruyucu stop
- core/short_native_algo_stop.py güncel USD-M Algo STOP_MARKET BUY, positionSide SHORT, closePosition true, algoId takibi ve yalnız SHORT1'e ait emri iptal eder.
- Executor artık gerçek Binance pozisyonRisk entryPrice + quantity doğrulandıktan sonra SHORT stopunu hesaplar; tahmini model fiyatı kullanılmaz.
- SHORT stop ROI_HARD_STOP_PCT=-0.12, kaldıraç 3x: yaklaşık +%4 fiyat hareketi üzerindeki stop; LONG stopuyla aynı değildir.
- MARKET CLOSE bildirimi sonrası Binance SHORT sıfır miktar doğrulanmadan mevcut native stopu iptal etmez.
- scripts/short1_native_stop_reconciler.py tek seferlik read-only çalıştırıldı, INJUSDT FLAT_CLEAN. DRY_RUN=1 iken --apply yasak. Sürekli servis henüz kurulmadı veya başlatılmadı.

## Model PENDING doğruluğu
- Geçmiş SHORT1 Redis: 205 intent, 163 READY ve 42 PENDING_CONTEXT. Geçmiş exec olaylarında açık kaynak-sinyal ID ve tamamlanmış 3m close zamanı yoktu.
- Model event_id ve completed_close_time artık Aggregator -> MasterExecutor -> IntentBridge boyunca açık alan olarak korunuyor.
- core/short_pending_freshness_contract.py farklı yeni intent, aynı model sağlayıcısı, daha yeni kapanmış mum, azami beş dakika bayatlık ve yeni model gate skoru şartlarını test eder.
- SHORT_PENDING_REQUIRE_NEW_MODEL_PROOF=1 ile mevcut eski skorun PENDING yeniden denemesi emir açamaz. Upstream henüz bağımsız yeni model kanıtını pending kaydına taşımıyor: bu nedenle mevcut PENDING yeniden açılışları fail-closed reddedilir. Yeni normal READY intent doğrudan değerlendirilebilir.
- Sonraki iş: gerçek yeni model kanıtını pending motoruna taşıyıp replay testleriyle performansı doğrulamak.

## Veri/maliyet ve durum
- V5 ve V6 economic shadow aynı worker dosyasını kullanıyor fakat farklı model dizinleri ve farklı Redis streamleri var; körlemesine durdurulmadı.
- SHORT1 exec stream son görülen 31 Ekim, signal stream 2 Kasım 2026; 13 Kasım'da taze SHORT giriş hattı doğrulanmış değil.
- SHORT main çalışmıyor, DRY_RUN=1, ARMED=0. Yeni SHORT sürekli servis eklenmedi.
- VM disk doluluğu ~91%. Gereksiz büyük dosya veya yeni collector oluşturulmadı.

## Yeni/değişen dosyalar
core/trade_executor_short.py
core/short_native_algo_stop.py
core/strategy_symbol_policy.py
core/short_pending_freshness_contract.py
orchestration/aggregator/aggregator.py
orchestration/executor/master_executor.py
orchestration/executor/intent_bridge.py
scripts/short1_native_stop_reconciler.py
research/tests/test_short_native_algo_stop_v1.py
research/tests/test_short_close_protection_v1.py
research/tests/test_short_coin_risk_live_guard_v1.py
research/tests/test_short_reconciler_v1.py
research/tests/test_short_actual_fill_native_stop_v1.py
research/tests/test_short_pending_freshness_contract_v1.py
research/tests/test_short_pending_enforcement_v1.py
research/tests/test_short_provenance_bus_v1.py

## Gelecek sıra
1. PENDING'in yeni gerçek model kanıtını üretme, 3m/5m bağlamı ve ücret sonrası replay.
2. Sekiz farklı coin için canlı Risk Master sermaye/pozisyon limitleri, merkezi muhasebe.
3. SHORT1 için başlangıçta/pozisyon sonrası Binance Algo stop uzlaştırma servisi, önce tam dry-run test.
4. SHORT1 PID, Redis state, servis ve log izolasyonu. Sonra kontrollü canlı test; şimdilik canlıya geçilmedi.
