# SHORT1 INJUSDT — 10-14 Kasım 2026 Canlı Shadow Contra-Trend Analizi
Tarih: 14 Kasım 2026, VM UTC. Sadece gözlem/araştırma; canlı veya ana SHORT1 execution pipeline yetkisi yok.

## Kapsam ve veri kaynakları
- Redis telemetry:short1:inj:model_shadow_v5/v6, source routing OFF; her iki modelin aynı tamamlanmış 3m mumuna ait verileri eşleştirildi.
- BINANCE Futures public klines 1m, tam kapanmış 6375 mum, 10 Kasım 00:00 UTC–14 Kasım 10:14:59 UTC, yalnız bir kere 5 sayfalık read-only GET; yeni collector veya sürekli API sorgusu yok.
- Local CSV: research/short/data/INJUSDT_1m_20261110_20261114_live_contra_audit.csv.
- As-of sinyal: modelin gerçek emission timestamp; 1m geçmiş fiyat, 15m/60m return ve 15m taker alıcı akışı için sinyal anından ÖNCE tamamlanmış 1m mumlar.
- Gerçek UTC takvim sınırlarına göre 1m+3m+5m OHLC mumları toplandı; yalnız tamamen kapanmış 3m ve 5m mumları kullanıldı. Kayan 3 dakikalık fiyat farkı ile gerçek tamamlanmış 3m mumu aynı kabul edilmedi.
- Gelecek için ayrı markout: model çıktısından sonraki ilk 1m OPEN üzerinden, 15/30/60m net yönsel fiyat senaryosu; bu gelecekteki bilgi giriş kararında kullanılmadı.

## V6 skor ve giriş kapısı
- 1439 eşleşmiş V5/V6 tamamlanmış mum olayı, yeterli 60m geçmişi olan 1438 analiz edilebilir olay.
- 68 olay V6'nın kendi 0.808288 model eşik değerini aştı.
- Yalnız 10 tanesi ayrıca kaynak model confirmation state CONFIRM ve sniper progress koşulunu geçti.
- 15m ve 60m birlikte yükselişte toplam 413 V6 olay; skor kapısı geçen 15; shadow READY 3.
- Diğer rejimlerde skor kapısını geçen 53, READY 7.
- V6 hâlâ +%4 ROI / MAE <=2% fiyat hedefli model; SHORT1'in güncel +%1.25 ROI kâr hedefi ve stopu için eğitim semantiği uyuşmuyor. Bu sinyaller gerçek Master/Execution Risk/Final Safety onayından geçmedi. SHORT main/executor canlı çalışmıyor.

## Gerçek tamamlanmış 1m/3m/5m trendi
- 68 V6-score-passed olayın 17'sinde son kapanan 1m+3m+5m mumlar aynı anda bearish.
- 10 shadow READY adayının sıfırında 1m+3m+5m hep bearish koşulu mevcut.
- 10 shadow READY'nin 7'sinde geriye dönük üçer adet tamamlanmış 3m ve 5m mumun kapanış eğilimi beraber aşağı; kalan 3 olayda bu koşul yok. Her iki grup da 30m senaryosunda TP'yi gördü.
- 3 shadow READY olayı 15m+60m yükseliş rejimindeydi; bunların hiçbirinde üç timeframe aynı anda bearish değildi.
- Dolayısıyla yeni CLOSED-3m/5m dönüş teyidini HARD REJECT olarak uygulamak, bu örnekte tüm 10 shadow READY'yi eleyecek ve testte görülen tüm olumlu fiyat hareketlerini kaçıracaktı.

## Yalnız fiyat yolu simülasyonu: 10 shadow READY
- Gerçek model emission sonrası ilk 1m OPEN varsayımsal fill; 3x, +%1.25 brüt ROI TP, +%2.99 aleyhte fiyat SL.
- Fiyat yolu Binance public 1m OHLC yüksek/düşük; aynı mumda stop+TP varsa STOP önceliği. Varsayımsal roundtrip notional maliyet %0.08 komisyon + %0.06 slippage, funding yok. Gerçek marjin geri dönüşü ve limit/piyasa emir latency/fill yok.
- İlk 15m: 8/10 TP, 2/10 TIME_EXIT, ortalama varsayımsal net ROI +%0.5433.
- İlk 30m: 10/10 TP, net ROI +%0.8300 (hedef 1.25 brüt ROI minus 0.42 ROI maliyet). Stop ilk 30m'de hiçbirinde gelmedi.
- Üç uptrend READY olayının tamamı 30m içinde TP; uptrend HARD veto bunları kaybettirirdi. Sonuç sadece 3 olaydan oluşuyor, istatistiksel kanıt değil.
- NOT: Bunlar V6 SCORE + SHADOW CONFIRM gözlemleri, gerçek işlemler değil; PENDING→fresh-model bağımsız doğrulama, Market Context, gerçek Master, Sniper, Execution Risk, Final Safety, Binance real fill, native Algo stop ve risk rezervasyonu çalıştırılmadı.
- Toplam 10 olay küçük, korelasyonlu, aynı piyasa döneminden; geçmiş Temmuz–Eylül 60 log ile rejim etkisi çelişiyor. Seçim için tek kaynak olmamalı.

## Karar
1. SHORT1 DRY_RUN=1/ARMED=0; mevcut kâr hedefi +1.25 ROI, normal native stop -8.97 ROI (2.99 fiyat), catastrophe +3.5 fiyat DEĞİŞMEDİ.
2. 15m+60m yükselişi yalnız SHADOW CAUTION ve alt-grup analizi için uygundur; HARD VETO için destek yok.
3. 1m+3m+5m tamamlanmış tüm mumlar aşağı dönmeli koşulu sert zorunluluk yapılmamalı; 10/10 adayı eler.
4. PENDING yalnız gerçek yeni bağımsız source_model_event_id + yeni completed-3m close + kendi model skor eşiği + gerçek Master score gate ve sonraki Sniper/Confirmation/Risk güvenlik kapıları ile READY olabilir.
5. Uygulanacak bir sonraki gerçek geliştirme: V7 hedef modelin güncel bitiş verisine uzanan ayrı OOS testi ve SHORT1'in market-context & central risk altında dry-run canonical intent köprüsü. 10 adet fiyat-yolu başarılı diye model terfi ettirilmez.
6. LONG1 aktif ve kodu/pozisyonu bu araştırmada değiştirilmedi.

## Rapor dosyaları
- research/results/SHORT1_INJ_V6_LIVE_CONTRA_96H_EVENTS_20261114.csv
- research/results/SHORT1_INJ_V6_LIVE_CONTRA_96H_SUMMARY_20261114.json
- research/results/SHORT1_INJ_V6_SHADOW_READY_INTRABAR_TP125_SL299_20261114.csv
- research/results/SHORT1_INJ_V6_SHADOW_READY_INTRABAR_SUMMARY_20261114.json
- research/results/SHORT1_INJ_V6_96H_ALIGNED_CLOSED_1M3M5M_CONTEXT_20261114.csv
- research/results/SHORT1_INJ_V6_96H_ALIGNED_CLOSED_MTF_SUMMARY_20261114.json
- research/short/audit_inj_live_shadow_contra_96h_v1.py
- research/short/replay_inj_shadow_ready_exact_1m_v1.py
- research/short/audit_inj_exact_closed_mtf_v1.py
