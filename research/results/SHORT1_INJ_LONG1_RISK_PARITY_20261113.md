# SHORT1 INJUSDT — LONG1 stop ve ROI TP uyarlaması
Tarih: 13 Kasım 2026 (VM UTC). SHORT1 dry-run; canlı SHORT emri gönderilmedi.

## LONG1 kaynak ayarları
LONG1: +%1,25 ROI hedefi (FIXED_ROI_TP_PCT=0.0125), giveback=0, SL_PCT=0.0299 fiyat stopu, LONG_DYNAMIC_STOP_CATASTROPHIC_PCT=0.035 fiyat, doğrulama=30 saniye, stall çıkışı kapalı, 3x kaldıraç.
LONG1 ROI_HARD_STOP_PCT=-0.135 uzak yedek korumadır; normal SL değil.

## SHORT1 .env değişiklikleri
- FIXED_ROI_TP_PCT 0.0400 -> 0.0125.
- FIXED_ROI_TP_GIVEBACK 0.0024 -> 0.
- ROI_HARD_STOP_PCT -0.1200 -> -0.0897. 3x kaldıraçta 2.99% aleyhte fiyat hareketine denk gelen normal SHORT hard/soft stop eşik değeri.
- SHORT_DYNAMIC_STOP_CATASTROPHIC_PCT 0.0600 -> 0.0350 (3.5% aleyhte fiyat hareketi).
Değişmeyen ayarlar: DRY_RUN=1, ARMED=0, SHORT_DYNAMIC_STOP_ENABLE=1, SHORT_DYNAMIC_STOP_CONFIRM_SEC=30, SHORT_NATIVE_STOP_ENABLE=1, SHORT_NATIVE_STOP_WORKING_TYPE=MARK_PRICE, leverage=3, bağımsız model doğrulaması açık.
SHORT1 stop yuvarlaması: Executor ile native stop reconciler geçerli Binance fiyat adımına aşağı yuvarlama yapar, stopu genişletmez.

## Çalışma mantığı
SHORT 15.000 girişte 3x: native stop ham 15.4485, 0.001 fiyat adımı varsayımıyla 15.448; katastrofik 15.525; teorik +1.25% ROI TP fiyatı 14.9375.
Binance native STOP_MARKET (canlıda kurulduğunda) fiyat eşiklerinde hemen tetiklenebilir, uygulamanın 30 saniyesini beklemez. STOP_MARKET stop fiyatında kesin kapanışı garanti etmez; kayma, komisyon, fonlama kaybı büyütebilir. Uygulama tabanlı +1.25% TP VM kapalı olduğunda çalışmaz.
LONG1 ROI_HARD_STOP=-.135 SHORT'a aynen kopyalanmadı: SHORT bu değeri doğrudan normal ve borsa native stopu için kullanıyor, bu halde zarar yaklaşık 4.5% fiyat hareketine genişlerdi.

## Güvenlik ve testler
- Binance INJUSDT işlem öncesi pozisyon boş, standart ve Algo emir sayısı 0 olarak doğrulandı.
- Geri alma yedekleri: research/backups/short1_long1_parity_20261113_211718/
- 12 SHORT test grubu başarılı; core/trade_executor_short.py ve scripts/short1_native_stop_reconciler.py derlendi.
- SHORT1 hâlâ dry-run; LONG1 supervisor ve koruyucu stop servisi aktif, SHORT1 izole shadow bus aktif; SHORT native stop otomasyon servisi henüz canlıya kurulmadı.
- Bir gözlemsel 9 shadow READY proxy testinde 1m kapanış fiyatıyla tüm 9 aday 60m içinde +1.25 ROI eşiğine ulaştı: bu canlı işlem veya backtest ispatı değildir, mum içi fiyat yolu ve gerçek fill/stop/komisyon doğrulanmadı. Dosya: research/results/SHORT1_INJ_9_SHADOW_READY_NEW_TP125_SL299_1M_CLOSE_PROXY_20261113.csv

## Sıradaki test
Çok daha fazla INJUSDT dönemiyle 1m OHLC advers-first replay, gerçek Binance ücretleri, stop kayması ve model skor -> Sniper -> Entry Confirmation -> Execution Risk -> Position Lifecycle zinciri.
