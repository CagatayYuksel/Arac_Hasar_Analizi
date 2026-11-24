# Araç Hasar Analizi — Tasarım Belgesi

Tarih: 2026-11-24 · Durum: onay bekliyor

## 1. Amaç ve neden

Fotoğraftan araç hasarını bulan, hasarın **hangi parçada** olduğunu ve parçanın **yüzde kaçını kapsadığını** söyleyen, **hafif / orta / ağır** sınıfı veren bir sistem.

- Başlangıç: ödev projesi. Hedef: sigorta ve ekspertiz şirketlerine satılabilir bir ürün, ayrıca son kullanıcılar için mobil uygulama.
- Projeyi **kullanıcı kendisi yazacak**; Claude yol gösterir, açıklar, kod inceler ve takıldığında yardım eder. Sıfırdan birlikte kurulacak, öğrenme süreci hedefin parçası.
- Doğru cevabın kaynağı kaporta ustaları ve ekspertiz uzmanlarıdır. Toplanan etiketler onlara gösterilip onaylatılacak.

## 2. Hedefler / hedef olmayanlar / başarısızlık sayılacaklar

**Hedefler**
- Tüm yaygın hasar türleri: çizik, göçük, çatlak, cam kırığı, far/lamba kırığı, patlak lastik.
- Her hasar için parça adı, parça içi hasar yüzdesi, hafif/orta/ağır sınıfı.
- Dolu hasarı (çok sayıda küçük göçük) — ayrı bir aşama.
- Tahmini onarım maliyeti — **aralık** olarak, son aşamada.
- Uzman etiketleriyle ölçülebilir doğruluk.

**Hedef olmayanlar (şimdilik)**
- Kesin TL tutarı vermek, hukuki bağlayıcı ekspertiz yerine geçmek.
- Aydınlatma düzeneği/istasyon kurmak (ileride konuşulacak).
- Mekanik/iç hasar, hasar öncesi-sonrası karşılaştırma.

**Anti-hedefler (teknik olarak çalışsa bile başarısızlık)**
- Aynı aracın fotoğrafları hem eğitimde hem testte (veri sızıntısı) → şişirilmiş skor.
- Plaka ve yüz içeren ham görüntüleri korumasız saklamak (KVKK).
- Sadece accuracy'ye bakıp küçük hasarları kaçıran model.

## 3. Kısıtlar
- Kontrollü ışık yok. Çekim kalitesini uygulama yönlendirecek (açı/mesafe/yansıma/bulanıklık uyarısı).
- Dolu için public veri yok; kendi veri toplanacak. Yoldan çekilen görüntüler çoğunlukla hasarsızdır.
- Hasar yüzdesi görüntüdeki **görünen alana** göre hesaplanır; açı/mesafe ile değişir → çok açılı çekim ve birleştirme zorunlu.
- CarDD görselleri Flickr/Shutterstock kaynaklı, lisans şartları kabul edilmeli; ticari kullanım öncesi her veri setinin lisansı ayrıca kontrol edilecek.

## 4. Yaklaşımlar (değerlendirilenler)

| | Yaklaşım | Artı | Eksi |
|---|---|---|---|
| A | İki segmentasyon modeli + kural tabanlı yüzde/şiddet | Hızlı, şeffaf | Eşikler tahmin |
| B | Tek model, "parça×hasar" sınıfları | Tek çıkarım | Sınıf patlaması, veri yetmez |
| **C** | **A ile başla, uzman etiketleriyle öğrenen şiddet modeline geç** | Doğruluk hedefine en gerçekçi yol | Uzman emeği gerekir |

**Seçilen: C.** B elendi (veri yetersizliği). Maliyet tahmini aralık olarak, üçüncü aşamada.

## 5. Mimari (kullanıcının onayladığı)

1. **Hasar modeli** (örnek segmentasyon): hasar maskesi + türü.
2. **Parça modeli** (örnek segmentasyon): araç parçalarının maskesi.
3. **Birleştirme:** her hasar maskesi, en çok örtüştüğü parçaya atanır; yüzde = hasar alanı / parça alanı.
4. **Şiddet:** önce eşik kuralları (A), sonra uzman etiketiyle eğitilmiş küçük model (C): girdi = yüzde, hasar türü, hasar sayısı, parça tipi.
5. **Çok açılı birleştirme:** aynı parçanın farklı açılardan sonuçları birleştirilir (kullanıcının onayladığı teknik uyarı).
6. **Çekim kalitesi kontrolü:** bulanıklık, mesafe, yansıma tespiti.

## 6. Aşamalar (yol haritası)

0. Ortam kurulumu, veri setlerini indirme ve inceleme.
1. Parça modeli (Carparts-seg ile ince ayar), değerlendirme.
2. Hasar modeli (CarDD ile), değerlendirme.
3. Birleştirme + yüzde + kural tabanlı şiddet (A), uçtan uca demo.
4. Kendi veri toplama: çekim, plaka/yüz bulanıklaştırma, etiketleme, **ayrılmış test seti (araç bazında)**.
5. Uzman etiketleri → şiddet modeli (C), uzmanla uyum ölçümü.
6. Çok açılı birleştirme + çekim kalitesi kontrolü.
7. Dolu hasarı: yüksek çözünürlük/karo (SAHI) ile küçük göçük tespiti, panel başına yoğunluk; Minor Dents veri seti ile ön eğitim.
8. Maliyet aralığı: uzman fiyat tablosu (parça × şiddet → aralık).
9. API (sunucu) ve mobil uygulama.

## 7. Kararlar (Claude'un kendi seçimleri, kullanıcıya düzeltme hakkı var)
- Dil: Python. Çatı: PyTorch + Ultralytics YOLO (segmentasyon). Sebep: hazır veri setleri ve eğitim akışı ile uyumlu, öğrenmesi kolay.
- Etiketleme aracı: Roboflow veya CVAT (ikisi de maske etiketler); aşama 4'te seçilecek.
- Başlangıç eşikleri (%5 / %20) geçicidir, aşama 5'te uzman verisiyle değişir.
- GPU yoksa Google Colab / Kaggle ortamı kullanılır (donanım bilgisi henüz alınmadı).
- Başarı hedeflerinin sayısal değerleri (mAP, mIoU, uzmanla uyum) ilk baz modeller ölçüldükten sonra belirlenir.

## 8. Yapıcıya bırakılanlar
Klasör yapısı, deney takibi aracı, augmentation ayrıntıları, mobil uygulama teknolojisi (aşama 9'da konuşulur).
