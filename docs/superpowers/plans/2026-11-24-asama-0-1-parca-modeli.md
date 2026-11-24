# Aşama 0–1: Ortam, Veri ve Parça Modeli — Uygulama Planı

> **Bu plan kimin için:** Kodu **sen yazıyorsun**. Claude her adımda neyi neden yaptığını açıklar, yazdığın kodu inceler, takıldığında yardım eder; senin yerine yazmaz. Adımlar `- [ ]` kutularıyla ilerler.

**Hedef:** Python ortamını kur, parça segmentasyon veri setini doğrula ve incele, ilk parça modelini (baseline) eğit, sonuçlarını ölç.

**Mimari:** Küçük, tek sorumluluklu Python modülleri (`src/hasar/`) ve testleri. Saf (pure) fonksiyonlar `pytest` ile test edilir; eğitim ve değerlendirme Ultralytics YOLO segmentasyonunu sarar.

**Teknoloji:** Python 3.11, PyTorch, Ultralytics (YOLO segmentasyon), OpenCV, NumPy, PyYAML, pytest.

**Spec:** `docs/superpowers/specs/2026-11-24-arac-hasar-analizi-design.md` (aşama 0 ve 1)

## Global Kısıtlar
- Dil: Python. Çatı: PyTorch + Ultralytics YOLO (segmentasyon).
- Test seti **araç bazında** ayrılır; aynı aracın görüntüsü hem train hem val/test'te olamaz (veri sızıntısı yok).
- Her veri setinin lisansı kullanılmadan önce kontrol edilir ve `docs/data-licenses.md` içine yazılır (ticari kullanım hedefi).
- Ham veri ve model ağırlıkları git'e girmez.
- GPU yoksa Colab/Kaggle kullanılır; kod `device` parametresiyle CPU/GPU'ya uyar.
- Başarı hedeflerinin sayısal değerleri bu planın sonunda, baseline ölçüldükten sonra belirlenir.

## Review Focus
1. Proje yolu boşluk içeriyor (`Arac Hasar Analizi`): tüm yol işlemleri `pathlib.Path` ile yapılır, boşluklu yol çalışmalı.
2. Etiket dosyası boş (görüntüde hiç parça yok): hata değil, "boş etiket" olarak sayılmalı.
3. Etiket dosyası eksik olan görüntü: rapor edilmeli, sessizce atlanmamalı.
4. Sınıf kimliği (class id) tanımlı sınıf sayısının dışında: rapor edilmeli.
5. Aynı dosya adı hem train hem val'de: sızıntı olarak rapor edilmeli.

---

### Task 1: Proje iskeleti ve ortam kontrolü

**Files:**
- Create: `requirements.txt`, `.gitignore`, `pyproject.toml` (pytest ayarı), `src/hasar/__init__.py`, `src/hasar/env_check.py`
- Test: `tests/test_env_check.py`

**Interfaces:**
- Produces: `check_environment() -> dict[str, str | bool]` — anahtarlar: `python`, `torch`, `ultralytics`, `cuda_available`.

- [ ] **Step 1:** Sanal ortam oluştur ve etkinleştir (`python -m venv .venv`), `git init` yap. `.gitignore` içine `.venv/`, `data/`, `runs/`, `*.pt`, `__pycache__/` ekle.
- [ ] **Step 2: Başarısız testi yaz** (`tests/test_env_check.py`)

```python
from hasar.env_check import check_environment

def test_check_environment_keys():
    info = check_environment()
    assert set(info) >= {"python", "torch", "ultralytics", "cuda_available"}
    assert isinstance(info["cuda_available"], bool)
```

- [ ] **Step 3:** `pytest tests/test_env_check.py -v` çalıştır. Beklenen: FAIL (modül yok).
- [ ] **Step 4:** `check_environment()` fonksiyonunu `src/hasar/env_check.py` içinde yaz; sürümleri `sys`, `torch`, `ultralytics` modüllerinden oku. Bağımlılıkları `requirements.txt`'e yazıp kur; `pip install -e .` ile `src/` yolunu tanıt.
- [ ] **Step 5:** `pytest tests/test_env_check.py -v` çalıştır. Beklenen: PASS. Çıktıdaki `cuda_available` değerine bak: `False` ise eğitimi Colab/Kaggle'da yapacağız (bana söyle).
- [ ] **Step 6: Commit:** `git add . && git commit -m "chore: proje iskeleti ve ortam kontrolü"`

### Task 2: Veri seti indirme ve doğrulama

**Files:**
- Create: `src/hasar/data/__init__.py`, `src/hasar/data/verify_dataset.py`, `docs/data-licenses.md`
- Test: `tests/test_verify_dataset.py`

**Interfaces:**
- Produces: `DatasetReport` (dataclass: `images: dict[str, int]`, `missing_labels: list[str]`, `empty_labels: int`, `bad_class_ids: list[str]`, `cross_split_duplicates: list[str]`) ve `verify_yolo_dataset(root: Path, num_classes: int) -> DatasetReport`.
- Veri düzeni (Ultralytics standardı): `root/{train,valid,test}/images/*.jpg` ve `root/{train,valid,test}/labels/*.txt`.

- [ ] **Step 1: Veri setini indir.** Ultralytics `datasets_dir` ayarını proje içindeki `data/` klasörüne yönlendir ve Carparts-seg'i indir (`docs.ultralytics.com/datasets/segment/carparts-seg`). Lisansını okuyup `docs/data-licenses.md`'ye yaz (kaynak, lisans, ticari kullanım durumu).
- [ ] **Step 2: Başarısız testleri yaz** (geçici klasörde (`tmp_path`) minik sahte veri seti kur). Her biri için bir test:
  - `test_counts_images_per_split`: 2 train, 1 valid görüntü → `report.images == {"train": 2, "valid": 1}`
  - `test_missing_label_reported`: etiketsiz görüntü → dosya adı `missing_labels` içinde
  - `test_empty_label_counted_not_error`: boş `.txt` → `empty_labels == 1`, `missing_labels == []`
  - `test_bad_class_id_reported`: `num_classes=3` iken `7 0.1 0.1 ...` satırı → dosya adı `bad_class_ids` içinde
  - `test_cross_split_duplicate_reported`: aynı dosya adı train ve valid'de → `cross_split_duplicates` içinde
  - `test_path_with_spaces`: `tmp_path / "Arac Hasar Analizi"` altında kurulan veri seti hatasız çalışır
- [ ] **Step 3:** `pytest tests/test_verify_dataset.py -v`. Beklenen: hepsi FAIL.
- [ ] **Step 4:** `verify_yolo_dataset` fonksiyonunu yaz. Her etiket satırının ilk sayısı sınıf kimliğidir. Çift kayıt kontrolü dosya adının (uzantısız) bölümler arasında tekrarına bakar.
- [ ] **Step 5:** `pytest tests/test_verify_dataset.py -v`. Beklenen: hepsi PASS.
- [ ] **Step 6:** Fonksiyonu gerçek veri üzerinde çalıştır (`num_classes=23`). Raporu oku, bulduğun sorunları bana göster. Sızıntı çıkarsa birlikte karar veririz.
- [ ] **Step 7: Commit:** `git add . && git commit -m "feat: yolo veri seti doğrulama"`

### Task 3: Veri keşfi (sınıf dağılımı ve görsel inceleme)

**Files:**
- Create: `src/hasar/data/explore.py`, `scripts/explore_dataset.py`
- Test: `tests/test_explore.py`

**Interfaces:**
- Consumes: Task 2'nin veri düzeni.
- Produces: `class_distribution(labels_dir: Path, class_names: list[str]) -> dict[str, int]` (sınıf adı → örnek sayısı) ve `draw_polygons(image: np.ndarray, label_path: Path, class_names: list[str]) -> np.ndarray` (poligonları ve sınıf adlarını çizilmiş kopya döndürür, girdiyi değiştirmez).

- [ ] **Step 1: Başarısız testleri yaz:**
  - `test_class_distribution_counts`: 2 dosyada 3 poligon (iki `0`, bir `1`) → `{"a": 2, "b": 1}`; hiç geçmeyen sınıf `0` ile listede yer alır
  - `test_draw_polygons_returns_copy`: dönen dizi girdiyle aynı şekilde (`shape`), girdi değişmemiş, dönen dizi girdiden farklı
- [ ] **Step 2:** `pytest tests/test_explore.py -v`. Beklenen: FAIL.
- [ ] **Step 3:** İki fonksiyonu yaz. Poligon koordinatları 0–1 arası normalize, görüntü boyutuyla çarpılır (`cv2.polylines` / `cv2.putText` kullan).
- [ ] **Step 4:** `pytest tests/test_explore.py -v`. Beklenen: PASS.
- [ ] **Step 5:** `scripts/explore_dataset.py` ile sınıf dağılımını yazdır ve 20 rastgele görüntüyü çizili kaydet. Bak: hangi sınıflar çok az? Etiketler görüntüyle uyuşuyor mu? Bulgularını bana anlat; bu bulgular eğitim ayarlarını (sınıf ağırlığı vb.) etkileyecek.
- [ ] **Step 6: Commit:** `git add . && git commit -m "feat: veri keşfi araçları"`

### Task 4: Baseline parça modeli eğitimi

**Files:**
- Create: `src/hasar/train_parts.py`, `configs/parts.yaml` (Ultralytics veri yapılandırması)
- Test: `tests/test_train_parts.py`

**Interfaces:**
- Produces: `train_parts(data_yaml: Path, epochs: int = 50, imgsz: int = 640, device: str = "cpu", fraction: float = 1.0, project: Path = Path("runs")) -> Path` — en iyi ağırlık dosyasının (`best.pt`) yolunu döndürür.

- [ ] **Step 1: Başarısız testi yaz** (`@pytest.mark.slow`, varsayılan çalıştırmada atlanır): küçük bir veri alt kümesiyle `epochs=1, fraction=0.01, imgsz=320` çağrısı → dönen yol vardır ve `.pt` ile biter.
- [ ] **Step 2:** `pytest tests/test_train_parts.py -v -m slow`. Beklenen: FAIL.
- [ ] **Step 3:** `train_parts` fonksiyonunu Ultralytics `YOLO("yolo11n-seg.pt").train(...)` ile yaz; `best.pt` yolunu eğitim sonucundan al. `configs/parts.yaml` veri yollarını ve 23 sınıf adını içerir.
- [ ] **Step 4:** `pytest tests/test_train_parts.py -v -m slow`. Beklenen: PASS.
- [ ] **Step 5:** Gerçek eğitimi başlat (küçük model, 50 epoch; GPU yoksa Colab'da). Eğitim eğrilerine (loss, mAP) bak ve ne gördüğünü bana anlat: aşırı öğrenme (overfitting) belirtisi var mı?
- [ ] **Step 6: Commit:** `git add . && git commit -m "feat: baseline parça modeli eğitimi"`

### Task 5: Parça modeli değerlendirmesi ve hedeflerin belirlenmesi

**Files:**
- Create: `src/hasar/evaluate_parts.py`, `docs/results/parts-baseline.md`
- Test: `tests/test_evaluate_parts.py`

**Interfaces:**
- Consumes: Task 4'ün `best.pt` dosyası.
- Produces: `weakest_classes(per_class: dict[str, float], k: int = 5) -> list[tuple[str, float]]` (en düşük skorlu `k` sınıf, artan sırada) ve `evaluate_parts(weights: Path, data_yaml: Path, split: str = "test") -> dict` — anahtarlar: `mask_map50`, `mask_map50_95`, `per_class` (sınıf adı → mask mAP50-95).

- [ ] **Step 1: Başarısız testi yaz:** `test_weakest_classes_sorted`: `{"a": 0.9, "b": 0.2, "c": 0.5}`, `k=2` → `[("b", 0.2), ("c", 0.5)]`; `test_weakest_classes_k_larger_than_len`: `k=10` hata vermez, 3 öğe döner.
- [ ] **Step 2:** `pytest tests/test_evaluate_parts.py -v`. Beklenen: FAIL.
- [ ] **Step 3:** İkisini de yaz; `evaluate_parts` Ultralytics `model.val(split=...)` sonuçlarını sözlüğe çevirir.
- [ ] **Step 4:** `pytest tests/test_evaluate_parts.py -v`. Beklenen: PASS.
- [ ] **Step 5:** Test setinde değerlendir, en zayıf 5 sınıfı bul. Birkaç tahmini görüntü üzerinde çiz ve hataları incele (yanlış parça mı, kaçırılmış parça mı?).
- [ ] **Step 6:** `docs/results/parts-baseline.md` içine metrikleri, zayıf sınıfları ve gözlemlerini yaz. Birlikte, tasarım belgesinin 7. bölümündeki "sayısal başarı hedeflerini" bu sonuçlara bakarak belirleriz ve belgeye ekleriz.
- [ ] **Step 7: Commit:** `git add . && git commit -m "feat: parça modeli değerlendirmesi ve baseline sonuçları"`

---

## Self-Review notları
- **Spec kapsamı:** Aşama 0 (ortam, veri indirme/inceleme) → Task 1–3; aşama 1 (parça modeli, değerlendirme) → Task 4–5. Lisans kontrolü (spec §3) → Task 2 Step 1. Veri sızıntısı (spec §2 anti-hedef) → Task 2 testi ve Global Kısıtlar. Sayısal hedeflerin sonradan belirlenmesi (spec §7) → Task 5 Step 6.
- **Tip tutarlılığı:** `DatasetReport`, `verify_yolo_dataset`, `class_distribution`, `draw_polygons`, `train_parts`, `weakest_classes`, `evaluate_parts` adları tüm görevlerde aynı.
- **Kapsam dışı (sonraki planlar):** Hasar modeli (aşama 2), birleştirme ve oran (aşama 3) ve sonrası. Bu aşamaların sonuçları bu planın çıktılarına bağlı olduğu için ayrı planlarda yazılacak.
