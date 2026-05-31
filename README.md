# 🌿 PlantDoc: Bitki Hastalığı Sınıflandırma Sistemi

Bu proje, **PlantDoc** veri setini kullanarak bitki yapraklarındaki hastalıkları tespit etmek için geliştirilmiş, derin öğrenme tabanlı bir sınıflandırma modelidir.

## 🚀 Öne Çıkan Özellikler
- **Model Mimarisi:** EfficientNet-B3 (Transfer Learning).
- **Hız:** Mixed Precision (AMP) ile GPU optimizasyonu.
- **Yerelleştirme:** 30 sınıf için Türkçe dil desteği.
- **Optimizasyon:** Warmup + Cosine Annealing LR Scheduler.

## 📊 Görsel Sonuçlar
![Eğitim Eğrileri](training_curves.png)
![Karışıklık Matrisi](confusion_matrix.png)

## 🎯 Örnek Çıktı
Modelin bir tahmin örneği:
- **Sınıf:** Üzüm Yaprağı Kara Çürüklüğü
- **Güven:** %82.8
