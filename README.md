# 🌿 PlantDoc: Bitki Hastalığı Sınıflandırma Sistemi

Bu proje, **PlantDoc** veri setini kullanarak bitki yapraklarındaki hastalıkları tespit etmek için geliştirilmiş, derin öğrenme tabanlı bir sınıflandırma modelidir.

## 🚀 Öne Çıkan Özellikler
- **Model Mimarisi:** EfficientNet-B3 (Transfer Learning).
- **Hız:** Mixed Precision (AMP) ile GPU optimizasyonu.
- **Yerelleştirme:** 30 sınıf için Türkçe dil desteği.

## 📸 Örnek Veri (Dataset)
Eğitim setinden bir örnek (Patates Yaprağı):
![Patates Yaprağı](sample_potato_leaf.jpg)

## 📊 Görsel Sonuçlar
### Eğitim Eğrileri
![Eğitim Eğrileri](training_curves.png)

### Karışıklık Matrisi
![Karışıklık Matrisi](confusion_matrix.png)

## 🎯 Tahmin Örneği
Modelin bu görsel üzerindeki sonucu:
- **Sınıf:** Patates Yaprağı (Potato leaf)
- **Güven:** %67.7
