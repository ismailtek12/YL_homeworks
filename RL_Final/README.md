# DQN Tabanlı Basit Depo Robotu Simülasyonu

##  Projenin Amacı

Bu proje, basit bir ızgara tabanlı depo ortamında çalışan bir robotun, takviyeli öğrenme (Reinforcement Learning) yöntemi olan **Deep Q-Learning (DQN)** kullanılarak görevleri öğrenmesini ve gerçekleştirmesini amaçlamaktadır. Robotun temel görevleri şunlardır:

- Belirli bir noktadan yük almak ve hedefe taşımak  
- Batarya seviyesini etkin şekilde yönetmek  
- Engellerden kaçınmak

---

##  Ortam Tanımı: `SimpleWarehouseEnv`

Özel olarak tanımlanmış `SimpleWarehouseEnv` sınıfı, `gym.Env` sınıfından türetilmiştir.

| Özellik          | Değer |
|------------------|-------|
| Izgara boyutu    | 5x5   |
| Yük pozisyonu    | (3, 3) |
| Hedef pozisyonu  | (4, 4) |
| Şarj istasyonu   | (0, 4) |
| Engeller         | [(2, 2)] |
| Batarya kapasitesi | 50 birim |

---

##  Aksiyon Uzayı

| Aksiyon Kodu | Açıklama      |
|--------------|---------------|
| 0            | Yukarı git    |
| 1            | Aşağı git     |
| 2            | Sağa git      |
| 3            | Sola git      |
| 4            | Yükü al       |
| 5            | Yükü bırak    |
| 6            | Şarj et       |

---

##  Ortamın Dinamikleri: `step()` Fonksiyonu

- Hareket aksiyonları bataryadan **1 birim enerji** tüketir.
- Engellerden geçiş engellenir ve **ceza puanı** verilir.
- Batarya sıfırlanırsa görev **başarısız** sayılır.
- Şarj istasyonuna ulaşıldığında batarya **tam şarj** edilir.
- Hedefe yük bırakıldığında **yüksek ödül** verilir.
- Yük taşınmış ve işlem bitmişse görev **başarıyla tamamlanır**.

---

##  Ödül Yapısı

| Durum                                | Ödül       |
|--------------------------------------|------------|
| Her adım                             | -0.05      |
| Engelle çarpışma                     | -1         |
| Yük alma                             | +30        |
| Yükü hedefte bırakma                 | +100       |
| Görevin tamamlanması                 | +200       |
| Bataryanın bitmesi                   | -200       |
| Maksimum adım sayısına ulaşma       | -100       |
| Şarj istasyonuna gitme              | +5         |

---

##  DQN Modeli

- **Giriş vektörü**: 7 boyutlu durum vektörü
- **Çıkış vektörü**: 7 boyutlu aksiyon uzayı

---
**Model mimarisi:**

Giriş (7) → Dense(128) + ReLU → Dense(64) + ReLU → Çıkış (7)

----

##  Modelin Karar Süreci

1. Güncel durum vektörü modele verilir.
2. Model, her aksiyon için bir Q-değeri üretir.
3. En yüksek Q-değerine sahip aksiyon seçilir (veya keşif için epsilon-greedy yöntemi uygulanır).
4. Seçilen aksiyon çevreye uygulanır, karşılık olarak yeni durum ve ödül alınır.
5. Bu deneyim, replay buffer’a kaydedilir.
6. Rastgele seçilen mini-batch ile ağ güncellenir.
