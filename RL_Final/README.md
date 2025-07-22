DQN Tabanlı Basit Depo Robotu Simülasyonu

Projenin Amacı
Bu proje, basit bir ızgara tabanlı depo ortamında çalışan bir robotun, takviyeli
öğrenme (Reinforcement Learning) yöntemi olan Deep Q-Learning (DQN)
kullanılarak görevleri öğrenmesini ve gerçekleştirmesini amaçlamaktadır. Robot,
belirli bir noktadan yük alarak hedefe ulaştırmak, batarya seviyesini yönetmek ve
engellerden kaçınmak zorundadır.




Ortam Tanımı: SimpleWarehouseEnv
SimpleWarehouseEnv, gym.Env sınıfından türetilerek oluşturulmuş özel bir ortamdır.
Bu ortamda:
● Izgara boyutu: 5x5
● Yüklerin başlangıç yeri: (3, 3)
● Hedef pozisyonu: (4, 4)
● Şarj istasyonu: (0, 4)
● Engeller: [(2, 2)]
● Batarya kapasitesi: 50 birim




Aksiyonlar: 0-yukarı,1-aşağı,2-sağa,3-sola,4-yükü al,5-yükü bırak,6-şarj et




Ortamın Dinamikleri: step() Fonksiyonu
● Hareket eylemleri bataryadan 1 birim tüketir.
● Engellerden geçilmez; ceza puanı uygulanır.
● Batarya 0 olursa görev başarısız kabul edilir.
● Şarj istasyonuna ulaşıldığında batarya dolabilir.
● Hedefe yük bırakıldığında yüksek ödül verilir.
● Yük biterse ve taşıma bitmişse görev tamamlanır.



Ödül Yapısı

Durum Ödül

Her adım -0.05

Engelle çarpışma -1

Yük alma +30

Yük bırakma (hedefte) +100

Tüm görev tamamlandıysa (bitiş) +200

Batarya bitmesi -200

Maksimum adım sayısına
ulaşma

-100

Şarj istasyonuna gitme +5 +
kazanç



DQN Modeli
Giriş vektörü: 7 boyutlu durum vektörü
Çıkış vektörü: 7 boyutlu aksiyon uzayı
Model mimarisi:
● 7 giriş → 128 → ReLU → 64 → ReLU → 7 çıkış (Q-değerleri)




Modelin Karar Süreci
Her adımda:
1. Durum vektörü modele verilir.
2. Model, her aksiyon için Q-değeri üretir.
3. En yüksek Q-değerine sahip aksiyon seçilir (veya epsilon ile keşif yapılır).
4. Çevreye bu aksiyon uygulanır, ödül ve yeni durum alınır.
5. Bu deneyim belleğe kaydedilir.
6. Mini-batch seçilerek ağ optimize edilir.
