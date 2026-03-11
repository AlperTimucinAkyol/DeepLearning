import random
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_turev(x):
    return x * (1 - x)

veriler = [
    [[0, 0], [0]],
    [[0, 1], [1]],
    [[1, 0], [1]],
    [[1, 1], [0]]
]

giris_noru = 2
gizli_noru = 3 
cikis_noru = 1
ogrenme_hizi = 0.5

w_giris_gizli = [[random.uniform(-1, 1) for _ in range(gizli_noru)] for _ in range(giris_noru)]
b_gizli = [random.uniform(-1, 1) for _ in range(gizli_noru)]

w_gizli_cikis = [[random.uniform(-1, 1) for _ in range(cikis_noru)] for _ in range(gizli_noru)]
b_cikis = [random.uniform(-1, 1) for _ in range(cikis_noru)]

print("Eğitim Başlıyor...")
for epoch in range(10000):
    toplam_hata = 0
    
    for giris, hedef in veriler:
        gizli_cikislar = []
        for j in range(gizli_noru):
            toplam = b_gizli[j]
            for i in range(giris_noru):
                toplam += giris[i] * w_giris_gizli[i][j]
            gizli_cikislar.append(sigmoid(toplam))
        cikis_degerleri = []
        for k in range(cikis_noru):
            toplam = b_cikis[k]
            for j in range(gizli_noru):
                toplam += gizli_cikislar[j] * w_gizli_cikis[j][k]
            cikis_degerleri.append(sigmoid(toplam))
        
        cikis = cikis_degerleri[0]
        hedef_deger = hedef[0]
        hata = hedef_deger - cikis
        toplam_hata += abs(hata)
        grad_cikis = hata * sigmoid_turev(cikis)
        b_cikis[0] += ogrenme_hizi * grad_cikis
        
        for j in range(gizli_noru):
            w_gizli_cikis[j][0] += ogrenme_hizi * grad_cikis * gizli_cikislar[j]
            
            hata_gizli = grad_cikis * w_gizli_cikis[j][0]
            grad_gizli = hata_gizli * sigmoid_turev(gizli_cikislar[j])
            b_gizli[j] += ogrenme_hizi * grad_gizli
            
            for i in range(giris_noru):
                w_giris_gizli[i][j] += ogrenme_hizi * grad_gizli * giris[i]

    if (epoch + 1) % 2000 == 0:
        print(f"Epoch {epoch+1}, Ortalama Hata: {toplam_hata/4:.4f}")

print("\n--- SONUÇLAR ---")
print("Giriş\t| Beklenen\t| Tahmin")
print("-" * 35)
dogru_sayisi = 0
 
for giris, hedef in veriler:
    gizli_cikislar = []
    for j in range(gizli_noru):
        toplam = b_gizli[j]
        for i in range(giris_noru):
            toplam += giris[i] * w_giris_gizli[i][j]
        gizli_cikislar.append(sigmoid(toplam))
    
    cikis_toplam = b_cikis[0]
    for j in range(gizli_noru):
        cikis_toplam += gizli_cikislar[j] * w_gizli_cikis[j][0]
    
    sonuc = sigmoid(cikis_toplam)
    tahmin = 1 if sonuc > 0.5 else 0
    
    if tahmin == hedef[0]:
        dogru_sayisi += 1
        
    print(f"{giris}\t|     {hedef[0]}\t\t|   {tahmin} ({sonuc:.2f})")

print(f"\nBasari Orani: {dogru_sayisi}/4")