import numpy as np
import cv2
import math
import os
from collections import deque

# video islemek icin gerekli
VIDEO_DOSYA_YOL = 'tennis.mp4'
KORT_RESIM_YOL = "kort.png"
IS_DEBUG = False

# geometrik olarak dogru araliklarda mı bakalım
UST_ALT_ORAN_ARALIK = (0.3, 0.98)
EN_BOY_ORAN_ARALIK = (1.0, 2.5)
DIKEY_CIZGI_ACI_ARALIK = (60, 120)
YAN_YUKSEKLIK_ORAN_ARALIK = (0.75, 1.25)
GECERSIZ_GORUNUM_ESIK = 5
DURUM_STABILITE_ESIGI = 15

KORT_UST_DOLGU_PIKSEL = 40
KORT_YATAY_DOLGU_PIKSEL = 60
KORT_ALT_DOLGU_PIKSEL = 40
HEDEF_KROKI_GENISLIK, HEDEF_KROKI_YUKSEKLIK = 400, 650
HEDEF_GENISLIK = 800

# oyuncu takip parametreleri
MAKS_OYUNCU_ZIPLAMA_MESAFE = 150
MAKS_OYUNCU_GORUNMEZ_KARE = 35
MAKS_OYUNCU_YUKSEKLIK_UST = 120
MAKS_OYUNCU_YUKSEKLIK_ALT = 280

# top icin degisken parametreler (genislik ve yukseklik olabildigince azalttigimizda daha dogru buluyor ama arada kaciriyor)
TOP_PARAMETRELERI = {
    'MIN_TOP_KONTUR_ALAN': 8,
    'MAKS_TOP_KONTUR_ALAN': 100,
    'TOP_MIN_GENISLIK_YUKSEKLIK': 3,
    'TOP_MAKS_GENISLIK_YUKSEKLIK': 25,
    'TOP_MIN_EN_BOY_ORAN': 0.7,
    'TOP_MAKS_EN_BOY_ORAN': 1.4,
    'TOP_MIN_YOGUNLUK': 0.75,
    'MAKS_TOP_KAYIP_KARE': 10,
    'MAKS_TOP_ZIPLAMA_MESAFE': 80,
    'ARAMA_BOLGE_DOLGU': 50
}

# gorsel degerler
GORSEL_PARAMETRELER = {
    'oyuncu_1_renk': (0, 0, 0),        # ilk oyuncu siyah renk
    'oyuncu_2_renk': (255, 255, 255),  # ikinci oyuncu beyaz renk
    'oyuncu_1_kayip_renk': (80, 80, 80),
    'oyuncu_2_kayip_renk': (160, 160, 160),
    'top_renk': (0, 255, 255),         # sari renk top
    'KROKI_TOP_GECMIS_UZUNLUK': 10,
    'TOP_SOLMA_SURESI_KROKI': 12
}

KENAR_X_KROKI, KENAR_Y_KROKI = 25, 35

KORT_MARGIN_X = 1      # kortun yanlarindaki turuncu alan (kroki cizimi icin)
KORT_MARGIN_Y_UST = -60  # kortun ustundeki turuncu alan (kroki cizimi icin)
KORT_MARGIN_Y_ALT = -50  # kortun altindaki turuncu alan (kroki cizimi icin)

KROKI_KORT_HEDEF_NOKTALAR = np.array([
    [KENAR_X_KROKI + KORT_MARGIN_X, KENAR_Y_KROKI + KORT_MARGIN_Y_UST],
    [HEDEF_KROKI_GENISLIK - KENAR_X_KROKI - KORT_MARGIN_X, KENAR_Y_KROKI + KORT_MARGIN_Y_UST],
    [HEDEF_KROKI_GENISLIK - KENAR_X_KROKI - KORT_MARGIN_X, HEDEF_KROKI_YUKSEKLIK - KENAR_Y_KROKI - KORT_MARGIN_Y_ALT],
    [KENAR_X_KROKI + KORT_MARGIN_X, HEDEF_KROKI_YUKSEKLIK - KENAR_Y_KROKI - KORT_MARGIN_Y_ALT]
], dtype=np.float32)

KROKI_HEDEF_NOKTALAR = np.array([
    [KENAR_X_KROKI, KENAR_Y_KROKI],
    [HEDEF_KROKI_GENISLIK - KENAR_X_KROKI, KENAR_Y_KROKI],
    [HEDEF_KROKI_GENISLIK - KENAR_X_KROKI, HEDEF_KROKI_YUKSEKLIK - KENAR_Y_KROKI],
    [KENAR_X_KROKI, HEDEF_KROKI_YUKSEKLIK - KENAR_Y_KROKI]
], dtype=np.float32)

# Mantiksal donusum icin parametreler
YUKARI_GORUNUM_GENISLIK, YUKARI_GORUNUM_YUKSEKLIK = 400, 800
MANTIKSAL_HEDEF_NOKTALAR = np.array([
    [0, 0],
    [YUKARI_GORUNUM_GENISLIK-1, 0],
    [YUKARI_GORUNUM_GENISLIK-1, YUKARI_GORUNUM_YUKSEKLIK-1],
    [0, YUKARI_GORUNUM_YUKSEKLIK-1]
], dtype=np.float32)
KORT_MERKEZ_Y_YUKARI_GORUNUM = YUKARI_GORUNUM_YUKSEKLIK / 2

class TenisAnalizi:
    # temel olarak kalman filtresi kullanarak oyuncu tahmini yapildi
    def __init__(self, baslangic_kutu):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.01
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
        
        x, y, w, h = baslangic_kutu
        self.kf.statePost = np.array([x + w/2, y + h/2, 0, 0], np.float32).T
        self.kutu = baslangic_kutu
        self.yas = 0
        self.ardisik_gorunmez_sayac = 0
        self.kort_tarafi = None

    # bir sonraki pozisyonu tahminlemeye çalışır bunu yaparken de bir öncekini de kullanıyoruz
    def tahmin_et(self):
        tahmin_durum = self.kf.predict()
        
        self.yas += 1
        self.ardisik_gorunmez_sayac += 1
        
        tahmin_x = int(tahmin_durum[0].item())
        tahmin_y = int(tahmin_durum[1].item())
        w, h = self.kutu[2], self.kutu[3]
        self.kutu = (tahmin_x - w//2, tahmin_y - h//2, w, h)
        return self.kutu

    # tespitten sonraki güncellemeleri yapıyoruz
    def guncelle(self, kutu):
        x, y, w, h = kutu
        olcum = np.array([[x + w/2], [y + h/2]], np.float32)
        self.kf.correct(olcum)
        self.kutu = kutu
        self.ardisik_gorunmez_sayac = 0

### kort tespiti

# hough transform ile çizgileri buldum sonra da roi belirlemeye çalıştım (bazen çizgiler kayıyor ancak daha stabil olmadı)
def roi_bolgesinde_kort_tespit_et(roi_karesi):
    gri = cv2.cvtColor(roi_karesi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gelistirilmis_gri = clahe.apply(gri)
    
    _, cizgi_maskesi = cv2.threshold(gelistirilmis_gri, 210, 255, cv2.THRESH_BINARY)
    h, w = roi_karesi.shape[:2]
    
    min_cizgi_uzunluk = int(w * 0.10)
    maks_cizgi_bosluk = int(w * 0.05)
    hough_esik = 30
    
    cizgiler = cv2.HoughLinesP(cizgi_maskesi, 1, np.pi / 180, threshold=hough_esik, 
                              minLineLength=min_cizgi_uzunluk, maxLineGap=maks_cizgi_bosluk)
    
    if cizgiler is None:
        return None
    
    yatay_cizgiler, dikey_cizgiler = [], []
    
    for cizgi in cizgiler:
        x1, y1, x2, y2 = cizgi[0]
        aci = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
        
        if aci < 45 or aci > 135:
            yatay_cizgiler.append(cizgi)
        else:
            dikey_cizgiler.append(cizgi)
    
    if len(yatay_cizgiler) < 2 or len(dikey_cizgiler) < 2:
        return None
    
    yatay_cizgiler.sort(key=lambda l: (l[0][1] + l[0][3]) / 2)
    dikey_cizgiler.sort(key=lambda l: (l[0][0] + l[0][2]) / 2)
    
    ust_cizgi = yatay_cizgiler[0]
    alt_cizgi = yatay_cizgiler[-1]
    sol_cizgi = dikey_cizgiler[0]
    sag_cizgi = dikey_cizgiler[-1]
    
    def cizgi_parametrelerini_al(cizgi):
        (x1, y1, x2, y2) = cizgi[0]
        if x1 == x2:
            m = float('inf')
            c = x1
        else:
            m = (y2 - y1) / (x2 - x1)
            c = y1 - m * x1
        return m, c
    
    def kesisim_bul(p1, p2):
        m1, c1 = p1
        m2, c2 = p2
        
        if m1 == m2:
            return None
        
        if m1 == float('inf'):
            return (int(c1), int(m2 * c1 + c2))
        if m2 == float('inf'):
            return (int(c2), int(m1 * c2 + c1))
        
        x = (c2 - c1) / (m1 - m2)
        y = m1 * x + c1
        return (int(x), int(y))
    
    # cizgi
    p_ust = cizgi_parametrelerini_al(ust_cizgi)
    p_alt = cizgi_parametrelerini_al(alt_cizgi)
    p_sol = cizgi_parametrelerini_al(sol_cizgi)
    p_sag = cizgi_parametrelerini_al(sag_cizgi)
    
    # kesisim
    sol_ust = kesisim_bul(p_ust, p_sol)
    sag_ust = kesisim_bul(p_ust, p_sag)
    sol_alt = kesisim_bul(p_alt, p_sol)
    sag_alt = kesisim_bul(p_alt, p_sag)
    
    if not all([sol_ust, sag_ust, sol_alt, sag_alt]):
        return None
    
    return np.array([sol_ust, sag_ust, sag_alt, sol_alt], dtype=np.int32)

# geometriyi dogrulamaya calis
def kort_geometrisini_dogrula(koseler, ust_alt_oran_aralik, en_boy_oran_aralik, dikey_cizgi_aci_aralik, yan_yukseklik_oran_aralik):
    if koseler is None or len(koseler) != 4:
        return False

    y_sirali = sorted(koseler, key=lambda p: p[1])
    ust_koseler = sorted(y_sirali[:2], key=lambda p: p[0])
    alt_koseler = sorted(y_sirali[2:], key=lambda p: p[0])
    
    sol_ust, sag_ust = ust_koseler[0], ust_koseler[1]
    sol_alt, sag_alt = (alt_koseler[0], alt_koseler[1]) if alt_koseler[0][0] < alt_koseler[1][0] else (alt_koseler[1], alt_koseler[0])

    ust_genislik = np.linalg.norm(sol_ust - sag_ust)
    alt_genislik = np.linalg.norm(sol_alt - sag_alt)
    sol_yukseklik = np.linalg.norm(sol_ust - sol_alt)
    sag_yukseklik = np.linalg.norm(sag_ust - sag_alt)
    
    if alt_genislik < 1 or sol_yukseklik < 1 or sag_yukseklik < 1:
        return False
    
    ust_alt_orani = ust_genislik / alt_genislik
    if not (ust_alt_oran_aralik[0] < ust_alt_orani < ust_alt_oran_aralik[1]):
        return False

    ortalama_yukseklik = (sol_yukseklik + sag_yukseklik) / 2
    en_boy_orani = alt_genislik / ortalama_yukseklik
    if not (en_boy_oran_aralik[0] < en_boy_orani < en_boy_oran_aralik[1]):
        return False
    
    def aci_hesapla(p1, p2):
        return abs(math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0])))
    
    sol_aci = aci_hesapla(sol_alt, sol_ust)
    sag_aci = aci_hesapla(sag_alt, sag_ust)
    min_aci, maks_aci = dikey_cizgi_aci_aralik
    
    if not (min_aci < sol_aci < maks_aci and min_aci < sag_aci < maks_aci):
        return False

    yukseklik_orani = sol_yukseklik / sag_yukseklik
    min_y_oran, maks_y_oran = yan_yukseklik_oran_aralik
    if not (min_y_oran < yukseklik_orani < maks_y_oran):
        return False
    
    return True

# kortu ikiye boldum ve alttakine oyuncu 2 usttekine oyuncu 1 dedim
def kort_tarafini_belirle(kutu, homografi_matrisi, kort_merkez_y):
    if homografi_matrisi is None:
        return None
    
    x, y, w, h = kutu
    ayak_noktasi = np.array([[[x + w / 2, y + h]]], dtype=np.float32)
    donusturulmus_nokta = cv2.perspectiveTransform(ayak_noktasi, homografi_matrisi)
    
    if donusturulmus_nokta is None or donusturulmus_nokta.size == 0:
        return None
    
    return "ust" if donusturulmus_nokta[0][0][1] < kort_merkez_y else "alt"

### kort tespiti

### oyuncu tespiti 

# bir oyuncu hizli hareket edebildiginden benzer kutuları tek bir kutu yaparak oyuncu tespitini düzlestirmeye calistim
def cakisan_kutulari_birlestir(kutular, yakinlik_esigi=50):
    if len(kutular) == 0:
        return []
    
    birlestirildi = True
    while birlestirildi:
        birlestirildi = False
        i = 0
        while i < len(kutular):
            j = i + 1
            while j < len(kutular):
                kutu1, kutu2 = kutular[i], kutular[j]
                
                uzaklik_x = max(0, max(kutu1[0], kutu2[0]) - min(kutu1[0] + kutu1[2], kutu2[0] + kutu2[2]))
                uzaklik_y = max(0, max(kutu1[1], kutu2[1]) - min(kutu1[1] + kutu1[3], kutu2[1] + kutu2[3]))
                
                if uzaklik_x < yakinlik_esigi and uzaklik_y < yakinlik_esigi:
                    min_x = min(kutu1[0], kutu2[0])
                    min_y = min(kutu1[1], kutu2[1])
                    maks_x = max(kutu1[0] + kutu1[2], kutu2[0] + kutu2[2])
                    maks_y = max(kutu1[1] + kutu1[3], kutu2[1] + kutu2[3])
                    
                    kutular[i] = (min_x, min_y, maks_x - min_x, maks_y - min_y)
                    kutular.pop(j)
                    birlestirildi = True
                    j = i + 1
                else:
                    j += 1
            i += 1
            if birlestirildi:
                break
    
    return kutular

# oyuncularin hareketini tespit ederek onları isaretle
def sadece_hareket_ile_oyuncu_tespit_et(on_plan_maskesi, kort_poligonu, kort_y_sinirlari, oyuncu_boyut_parametreleri, min_alan=300):
    if kort_poligonu is None:
        return []
    
    kort_ust_y, kort_alt_y = kort_y_sinirlari
    maks_y_ust = oyuncu_boyut_parametreleri['maks_yukseklik_ust']
    maks_y_alt = oyuncu_boyut_parametreleri['maks_yukseklik_alt']
    kort_yuksekligi = kort_alt_y - kort_ust_y
    
    if kort_yuksekligi <= 0:
        kort_yuksekligi = 1
    
    cekirdek = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    islenmis_maske = cv2.dilate(on_plan_maskesi, cekirdek, iterations=2)
    islenmis_maske = cv2.erode(islenmis_maske, cekirdek, iterations=1)
    
    konturlar, _ = cv2.findContours(islenmis_maske, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    aday_kutular = []
    for kontur in konturlar:
        (x, y, w, h) = cv2.boundingRect(kontur)
        
        # kort disindakilere bakma ve islem yapma
        if cv2.pointPolygonTest(kort_poligonu, (x + w // 2, y + h // 2), False) < 0:
            continue
        
        if cv2.contourArea(kontur) < min_alan or w > h * 1.5 or h > w * 8:
            continue
        
        alan = cv2.contourArea(kontur)

        if alan < min_alan or w > h * 1.5 or h > w * 8:
            continue

        en_boy_orani = w / float(h) if h > 0 else 0

        if en_boy_orani > 3.0 or en_boy_orani < 0.3:  
            continue

        if h < 15:
            continue

        if w < 8:
            continue

        cevre = cv2.arcLength(kontur, True)
        if cevre > 0:
            kompaktlık = 4 * math.pi * alan / (cevre * cevre)
            if kompaktlık < 0.1:
                continue

        kutu_merkez_y = y + h / 2
        y_orani = (kutu_merkez_y - kort_ust_y) / kort_yuksekligi
        y_orani = np.clip(y_orani, 0, 1)
        izin_verilen_maks_yukseklik = maks_y_ust + y_orani * (maks_y_alt - maks_y_ust)
        
        if h > izin_verilen_maks_yukseklik:
            continue
        
        aday_kutular.append((x, y, w, h))
    
    oyuncular = cakisan_kutulari_birlestir(aday_kutular, yakinlik_esigi=50)
    return [kutu for kutu in oyuncular if (kutu[2] * kutu[3]) > 500]

### oyuncu tespiti 

### top tespiti kodlari
def gecerli_konturlari_al(on_plan_maskesi, min_alan, maks_alan):
    konturlar, _ = cv2.findContours(on_plan_maskesi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [kontur for kontur in konturlar if min_alan < cv2.contourArea(kontur) < maks_alan]

def nokta_kutu_icinde_mi(nokta, kutu):
    x_nokta, y_nokta = nokta
    x_kutu, y_kutu, w_kutu, h_kutu = kutu
    return x_kutu < x_nokta < x_kutu + w_kutu and y_kutu < y_nokta < y_kutu + h_kutu

# topu surekli tespit etmeye calistigimda hata oldugundan soyle bir yontem uyguladım top bir onceki pozisyonundan cok uzaga gidemez bu yuzden yakın pozisyonları kontrol ederek ilerliyoruz
def top_tespit_et(on_plan_maskesi, son_bilinen_top_merkezi, top_kayip_sayaci, kort_sinir_noktalari, oyuncu_kutulari, parametreler):
    if son_bilinen_top_merkezi and top_kayip_sayaci < parametreler['MAKS_TOP_KAYIP_KARE']:
        arama_maskesi = np.zeros_like(on_plan_maskesi)
        sx = int(son_bilinen_top_merkezi[0] - parametreler['ARAMA_BOLGE_DOLGU'])
        sy = int(son_bilinen_top_merkezi[1] - parametreler['ARAMA_BOLGE_DOLGU'])
        sw = int(parametreler['ARAMA_BOLGE_DOLGU'] * 2)
        sh = int(parametreler['ARAMA_BOLGE_DOLGU'] * 2)
        cv2.rectangle(arama_maskesi, (sx, sy), (sx + sw, sy + sh), 255, -1)
        arama_alani = cv2.bitwise_and(on_plan_maskesi, on_plan_maskesi, mask=arama_maskesi)
    else:
        arama_alani = on_plan_maskesi
    
    konturlar = gecerli_konturlari_al(arama_alani, parametreler['MIN_TOP_KONTUR_ALAN'], 
                                     parametreler['MAKS_TOP_KONTUR_ALAN'])
    
    potansiyel_toplar = []
    aktif_oyuncu_kutulari = [t.kutu for t in oyuncu_kutulari if t is not None]
    
    for kontur in konturlar:
        x, y, w, h = cv2.boundingRect(kontur)
        
        if not (parametreler['TOP_MIN_GENISLIK_YUKSEKLIK'] <= w <= parametreler['TOP_MAKS_GENISLIK_YUKSEKLIK'] and 
                parametreler['TOP_MIN_GENISLIK_YUKSEKLIK'] <= h <= parametreler['TOP_MAKS_GENISLIK_YUKSEKLIK']):
            continue
        
        en_boy_orani = w / float(h) if h > 0 else 0
        if not (parametreler['TOP_MIN_EN_BOY_ORAN'] <= en_boy_orani <= parametreler['TOP_MAKS_EN_BOY_ORAN']):
            continue
        
        alan = cv2.contourArea(kontur)
        
        if len(kontur) >= 5:
            yogunluk = float(alan) / cv2.contourArea(cv2.convexHull(kontur))
            if yogunluk < parametreler['TOP_MIN_YOGUNLUK']:
                continue
        
        merkez_aday = (x + w // 2, y + h // 2)
        
        if cv2.pointPolygonTest(kort_sinir_noktalari, merkez_aday, False) < 0:
            continue
        
        if any(nokta_kutu_icinde_mi(merkez_aday, oyuncu_kutusu) for oyuncu_kutusu in aktif_oyuncu_kutulari):
            continue
        
        potansiyel_toplar.append({
            'kutu': (x, y, w, h),
            'merkez': merkez_aday,
            'alan': alan
        })
    
    if not potansiyel_toplar:
        return None
    
    en_iyi_top = None
    
    if son_bilinen_top_merkezi and top_kayip_sayaci < parametreler['MAKS_TOP_KAYIP_KARE']:
        min_mesafe = float('inf')
        for top in potansiyel_toplar:
            mesafe = np.linalg.norm(np.array(son_bilinen_top_merkezi) - np.array(top['merkez']))
            if mesafe < parametreler['MAKS_TOP_ZIPLAMA_MESAFE'] and mesafe < min_mesafe:
                min_mesafe = mesafe
                en_iyi_top = top
    
    if en_iyi_top is None:
        potansiyel_toplar.sort(key=lambda t: t['alan'], reverse=True)
        en_iyi_top = potansiyel_toplar[0]
    
    return (en_iyi_top['kutu'], en_iyi_top['merkez'])

### top tespiti

### kroki

# kroki boyutları farkli oldugundan en yukarida belirledigimiz marginler ile krokiye aktardik
def kroki_icin_noktalari_donustur(nokta_listesi, H_matrisi):
    if not nokta_listesi or H_matrisi is None:
        return []
    
    np_noktalar = np.array([[n[0], n[1]] for n in nokta_listesi], dtype=np.float32)
    if np_noktalar.shape[0] == 0:
        return []
    
    donusturulmus_noktalar = cv2.perspectiveTransform(np.array([np_noktalar]), H_matrisi)
    if donusturulmus_noktalar is None:
        return []
    
    return donusturulmus_noktalar[0]

def kort_krokisini_ciz(temel_kroki_resmi, homografi_matrisi, oyuncu_takipcileri, top_izi_video, mevcut_kare_sayisi, gorsel_parametreler, durum_metni="", durum_rengi=(255,255,255)):

    kroki_gorunumu = temel_kroki_resmi.copy()
    
    if homografi_matrisi is None:
        return kroki_gorunumu
    
    if top_izi_video:
        top_noktalari = [pos for pos, kare_no in top_izi_video]
        donusturulmus_top_izi = kroki_icin_noktalari_donustur(top_noktalari, homografi_matrisi)
        
        for i, (orijinal_pos, kare_no) in enumerate(top_izi_video):
            if i < len(donusturulmus_top_izi):
                donusturulmus_nokta = tuple(map(int, donusturulmus_top_izi[i]))
                yas = mevcut_kare_sayisi - kare_no
                solma_faktoru = max(0.3, 1.0 - (yas / float(gorsel_parametreler['TOP_SOLMA_SURESI_KROKI'])))
                
                solmus_renk = (np.array(gorsel_parametreler['top_renk'], dtype=np.float32) * solma_faktoru).astype(np.uint8)
                solmus_renk_tuple = tuple(solmus_renk.tolist())
                
                # Yaslanan toplar icin kucuk yaricap
                yaricap = 5 if yas < 2 else (4 if yas < gorsel_parametreler['KROKI_TOP_GECMIS_UZUNLUK'] // 2 else 3)
                cv2.circle(kroki_gorunumu, donusturulmus_nokta, yaricap, solmus_renk_tuple, -1)
    
    oyuncu_noktalari, oyuncu_renkleri, oyuncu_yaricaplari = [], [], []
    aktif_takipciler = [t for t in oyuncu_takipcileri.values() if t is not None]
    
    for takipci in aktif_takipciler:
        x, y, w, h = takipci.kutu
        oyuncu_noktalari.append((x + w/2, y + h))
        
        kayip_mi = takipci.ardisik_gorunmez_sayac > 0
        oyuncu_id = 0 if takipci.kort_tarafi == 'ust' else 1
        
        if kayip_mi:
            renk = gorsel_parametreler['oyuncu_1_kayip_renk'] if oyuncu_id == 0 else gorsel_parametreler['oyuncu_2_kayip_renk']
        else:
            renk = gorsel_parametreler['oyuncu_1_renk'] if oyuncu_id == 0 else gorsel_parametreler['oyuncu_2_renk']
        
        oyuncu_renkleri.append(renk)
        oyuncu_yaricaplari.append(6 if kayip_mi else 7)
    
    # burada oyuncuları sinirda tutmaya calistim
    if oyuncu_noktalari:
        donusturulmus_oyuncu_noktalari = kroki_icin_noktalari_donustur(oyuncu_noktalari, homografi_matrisi)
        if not hasattr(kort_krokisini_ciz, 'eski_pozisyonlar'):
            kort_krokisini_ciz.eski_pozisyonlar = {}
        
        for i, donusturulmus_nokta in enumerate(donusturulmus_oyuncu_noktalari):
            oyuncu_key = f"oyuncu_{i}"
            
            if oyuncu_key in kort_krokisini_ciz.eski_pozisyonlar:
                eski_pos = kort_krokisini_ciz.eski_pozisyonlar[oyuncu_key]
                
                aktif_takipciler = [t for t in oyuncu_takipcileri.values() if t is not None]
                oyuncu_taraf = None
                if i < len(aktif_takipciler):
                    oyuncu_taraf = aktif_takipciler[i].kort_tarafi

                if oyuncu_taraf == 'alt':
                    donusturulmus_nokta = (
                        np.clip(donusturulmus_nokta[0], 10, HEDEF_KROKI_GENISLIK - 10),
                        np.clip(donusturulmus_nokta[1], 10, HEDEF_KROKI_YUKSEKLIK - 30)
                    )
                else:
                    donusturulmus_nokta = (
                        np.clip(donusturulmus_nokta[0], 10, HEDEF_KROKI_GENISLIK - 10),
                        np.clip(donusturulmus_nokta[1], 10, HEDEF_KROKI_YUKSEKLIK - 10)
                    )
                
                if oyuncu_key in kort_krokisini_ciz.eski_pozisyonlar:
                    eski_pos = kort_krokisini_ciz.eski_pozisyonlar[oyuncu_key]
               
                mesafe = np.sqrt((eski_pos[0] - donusturulmus_nokta[0])**2 + (eski_pos[1] - donusturulmus_nokta[1])**2)

                if mesafe > 15: 
                    smooth_x = int(eski_pos[0] * 0.3 + donusturulmus_nokta[0] * 0.7)
                    smooth_y = int(eski_pos[1] * 0.3 + donusturulmus_nokta[1] * 0.7)
                elif mesafe > 5:
                    smooth_x = int(eski_pos[0] * 0.5 + donusturulmus_nokta[0] * 0.5)
                    smooth_y = int(eski_pos[1] * 0.5 + donusturulmus_nokta[1] * 0.5)
                else:
                    smooth_x = int(eski_pos[0] * 0.7 + donusturulmus_nokta[0] * 0.3)
                    smooth_y = int(eski_pos[1] * 0.7 + donusturulmus_nokta[1] * 0.3)
                
                nokta_int = (smooth_x, smooth_y)
            else:
                nokta_int = tuple(map(int, donusturulmus_nokta))
            
            kort_krokisini_ciz.eski_pozisyonlar[oyuncu_key] = nokta_int
            
            cv2.circle(kroki_gorunumu, nokta_int, oyuncu_yaricaplari[i], oyuncu_renkleri[i], -1)
    
    if durum_metni:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2

        (text_width, text_height), _ = cv2.getTextSize(durum_metni, font, font_scale, thickness)

        x = HEDEF_KROKI_GENISLIK - text_width - 10
        y = 30
        
        cv2.putText(kroki_gorunumu, durum_metni, (x, y), font, font_scale, durum_rengi, thickness)
        
    return kroki_gorunumu

### kroki 

def temel_ekranlari_goster(ana_kare, hareket_maskesi, kroki_gorunumu, hata_ayiklama_modu, aktif_mi):
    cv2.imshow("Tenis Mac Analizi", ana_kare)
    
    if hata_ayiklama_modu and aktif_mi and hareket_maskesi is not None:
        hareket_maskesi_renkli = cv2.cvtColor(hareket_maskesi, cv2.COLOR_GRAY2BGR)
        hareket_maskesi_renkli = cv2.applyColorMap(hareket_maskesi, cv2.COLORMAP_JET)
        cv2.imshow("Hareket Maskesi", hareket_maskesi_renkli)
    elif hata_ayiklama_modu:
        h, w = 480, 640
        bos_maske = np.zeros((h, w, 3), dtype=np.uint8)
        durum_metni = "GAME" if aktif_mi else "REPLAY"
        cv2.putText(bos_maske, durum_metni, (w//2 - 100, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("Hareket Maskesi", bos_maske)
    
    if kroki_gorunumu is not None:
        cv2.imshow("Kort", kroki_gorunumu)
    else:
        bos_kroki = np.zeros((HEDEF_KROKI_YUKSEKLIK, HEDEF_KROKI_GENISLIK, 3), dtype=np.uint8)
        cv2.putText(bos_kroki, "Kroki Yok", (HEDEF_KROKI_GENISLIK//2 - 50, HEDEF_KROKI_YUKSEKLIK//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Kort", bos_kroki)

def main(hata_ayiklama=False):
    if not os.path.exists(VIDEO_DOSYA_YOL):
        print(f"UYARI: Video dosyasi okunamadi. (Lokasyon yanlis olabilir.)")
        return
    
    kamera = cv2.VideoCapture(VIDEO_DOSYA_YOL)
    ret, test_karesi = kamera.read()
    if not ret:
        print("UYARI: Video dosyasi okunamadi. (Lokasyon yanlis olabilir.)")
        return
    
    video_fps = kamera.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0: video_fps = 30
    frame_delay = int(1000 / video_fps)
    hiz_carpani = 1.0
    
    olcek = HEDEF_GENISLIK / test_karesi.shape[1]
    hedef_yukseklik = int(test_karesi.shape[0] * olcek)
    kamera.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    arka_plan_cikarici = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=16, detectShadows=False)
    
    temel_kroki_yeniden_boyutlandirilmis = None
    if os.path.exists(KORT_RESIM_YOL):
        kroki_resmi = cv2.imread(KORT_RESIM_YOL)
        temel_kroki_yeniden_boyutlandirilmis = cv2.resize(kroki_resmi, (HEDEF_KROKI_GENISLIK, HEDEF_KROKI_YUKSEKLIK))
    else:
        print(f"UYARI: Kroki resmi su yolda: '{KORT_RESIM_YOL}' , bulunamadi.")
    
    son_bilinen_koseler = None
    kort_gorunumu_aktif_mi = True
    gecersiz_gorunum_sayaci = 0
    kare_sayaci = 0
    
    oyuncu_takipcileri = {'ust': None, 'alt': None}
    homografi_matrisi = None
    homografi_matrisi_kroki = None
    
    son_bilinen_top_merkezi = None
    top_kayip_sayaci = 0
    top_izi_video = deque(maxlen=GORSEL_PARAMETRELER['KROKI_TOP_GECMIS_UZUNLUK'])
    
    durum_metni = ""
    durum_rengi = (0, 255, 255)
    
    durum_degisim_sayaci = 0
    onceki_durum = "GAME"
    
    print("Islem basliyor... Cikmak icin 'q' tusuna basin.")
    
    while kamera.isOpened():
        ret, kare = kamera.read()
        if not ret:
            break
        
        mevcut_delay = int(frame_delay / hiz_carpani)
        
        kare = cv2.resize(kare, (HEDEF_GENISLIK, hedef_yukseklik))
        kare_sayaci += 1
        
        h, w, _ = kare.shape
        roi_kontrol_icin = kare[int(h*0.2):, :]
        mevcut_koseler_goreceli = roi_bolgesinde_kort_tespit_et(roi_kontrol_icin)
        
        geometri_tamam_mi = False
        if mevcut_koseler_goreceli is not None:
            mevcut_koseler_mutlak = mevcut_koseler_goreceli + [0, int(h*0.2)]
            if kort_geometrisini_dogrula(mevcut_koseler_mutlak, UST_ALT_ORAN_ARALIK, EN_BOY_ORAN_ARALIK, DIKEY_CIZGI_ACI_ARALIK, YAN_YUKSEKLIK_ORAN_ARALIK):
                geometri_tamam_mi = True
        
        if geometri_tamam_mi:
            gecersiz_gorunum_sayaci = 0
            kort_gorunumu_aktif_mi = True
            son_bilinen_koseler = mevcut_koseler_mutlak
            
            sirali_koseler = sorted(son_bilinen_koseler, key=lambda p: p[1])
            ust_koseler = sorted(sirali_koseler[:2], key=lambda p: p[0])
            alt_koseler = sorted(sirali_koseler[2:], key=lambda p: p[0])
            
            kaynak_noktalar = np.array([ust_koseler[0], ust_koseler[1], alt_koseler[1], alt_koseler[0]], dtype=np.float32)
            homografi_matrisi, _ = cv2.findHomography(kaynak_noktalar, MANTIKSAL_HEDEF_NOKTALAR)
            homografi_matrisi_kroki, _ = cv2.findHomography(kaynak_noktalar, KROKI_KORT_HEDEF_NOKTALAR)
        else:
            gecersiz_gorunum_sayaci += 1
        
        if gecersiz_gorunum_sayaci > GECERSIZ_GORUNUM_ESIK:
            kort_gorunumu_aktif_mi = False

        gecici_durum = "GAME" if kort_gorunumu_aktif_mi else "REPLAY"
        
        if gecici_durum == onceki_durum:
            durum_degisim_sayaci = 0
        else:
            durum_degisim_sayaci += 1
            if durum_degisim_sayaci >= DURUM_STABILITE_ESIGI:
                onceki_durum = gecici_durum
                durum_degisim_sayaci = 0
                print(f"Durum degisti: {onceki_durum}")
        
        if onceki_durum == "GAME":
            durum_metni = "GAME"
            durum_rengi = (0, 255, 0)
        else:
            durum_metni = "REPLAY"
            durum_rengi = (0, 255, 255)
        
        tespit_edilen_top_bilgisi = None
        on_plan_maskesi_sadece_kort = None
        
        if kort_gorunumu_aktif_mi and son_bilinen_koseler is not None and homografi_matrisi is not None:
            dolgulu_koseler = np.copy(son_bilinen_koseler)
            y_indisleri = np.argsort(dolgulu_koseler[:, 1])
            x_indisleri = np.argsort(dolgulu_koseler[:, 0])
            
            dolgulu_koseler[y_indisleri[:2], 1] -= KORT_UST_DOLGU_PIKSEL
            dolgulu_koseler[y_indisleri[2:], 1] += KORT_ALT_DOLGU_PIKSEL
            dolgulu_koseler[x_indisleri[:2], 0] -= KORT_YATAY_DOLGU_PIKSEL
            dolgulu_koseler[x_indisleri[2:], 0] += KORT_YATAY_DOLGU_PIKSEL
            
            kort_poligonu = cv2.convexHull(dolgulu_koseler)
            
            oyun_alani_maskesi = np.zeros(kare.shape[:2], dtype=np.uint8)
            cv2.drawContours(oyun_alani_maskesi, [kort_poligonu], -1, 255, -1)
            
            bulaniklik_karesi = cv2.GaussianBlur(kare, (9, 9), 0)
            on_plan_maskesi_ham = arka_plan_cikarici.apply(bulaniklik_karesi, learningRate=0.005)
            on_plan_maskesi_sadece_kort = cv2.bitwise_and(on_plan_maskesi_ham, on_plan_maskesi_ham, mask=oyun_alani_maskesi)
            
            for taraf in oyuncu_takipcileri:
                if oyuncu_takipcileri[taraf]:
                    oyuncu_takipcileri[taraf].tahmin_et()

            kort_y_sinirlari = (np.min(son_bilinen_koseler[:, 1]), np.max(son_bilinen_koseler[:, 1]))
            oyuncu_boyut_parametreleri = {
                'maks_yukseklik_ust': MAKS_OYUNCU_YUKSEKLIK_UST,
                'maks_yukseklik_alt': MAKS_OYUNCU_YUKSEKLIK_ALT
            }
            
            oyuncu_tespitleri = sadece_hareket_ile_oyuncu_tespit_et(on_plan_maskesi_sadece_kort, kort_poligonu, kort_y_sinirlari, oyuncu_boyut_parametreleri)
            
            ust_tespitler, alt_tespitler = [], []
            for i, tespit in enumerate(oyuncu_tespitleri):
                taraf = kort_tarafini_belirle(tespit, homografi_matrisi, KORT_MERKEZ_Y_YUKARI_GORUNUM)
                if taraf == 'ust':
                    ust_tespitler.append({'kutu': tespit, 'id': i})
                elif taraf == 'alt':
                    alt_tespitler.append({'kutu': tespit, 'id': i})
            
            kullanilan_tespit_idleri = set()
            for taraf, takipci in list(oyuncu_takipcileri.items()):
                if takipci:
                    x, y, w, h = takipci.kutu
                    merkez_y = y + h / 2
                    
                    ayak_noktasi = np.array([[[x + w / 2, y + h]]], dtype=np.float32)
                    donusturulmus_nokta = cv2.perspectiveTransform(ayak_noktasi, homografi_matrisi)
                    
                    if donusturulmus_nokta is not None and donusturulmus_nokta.size > 0:
                        gercek_taraf = "ust" if donusturulmus_nokta[0][0][1] < KORT_MERKEZ_Y_YUKARI_GORUNUM else "alt"

                        if gercek_taraf != taraf:
                            takipci.ardisik_gorunmez_sayac += 2  
                            
            for taraf, takipci in oyuncu_takipcileri.items():
                if takipci:
                    taraftaki_tespitler = ust_tespitler if taraf == 'ust' else alt_tespitler
                    en_iyi_eslesme = None
                    min_mesafe = MAKS_OYUNCU_ZIPLAMA_MESAFE
                    
                    for tespit_bilgisi in taraftaki_tespitler:
                        mesafe = np.linalg.norm(np.array(takipci.kutu[:2]) - np.array(tespit_bilgisi['kutu'][:2]))
                        if mesafe < min_mesafe:
                            min_mesafe = mesafe
                            en_iyi_eslesme = tespit_bilgisi
                    
                    if en_iyi_eslesme:
                        takipci.guncelle(en_iyi_eslesme['kutu'])
                        kullanilan_tespit_idleri.add(en_iyi_eslesme['id'])
            
            for taraf in ['ust', 'alt']:
                if oyuncu_takipcileri[taraf] is None:
                    taraftaki_tespitler = ust_tespitler if taraf == 'ust' else alt_tespitler
                    musait_tespitler = [t for t in taraftaki_tespitler if t['id'] not in kullanilan_tespit_idleri]
                    
                    if musait_tespitler:
                        if taraf == 'ust':
                            en_iyi_yeni_tespit = min(musait_tespitler, key=lambda t: t['kutu'][1])
                        else:
                            en_iyi_yeni_tespit = max(musait_tespitler, key=lambda t: t['kutu'][1] + t['kutu'][3])
                        
                        yeni_takipci = TenisAnalizi(en_iyi_yeni_tespit['kutu'])
                        yeni_takipci.kort_tarafi = taraf
                        oyuncu_takipcileri[taraf] = yeni_takipci
                        kullanilan_tespit_idleri.add(en_iyi_yeni_tespit['id'])
            
            for taraf, takipci in oyuncu_takipcileri.items():
                if takipci and takipci.ardisik_gorunmez_sayac > MAKS_OYUNCU_GORUNMEZ_KARE:
                    oyuncu_takipcileri[taraf] = None
            
            aktif_oyuncu_kutulari = [t for t in oyuncu_takipcileri.values() if t is not None]
            tespit_edilen_top_bilgisi = top_tespit_et(on_plan_maskesi_sadece_kort, son_bilinen_top_merkezi, top_kayip_sayaci, cv2.convexHull(son_bilinen_koseler), aktif_oyuncu_kutulari, TOP_PARAMETRELERI)
            
            if tespit_edilen_top_bilgisi:
                son_bilinen_top_merkezi = tespit_edilen_top_bilgisi[1]
                top_kayip_sayaci = 0
                top_izi_video.append((son_bilinen_top_merkezi, kare_sayaci))
            else:
                top_kayip_sayaci += 1
                if top_kayip_sayaci > TOP_PARAMETRELERI['MAKS_TOP_KAYIP_KARE']:
                    son_bilinen_top_merkezi = None
                    top_izi_video.clear()
        else:
            top_kayip_sayaci += 1
            if top_kayip_sayaci > TOP_PARAMETRELERI['MAKS_TOP_KAYIP_KARE']:
                son_bilinen_top_merkezi = None
                top_izi_video.clear()
        
        gosterim_karesi = kare.copy()
        kroki_gorunumu = None
        
        if temel_kroki_yeniden_boyutlandirilmis is not None:
            kroki_gorunumu = kort_krokisini_ciz(temel_kroki_yeniden_boyutlandirilmis, homografi_matrisi_kroki, oyuncu_takipcileri if kort_gorunumu_aktif_mi else {}, top_izi_video, kare_sayaci, GORSEL_PARAMETRELER, durum_metni, durum_rengi)
        
        if kort_gorunumu_aktif_mi and son_bilinen_koseler is not None:
            dolgulu_koseler_gorsel = np.copy(son_bilinen_koseler)
            y_indisleri_gorsel = np.argsort(dolgulu_koseler_gorsel[:, 1])
            x_indisleri_gorsel = np.argsort(dolgulu_koseler_gorsel[:, 0])
            
            dolgulu_koseler_gorsel[y_indisleri_gorsel[:2], 1] -= KORT_UST_DOLGU_PIKSEL
            dolgulu_koseler_gorsel[y_indisleri_gorsel[2:], 1] += KORT_ALT_DOLGU_PIKSEL
            dolgulu_koseler_gorsel[x_indisleri_gorsel[:2], 0] -= KORT_YATAY_DOLGU_PIKSEL
            dolgulu_koseler_gorsel[x_indisleri_gorsel[2:], 0] += KORT_YATAY_DOLGU_PIKSEL
            
            gorsel_maske = np.zeros(kare.shape[:2], dtype=np.uint8)
            cv2.drawContours(gorsel_maske, [cv2.convexHull(dolgulu_koseler_gorsel)], -1, 255, -1)
            
            bulanik_kare = cv2.GaussianBlur(gosterim_karesi, (15, 15), 0)
            
            for i in range(3): 
                gosterim_karesi[:,:,i] = np.where(gorsel_maske == 255, gosterim_karesi[:,:,i], bulanik_kare[:,:,i]) 
        
        cv2.putText(gosterim_karesi, durum_metni, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
        cv2.putText(gosterim_karesi, durum_metni, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, durum_rengi, 2)
        
        if kort_gorunumu_aktif_mi:
            aktif_takipciler = sorted([t for t in oyuncu_takipcileri.values() if t], 
                                    key=lambda t: 0 if t.kort_tarafi == 'ust' else 1)
            
            for takipci in aktif_takipciler:
                x, y, w_kutu, h_kutu = map(int, takipci.kutu)
                oyuncu_id = 0 if takipci.kort_tarafi == 'ust' else 1
                
                renk = GORSEL_PARAMETRELER['oyuncu_1_renk'] if oyuncu_id == 0 else GORSEL_PARAMETRELER['oyuncu_2_renk']
                etiket = "Oyuncu 1" if takipci.kort_tarafi == 'ust' else "Oyuncu 2"
                
                cv2.rectangle(gosterim_karesi, (x, y), (x + w_kutu, y + h_kutu), renk, 2)
                cv2.putText(gosterim_karesi, etiket, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, renk, 2)
            
            if top_izi_video:
                for i, (top_pozisyon, kare_no) in enumerate(top_izi_video):
                    yas = kare_sayaci - kare_no
                    solma_faktoru = max(0.2, 1.0 - (yas / float(GORSEL_PARAMETRELER['TOP_SOLMA_SURESI_KROKI'] * 0.7)))
                    
                    top_izi_renk = (np.array(GORSEL_PARAMETRELER['top_renk'], dtype=np.float32) * solma_faktoru).astype(np.uint8)
                    top_izi_renk_tuple = tuple(top_izi_renk.tolist())
                    
                    yaricap = 6 if yas < 2 else (4 if yas < 4 else 2)
                    cv2.circle(gosterim_karesi, top_pozisyon, yaricap, top_izi_renk_tuple, -1)

            if tespit_edilen_top_bilgisi:
                _, top_merkezi = tespit_edilen_top_bilgisi
                cv2.circle(gosterim_karesi, top_merkezi, 8, GORSEL_PARAMETRELER['top_renk'], -1)
                cv2.circle(gosterim_karesi, top_merkezi, 12, GORSEL_PARAMETRELER['top_renk'], 2)
        
        temel_ekranlari_goster(gosterim_karesi, on_plan_maskesi_sadece_kort, kroki_gorunumu, hata_ayiklama, kort_gorunumu_aktif_mi)
        
        tus = cv2.waitKey(mevcut_delay) & 0xFF
        if tus == ord('q'): break
        elif tus == ord('p'): cv2.waitKey(0)
        elif tus == ord('+'): hiz_carpani = min(4.0, hiz_carpani + 0.5)
        elif tus == ord('-'): hiz_carpani = max(0.25, hiz_carpani - 0.5)
        elif tus == ord('r'): hiz_carpani = 1.0
    
    kamera.release()
    cv2.destroyAllWindows()
    print("Islemler tamamlandi.")

if __name__ == "__main__":
    main(IS_DEBUG)