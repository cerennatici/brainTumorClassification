############################### CNN İLE BEYİN TUMOR TESPİTİ ###############################
"""
!pip install split-folders
!pip install torch-summary

split-folders → Veriyi eğitim/test/doğrulama kümelerine bölmek için
torch-summary → PyTorch modellerinin detaylarını özetlemek için, katmanlar ve parametrelerin kullanmı için
"""
#bu iki kütüphaneyi yükledik

# Gerekli kütüphaneleri içe aktarın
import pandas as pd  # Veri işleme ve analiz için kullanılan kütüphane
import seaborn as sns
sns.set(style='darkgrid')  # Gelişmiş veri görselleştirme kütüphanesi (karanlık ızgara temasıyla)
import pathlib  # Dosya ve dizin yollarıyla çalışmak için kullanılan kütüphane
import os
import matplotlib.pyplot as plt
from PIL import Image

####################### VERİ YUKLEME ###################################

labels_df = pd.read_csv("!!!!...Enter csv file path...!!!!!")   #pip install tabulate  kütüphanesini indirdik
print("----------------------- VERİ SETİ -----------------------")
print(labels_df.head().to_markdown())  #verinin özelliklerini tablo şekilnde getirir.
print(f"\nSatır ve Sütun Sayısı: {labels_df.shape}") #kaç satır ve kaç sütun adından oluştuğunu getirir

##################### veri setini ayırmamız gerekiyor ama ben ilk önce hem veri setini arttırmak hem de çeşitlilik sağlamak için her görüntünün horizontal ve 10 derece döndürülmüş halini de ekleyeceğim



# Veri seti klasörleri
input_folder = "!!!!...Enter input dataset path...!!!!!"
output_folder = "!!!!...Enter new output dataset path...!!!!!"

# Sağlıklı ve tümörlü klasörlerini al
categories = ["Brain Tumor", "Healthy"]

# Görselleri döndürme fonksiyonu
def transform_rotate(image):
    angle = 10  # 10 derece döndürme
    return image.rotate(angle)

# Veri artırma işlemi
for category in categories:
    category_path = os.path.join(input_folder, category)  # Kategoriye özel yol
    save_path = os.path.join(output_folder, category)  # Yeni resimleri kaydetme klasörü

    # Kategorinin çıktısını oluştur
    os.makedirs(save_path, exist_ok=True) #yoldaki klasörleri oluşturur dosyalar zaten varsa hata vermez.

    for img_name in os.listdir(category_path):  #brain tumor, healty deki tüm resimleri gezeceğiz.
        img_path = os.path.join(category_path, img_name)

        # Görseli aç
        img = Image.open(img_path)

        # Orijinal görseli de kaydet

        img = img.convert('RGB')  # farklı türdeki resimleri RGB'ye dönüştür
        img.save(os.path.join(save_path, f"original_{img_name}"))

        # Görseli yatay olarak döndürme (flip)
        flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # Döndürülen görseli kaydetme
        flipped_img.save(os.path.join(save_path, f"flip_{img_name}"))

        # Görseli döndürme (10 derece)
        rotated_img = transform_rotate(img)

        # Döndürülen görseli kaydetme
        rotated_img.save(os.path.join(save_path, f"rotate_{img_name}"))

print("Veri artırma tamamlandı! Yeni resimler kaydedildi.")

# DataFrame için boş listeler
metadata = []
metadata_rgb = []

# Görsellerin bilgilerini toplama
for category in categories:
    category_path = os.path.join(output_folder, category)  # Kategoriye özel yol

    for img_name in os.listdir(category_path):
        img_path = os.path.join(category_path, img_name)

        # Görseli aç
        img = Image.open(img_path)

        # Görselin formatı, modu ve şekli
        img_format = img.format
        img_mode = img.mode
        img_shape = img.size + (len(img.getbands()),)  # width, height, num_channels

        # Resmin bilgilerini dataframe için hazırlama
        row = {
            'image': img_name,  # --> ismi
            'class': category,   #--> tumor, healthy
            'format': img_format, #--> dosya uzantısı
            'mode': img_mode, #--> RGB falan
            'shape': img_shape # -->boyut
        }

        # Tüm görsellerin metadata'sını ekleme
        metadata.append(row)

        # Sadece RGB olanları ayırma
        if img_mode == 'RGB':
            metadata_rgb.append(row)

# DataFrame oluşturma
metadata_df = pd.DataFrame(metadata)
metadata_rgb_df = pd.DataFrame(metadata_rgb)

# Excel dosyasını kaydetme
metadata_file = "...new output folder path.../metadata.xlsx"
metadataonlyrgb_file = "...new output folder path.../metadataonlyrgb.xlsx"

# Excel dosyasına yazma
metadata_df.to_excel(metadata_file, index=False)  #false olduğu için satır numaralarını excele yazmaz.
metadata_rgb_df.to_excel(metadataonlyrgb_file, index=False)

print(f"Metadata dosyaları oluşturuldu: \n{metadata_file} \n{metadataonlyrgb_file}")

############################################################################ veri setindeki tüm resimleri yatayda ve 10 derece döndürerek sağlıklı ve tümorlü veri sayısını *2 kat arttırdım.
#10 derece döndürmemin sebebi bunlar sağlık verisi olduğu için hem çeşitlendirmeyi sağlamak hem de çok fazla döndürmediğim için görselin ana yapısını da bozup öğrenmeyi zorlaştırmamak
#excele de yeni verileri yazdırdım.
## veri setini çevirdikten sonra tekrardan güncel verilerimizi okuyalım


####################### YENİ VERİ YUKLEME ###################################

labels_df = pd.read_excel("...new output folder path.../metadata.xlsx")
print("\n----------------------- YENİ VERİ SETİ -----------------------")
print(labels_df.head().to_markdown())
print(f"\nSatır ve Sütun Sayısı: {labels_df.shape}") #kaç satır ve kaç sütun adından oluştuğunu getirir.

tumor_path = "...new output folder path.../Brain Tumor"
healthy_path = "...new output folder path.../Healthy"

# Klasördeki dosya sayısını hesapla
num_tumor = len(os.listdir(tumor_path))
num_healthy = len(os.listdir(healthy_path))

# Verileri hazırla
categories = ["Tumor", "Healthy"]
counts = [num_tumor, num_healthy]


plt.figure(figsize=(6,5)) #Histogram için ekran boyutu
plt.bar(categories, counts, color=['blue', 'purple']) #kategori renkleri
plt.xlabel("Sınıf")# X eksenine etiket ekle
plt.ylabel("Veri Sayısı")# Y eksenine etiket ekle
plt.title("Tümörlü ve Sağlıklı Beyin MR Görüntüsü Veri Dağılımı")

# Y eksenine grid çizgileri ekle, sadece yatay eksende ve çizgilerin stilini belirle
plt.grid(axis='y', linestyle='--', alpha=0.5, color='black')   #0.5 saydamlığı belirler

# Her bir barın üstüne veri sayısını ekle
for i, count in enumerate(counts):  # 'counts' listesinde her bir veri için
    plt.text(i, count + 10, str(count), ha='center', fontsize=12)  # 'count' sayısını, barın biraz üstüne ekler    x,y,yazı,ortala,font
plt.show()

#####################################################################################################################
#burada veri setimizi stratify olacak şekilde train test ve validation setlerine bölüyoruz artık modelimizi bu veriler üzerinden işleyeceğiz.
#Veriler orantılı bir şekilde geliyor sıkıntı yok bu kısım da sadece eğitim test val. verilerini oluşturmak için önemliydi artık kullanmayacağız yorum satırına alabiliriz.

import pathlib
import splitfolders
import os

# Dataset Path
data_dir = "C:/Users/Ceren/Desktop/beyinVeri-KopyaCevrilmis"
data_dir = pathlib.Path(data_dir)

# Ratio fonksiyonu ile birlikte veri seti dengeli (stratify) bölünmüştür.
splitfolders.ratio(data_dir, output="C:/Users/Ceren/Desktop/beyinVeri-KopyaCevrilmis/brain",
                   seed=42, ratio=(0.7, 0.15, 0.15), group_prefix=None)  #none demeseydik dosya isimlerine göre gruplandırılacaklardı şu anda rastgele oluyor.

# Yeni yol
data_dir = pathlib.Path("C:/Users/Ceren/Desktop/beyinVeri-KopyaCevrilmis/brain")

# Veri sayısı hesaplama --> walk fonksiyonu recursive olduğu için dosyalar arasında derinlemesine ilerler.
def count_images(directory):
    return sum([len(files) for _, _, files in os.walk(directory)])   #-->	O anda gezilen klasörün tam yolu,Bu klasör içindeki alt klasörlerin isim listesi,Bu klasör içindeki dosya isimlerinin listesi

train_count = count_images(data_dir / "train")
val_count = count_images(data_dir / "val")
test_count = count_images(data_dir / "test")
total_count = train_count + val_count + test_count

# Print counts
print(f"Train Set: {train_count} images")
print(f"Validation Set: {val_count} images")
print(f"Test Set: {test_count} images")
print(f"Total Images: {total_count}")

############################## TRAİN TEST VALİDATİON VERİLERİMİZİ GÖRSELLEŞTİRME ############################
# Dosya yolları
path = "C:/Bilgisayarim/bilgisayarMuhendisligi/3.sınıfBaharKod/yapayZeka/beyinVeri-KopyaCevrilmis/brain"
paths = {
    "train": os.path.join(path, "train"),
    "val": os.path.join(path, "val"),
    "test": os.path.join(path, "test")
}

# Dosya sayısını almak için fonksiyon
def count_files(path):
    tumor_count = len(os.listdir(os.path.join(path, 'Brain Tumor')))
    healthy_count = len(os.listdir(os.path.join(path, 'Healthy')))
    return tumor_count, healthy_count

# Histogram çizim fonksiyonu
def plot_histogram(categories, counts, title):
    plt.figure(figsize=(6, 5))  # Histogram için ekran boyutu
    plt.bar(categories, counts, color=['green', 'orange'])  # Kategori renkleri
    plt.xlabel("Sınıf")  # X eksenine etiket ekle
    plt.ylabel("Veri Sayısı")  # Y eksenine etiket ekle
    plt.title(title)
    plt.grid(axis='y', linestyle='--', alpha=0.5, color='black')
    for i, count in enumerate(counts):
        plt.text(i, count + 10, str(count), ha='center', fontsize=12)  # Veri sayısını barın üstüne ekle
    plt.show()

# Verileri hazırla
categories = ["Tumor", "Healthy"]

# Train, Validation ve Test setleri için dosya sayısını hesapla
train_counts = count_files(paths["train"])
val_counts = count_files(paths["val"])
test_counts = count_files(paths["test"])

# Histogramları çiz
plot_histogram(categories, train_counts, "Train Set: Tümörlü ve Sağlıklı Beyin MR Görüntüsü Veri Dağılımı")
plot_histogram(categories, val_counts, "Validation Set: Tümörlü ve Sağlıklı Beyin MR Görüntüsü Veri Dağılımı")
plot_histogram(categories, test_counts, "Test Set: Tümörlü ve Sağlıklı Beyin MR Görüntüsü Veri Dağılımı")



#################### VERİ ÇEŞİTLENDİRME UYGULAYACAĞIZ (ilk başta biz veriyi çevirmiştik ama burada resimlerin de aynı boyutta olmalarını sağlayacağız  ###################

import torch  #Tensor işlemleri, model oluşturma
import torchvision  #PyTorcha bağlı görüntü işleme kütüphanesi
from torchvision import datasets,transforms  #Görüntülerin modele girmeden önce dönüştürlmesini sağlar.
import numpy as np   #tensorleri numpy arraylerine çevirir.

# Veri seti ve transform ayarlarını tanımlıyoruz
data_dir = pathlib.Path(path)

# Görselleri yükleyerek her bir kanal için mean ve std değerlerini hesaplıyoruz
def calculate_mean_std(dataset):
    mean = torch.zeros(3)   #RGB
    std = torch.zeros(3)
    total_images = len(dataset)

    for img, _ in dataset:  #resim ve etiktei
        img = transforms.ToTensor()(img)  # Görseli tensöre dönüştür, 0-255 arasında yer alan görsel değerlerini 0-1 aralığına getirir.
        mean += img.mean(dim=[1, 2])  # (C, H, W) formatında: her kanalın ortalamasını al  örneğin 5,4 3 kanallı bir resim 3 kanal için de tüm pixeldeki değerlerin ort ve std si hesaplanır
        # sonuç  tensor([R_mean, G_mean, B_mean])
        std += img.std(dim=[1, 2])  # Her kanalın standart sapmasını al
        # sonuç  tensor([R_std, G_std, B_std])

    mean /= total_images  # Ortalama değeri tüm görseller üzerinde alıyoruz
    std /= total_images  # Standart sapmayı tüm görseller üzerinde alıyoruz

    return mean, std


train_set = torchvision.datasets.ImageFolder(data_dir.joinpath("train"))   #resimlerin bulunduğu klasörü etiket olarak algılar etiket ve resim alır.
mean, std = calculate_mean_std(train_set)

print("Eğitim Seti için Mean: ", mean)
print("Eğitim Seti için Std: ", std)


#İLK ÖNCE EĞİTİM VE VALİDATİON İÇİN transfor işlemi
transformTV = transforms.Compose(   #compose işlemi işlemlerin sırayla yapılmasını sağlar
    [
        transforms.Resize((256,256)),
        transforms.RandomVerticalFlip(p=0.25), #yukarıdan aşağıya çevirme  oranı küçük tutacağım
        transforms.RandomApply([transforms.RandomRotation(20)], p=0.25), #resmi çevirme biz 10 derece çevirmiştik her resmi %25 ini de 20 derece çevirsin
        transforms.ToTensor(),  #görüntülerin PyTorch un tensor formatına dönmesini sağlar.  [3 kanal, en ,boy]
        #3 kanal RGB renkleri oluyor ve bunların ortalamaları alınıp normalize ediliyorlar. normalize sonucunda pixellerin z skore değerleri hesaplanmış oluyor.
        #(H,W,C) (Image tipi) ---> (C,H,W) (Tensor) oluyor.
        transforms.Normalize(mean = mean,std = std)

        #ilk üç adım PIL İmage üzerinde
        #4.adım PIL Image pytorch tensorune çevrilir.
        #4. adım normalized = value - mean/std  değerler  0-1 arasına gelir.
    ]
)
#sonra test için transform işlemi
transformT = transforms.Compose(
    [
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize(mean = mean,std = std) #test verimiz için de aynı çevrimleri yapıyoruz
    ]
)
#şimdi bu transformları verilerimize uygulayalım
data_dir = pathlib.Path(path)
train_set = torchvision.datasets.ImageFolder(data_dir.joinpath("train"), transform=transformTV)
val_set = torchvision.datasets.ImageFolder(data_dir.joinpath("val"), transform=transformTV)
test_set = torchvision.datasets.ImageFolder(data_dir.joinpath("test"), transform=transformT)

print("\nTransform Sonuçları\n")
print(train_set.transform)
print(val_set.transform)
print(test_set.transform)

################################## TRANSFORMSDAN SONRA VERİLERİMİZİ GÖRSELLEŞTİRDİK ###############################

# `brain_label` sözlüğü, etiket numarasına karşılık gelen sınıf adlarını saklar
brain_label = {
    0: 'Brain Tumor',
    1: 'Healthy'
}

# Yeni bir figür oluşturuyoruz, boyutu 10x10 inch olarak ayarlanıyor
figure = plt.figure(figsize=(10, 10))
cols, rows = 5, 5 #kaç adet resim olacağını da belirler

for i in range(1, cols * rows + 1):    # `train_set`ten rastgele bir resim ve etiket seçiyoruz
    sample_idx = torch.randint(len(train_set), size=(1,)).item()  # --> 0- train set boyutu kadar bir random değer seçilir
    #seçilen değerin formatı tensor([5]) şeklinde gelir . .item() ile bu 5'e çevrilir.
    img, label = train_set[sample_idx]  #label kolasörün adını alıyor yani etiketi

    # Bu görseli figürün içine uygun konuma yerleştiriyoruz
    ax = figure.add_subplot(rows, cols, i)

    # Görselin etiketine göre başlık rengi belirliyoruz
    # Eğer etiket 'Healthy' ise başlık yeşil, diğer durumda kırmızı olacak
    color = "green" if brain_label[label] == 'Healthy' else "red"

    # Başlık ayarlanıyor, font büyüklüğü 14 ve kalın yapılıyor, başlık rengi belirleniyor
    ax.set_title(brain_label[label], fontsize=14, fontweight='bold', color=color)

    # Görselin eksenlerini gizliyoruz yoksa her resim için ölçekler çıkıyor
    ax.axis("off")

    # Görselin Tensor formatından NumPy formatına çevrilişi
    img_np = img.numpy().transpose((1, 2, 0))  # (C, H, W) → (H, W, C) formatına dönüşüm PIL
    """(C, H, W) formatında bir Tensor'dan,
    (H, W, C) formatında bir NumPy array'ine dönüşüm yapılır."""
    # Görselin değerlerini 0 ile 1 arasında kısıtlıyoruz, böylece görselin doğru şekilde gösterilmesini sağlıyoruz
    img_valid_range = np.clip(img_np, 0, 1)

    # Görseli, belirlediğimiz renk aralığı ile gösteriyoruz
    ax.imshow(img_valid_range)

    """
    Tensor'dan NumPy array'e dönüşüm: Görüntü verilerini işlemeye uygun bir formata dönüştürür.
    Boyutların transpose edilmesi: Görselin boyutlarının doğru şekilde sıralanmasını sağlar, çünkü görselleştirme işlemi için matplotlib genellikle (H, W, C) formatını bekler.
    np.clip ile 0-1 arası normalizasyon: Görüntü değerlerinin doğru aralıkta olmasını sağlayarak, görselleştirme için doğru renklerin kullanılmasını sağlar.
    """

# Görseller arasındaki boşlukları ayarlıyoruz, hem yatay hem de dikey
plt.subplots_adjust(wspace=0.3, hspace=0.3)
plt.suptitle('Brain Tumor and Healthy Images', fontsize=25, fontweight='bold', y=0.95, color='black')
plt.show()



######################### VERİ YUKLEME #############################
batch= 100 #100'ER 100'ER veri alacak

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size = batch, shuffle = True)

#Bu kısmı her bachte gelen verilerin boyutunu görmek için yazdırdım.
""" Çıktılar aşağıdaki gibi gelecek bunların anlamı;

Shape of X : torch.Size([100, 3, 256, 256])
    bach 100 boyutlu bir resim 
    3 kanallı yani RGB
    256 EN 256 BOY
Shape of y: torch.Size([100]) torch.int64
    100 bach boyutu
    torch.int64 ise etiketli verilerde kullanılan tensordeki elemanların veri tipidir.
    X RESİM VERİSİ , Y İSE Etiketleri """

# Veri yükleyicilerdeki eleman sayılarını yazdırmak için kod
for key, value in {'Training data': train_loader, "Validation data": val_loader, "Test data": test_loader}.items():
    # Her bir batch'in boyutunu görelim
    for X, y in value:
        print(f"{key}:")
        print(f"Shape of X (batch): {X.shape}")
        print(f"Shape of y (batch): {y.shape} {y.dtype}")
        break

    # Toplam eleman sayısını görelim
    total_samples = len(value.dataset)
    total_batches = len(value)
    print(f"Total samples in {key}: {total_samples}")   #toplam örnek sayısı
    print(f"Total batches in {key}: {total_batches}")   # batch sayısı
    print(f"Each batch contains approximately {total_samples / total_batches:.0f} samples")   #her bathcteki örnek sayısı 100 99
    print()

"""
Örnek çıktı 

Training data:
Shape of X (batch): torch.Size([100, 3, 256, 256])
Shape of y (batch): torch.Size([100]) torch.int64
Total samples in Training data: 9659
Total batches in Training data: 97
Each batch contains approximately 100 samples

"""
######################### CNN MODELİ OLUŞTURMA ##############################

import torch.nn as nn  #PyTorchun sinir ağı modelleri yer alır
import torch.nn.functional as F  #aktivasyon fonksiyonları yer alır.
import torch.optim as optim   #optimizasyon algoritmalarını içerir.
from torch.optim.lr_scheduler import ReduceLROnPlateau   #train loss iyileşmediğinde azalmasını sağlar.


torch.backends.cudnn.benchmark = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

"""
torch.backends.cudnn.benchmark = True: Bu, CUDA ile çalışan sistemlerde, giriş verisinin boyutlarına göre en uygun algoritmayı seçmek için kullanılır. Bu, modelin performansını artırabilir.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'): Eğer bir GPU mevcutsa (cuda), model GPU'yu kullanacaktır. Aksi takdirde, model CPU üzerinde çalışacaktır.
print(f"Using device: {device}"): Çalışma cihazının (CPU veya GPU) hangi cihaz olduğu ekrana yazdırılır.
"""

# KENDİ CNN MODELİM
class BrainMRCNN(nn.Module):
    def __init__(self):   #modeli başlatan fonksiyon
        super(BrainMRCNN, self).__init__()
        # Convolutional Layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)  # 3*3 bir maske, padding işleminde kenarlara 0 eklenir.
        self.bn1 = nn.BatchNorm2d(16)  #Her bir mini-batch içindeki değerlerin ortalamasını sıfıra, standart sapmasını bire getirir
        """
        : Her konvolüsyon katmanından sonra, özellik haritalarını normalleştirerek öğrenmeyi hızlandırmak için kullanılan bir katmandır. Bu, modelin daha hızlı ve stabil öğrenmesine yardımcı olur.
        Aktivasyonları normalize eder (ortalama 0, standart sapma 1 olacak şekilde
        """

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        # Pooling Layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully Connected Layers
        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)  # Yeni eklenen katman
        self.fc4 = nn.Linear(128, 64)   # Yeni eklenen katman
        self.fc5 = nn.Linear(64, 2)     # Çıkış katmanı

        # Dropout Layer
        self.dropout = nn.Dropout(0.2)    # %20 oranında nöronu overfiti önlemek amacıyla devre dışı bırakır.

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))

        """
         ReLU (Rectified Linear Unit) aktivasyon fonksiyonu uygulanır. Bu, negatif değerleri sıfırlar ve modelin doğrusal olmayan ilişkileri öğrenmesini sağlar."""
        # Flatten işlemi
        x = torch.flatten(x, start_dim=1)  #2 boyut tek boyuta düzleştiriliyor.
        #sınıflandırma performansını arttırmak için birden fazla Fully connected katman uygulanır.

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))  # Yeni eklenen katman
        x = self.dropout(x)
        x = F.relu(self.fc4(x))  # Yeni eklenen katman
        x = self.dropout(x)
        x = self.fc5(x)  # Çıkış katmanı (Aktivasyon uygulanmaz)

        return F.log_softmax(x, dim=1)  #Çıkışa softmax fonksiyonu uygulanır. Bu, her sınıf için olasılık değerini döndürür.
        # Eğer CrossEntropyLoss kullanıyorsan, bunu kaldırabilirsin. ama NLLLoss Kullanıldığı için kalacak

    """
Katman	   Giriş	          Çıkış	            Açıklama
Conv1	(3, 256, 256)	(16, 128, 128)	3→16 filtre, pooling ile yarıya (Maxpooling  2*2 olduğu için her iki pixelde bir ilerliyoruz boy yarıya iniyor)
Conv2	(16, 128, 128)	(32, 64, 64)	16→32 filtre
Conv3	(32, 64, 64)	(64, 32, 32)	32→64 filtre
Conv4	(64, 32, 32)	(128, 16, 16)	64→128 filtre
Flatten	(128, 16, 16) → 32768		Tek boyutlu yapılır
FC1 → FC5	32768 → 2		Sonuçta 2 sınıf çıkışı alınır

Filtre sayısı ne kadar fazla olursa o  kadar detay öğrenilir. Örneğin 16 filtrenin olması
Bir giriş resmine uygulanacak 16 ayrı filtrenin olacağını söyler. Bu da sadece o katmanda 16 
ayrı özellik haritası çıkarılacağını belirtir.

Her konvulasyon katmanında bir önceki filtreden gelen feature maplerden ortak bir map çıkar.
Diğer konvulasyon katmanı bu map üzerinde işlem yapar.

Herbir konum için filtre sayısı kadar patch oluşur (sayı gelir. sayılar toplanarak o pixelin yeni değeri oluşturulur)

Fully Con. katmanlarında her gelen vektör bir ağırlık çarpanı ile çarpılarak bias uygulanır. 
Relu fonksiyonu negatif çıkan değerleri sıfırlar.
Bias başlangıç noktasını kaydırmaktadır. Sabit bir değerdir. Bias da ağılıklar ile birlikte güncellenir.

    """

# Modeli oluştur ve parametreleri yazdır
modelCNN = BrainMRCNN().to(device)
loss_func_cnn = nn.NLLLoss(reduction="sum")
opt_cnn = optim.Adam(modelCNN.parameters(), lr=1e-4)
lr_scheduler_cnn = ReduceLROnPlateau(opt_cnn, mode='min', factor=0.5, patience=20, verbose=1)

"""
2. loss_func_cnn = nn.NLLLoss(reduction="sum")
Bu satır, kayıp fonksiyonu (loss function) belirler. Burada NLLLoss (Negative Log Likelihood Loss) kullanılıyor.
nn.NLLLoss: Bu, çok sınıflı sınıflandırma problemlerinde kullanılan bir kayıp fonksiyonudur. Modelin çıktılarını, hedef etiketlerle karşılaştırarak kaybı hesaplar. Genelde, log softmax fonksiyonu ile birlikte kullanılır.
reduction="sum": Bu parametre, kaybın nasıl hesaplanacağını belirtir. "sum" değeri, tüm örnekler için kaybın toplamını alır. Alternatif olarak "mean" seçeneği, kaybın ortalamasını alır. "sum" kullanıldığında, toplam kaybı elde etmek için tüm mini-batch üzerindeki kayıplar toplanır.
Bu kayıp fonksiyonu, modelin tahmin ettiği log-olasılıklarla gerçek etiketler arasındaki farkı ölçer ve bu farkın toplamını geri döndürür.

3. opt_cnn = optim.Adam(modelCNN.parameters(), lr=1e-4)
Bu satır, optimizer (optimizatör) ayarlarını belirler. Burada, Adam optimizatörü kullanılıyor.
optim.Adam: Adam (Adaptive Moment Estimation), yaygın bir optimizasyon algoritmasıdır. Adam, öğrenme oranını her parametre için ayrı ayrı ayarlar ve daha hızlı ve etkili bir şekilde eğitimi sağlar. Adam, öğrenme oranı ve momentum gibi parametreleri otomatik olarak günceller.
modelCNN.parameters(): Bu, modeldeki tüm parametreleri (ağırlıklar ve biaslar) optimizer'a geçer. Bu sayede optimizer, bu parametreleri güncelleyebilir.
lr=1e-4: Bu, öğrenme oranını belirtir. Öğrenme oranı, optimizer’ın parametreleri ne kadar hızlı güncelleyeceğini belirler. 1e-4, yani 0.0001, genellikle iyi bir başlangıçtır. 
Çok büyük bir öğrenme oranı modelin çok hızlı öğrenmesine neden olabilir, ancak eğitimin stabilitesini bozabilir. Küçük bir öğrenme oranı ise daha güvenli bir eğitim süreci sağlar, ancak öğrenme süreci daha yavaş olabilir.

4. lr_scheduler_cnn = ReduceLROnPlateau(opt_cnn, mode='min', factor=0.5, patience=20, verbose=1)
Bu satır, öğrenme oranı planlayıcısı (learning rate scheduler) oluşturur. Burada kullanılan planlayıcı ReduceLROnPlateau'dir.
ReduceLROnPlateau: Bu scheduler, modelin doğruluğu veya kaybı sabit kaldığında veya iyileşme durduğunda öğrenme oranını azaltır. Bu, modelin eğitim sürecinde daha verimli bir şekilde öğrenmesini sağlar.
Bu planlayıcı, eğitim sırasında val_loss gibi bir metriği izler ve eğer belirli bir sayıda epoch boyunca bu metrik iyileşmezse, öğrenme oranını azaltır. Bu, modelin daha küçük öğrenme oranlarıyla daha hassas öğrenmesine yardımcı olur.
mode='min': Bu parametre, hangi metriğin izleneceğini belirler. mode='min' olarak ayarlandığında, val_loss gibi bir kayıp değeri küçüldükçe öğrenme oranı düşer. Yani, kayıp değeri iyileşmezse öğrenme oranı azaltılır.
factor=0.5: Bu parametre, öğrenme oranının ne kadar azaltılacağını belirtir. factor=0.5 demek, öğrenme oranının her azalmada yarıya düşeceği anlamına gelir.
patience=20: Bu, sabır parametresidir. Eğer belirtilen metrik (örneğin kayıp) patience kadar epoch boyunca iyileşmezse, öğrenme oranı azaltılmaya başlanır. Yani, patience=20 demek, 20 epoch boyunca kayıp iyileşmezse, öğrenme oranı düşürülür.
verbose=1: Bu, süreçle ilgili ayrıntıların yazdırılmasını sağlar. verbose=1 olduğunda, öğrenme oranının azalması hakkında bilgi yazdırılır.
"""
#--------------------------------------------------------------------------------------

# MODEL: DENSENET169
"""
DenseNet'in en önemli özelliği, her katmanın kendisinden önceki tüm katmanlardan giriş almasıdır. 
Eğer L katmanlı bir ağ varsa, geleneksel CNN'de L bağlantı vardır, ancak DenseNet'te L(L+1)/2 bağlantı bulunur. 
Bu, bilginin daha iyi akmasını ve gradyanların daha iyi yayılmasını sağlar.

Geleneksel CNN: x₀ → H₁ → H₂ → ... → Çıktı
DenseNet: x₀ → H₁ → H₂(x₀,H₁) → H₃(x₀,H₁,H₂) → ... → Çıktı

DenseNet169: Önceden eğitilmiş (pretrained) bir model kullanıyor. ImageNet veri seti üzerinde milyonlarca görüntüyle eğitilmiş derin bir mimari.
"""


from torchvision import models

# Pretrained DenseNet169 modelini yükle
modelDense = models.densenet169(pretrained=True)

# Tüm katmanları dondur (sadece yeni eklenen classifier eğitilecek)
for param in modelDense.parameters():
    param.requires_grad = False
"""
Bu kod, mevcut tüm katmanları "donduruyor" - yani bu katmanların ağırlıkları eğitim sırasında güncellenmeyecek.

DenseNet169, farklı bir görev (genel nesne tanıma) için eğitilmiş olsa da, ilk katmanları kenarlar, dokular ve şekiller gibi temel görsel özellikleri tanımayı öğrenmiştir.
Bu temel özellikler, beyin MR görüntülerindeki yapıları tanımak için de yararlıdır.
Dondurma işlemi ile, temel özellikleri tanıma yeteneğini korurken, sadece son katmanları (sınıflandırıcı) yeni görev için özelleştiriyorsunuz.

Bu yaklaşım, özellikle veri setiniz küçükse (birkaç yüz veya birkaç bin görüntü), sıfırdan eğitilen bir modele göre genellikle daha iyi sonuçlar verir ve eğitim süresi çok daha kısadır.
"""
# Son katmanı (classifier) değiştir
num_ftrs = modelDense.classifier.in_features
modelDense.classifier = nn.Sequential(
    nn.Linear(num_ftrs, 512),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(64, 2),  # Çıkış katmanı
    nn.LogSoftmax(dim=1)
)

# Modeli cihaza gönder
modelDense = modelDense.to(device)
# Kayıp fonksiyonu ve optimizer
loss_func_dense = nn.NLLLoss(reduction="sum")  # Eğer CrossEntropyLoss kullanıyorsan LogSoftmax'ı kaldır
opt_dense = optim.Adam(modelDense.classifier.parameters(), lr=1e-4)  # Sadece classifier eğitilecek
lr_scheduler_dense = ReduceLROnPlateau(opt_dense, mode='min', factor=0.5, patience=20, verbose=1)
#--------------------------------------------------------------------------------------

# MODEL: Inception-ResNetV2
"""
Inception-ResNetV2: İki güçlü mimariyi birleştirir - Inception ve ResNet.
Inception modülleri (paralel farklı boyutlardaki filtreler) ve ResNet'in artık bağlantılarını (residual connections) bir arada kullanır.

Google'ın geliştirdiği çok derin (572 katman) ve karmaşık bir mimaridir. 55.8 milyon parametre içerir.
En yüksek hesaplama ve bellek gereksinimine sahiptir.
Inception-ResNetV2'nin belirleyici özelliklerinden biri, paralel farklı boyutlarda filtreler (1x1, 3x3, 5x5) kullanmasıdır. 
Bu, farklı ölçeklerdeki özellikleri aynı anda yakalayabilme yeteneği sağlar.
"""

import pretrainedmodels
import torch
import torch.nn as nn

# Inception-ResNet-V2 modelini yükle
modelResnet = pretrainedmodels.__dict__['inceptionresnetv2'](pretrained='imagenet')
"""
 (pretrainedmodels) kullanılıyor. Bu kütüphane, PyTorch'ta resmi olarak bulunmayan bazı modern mimarilere erişim sağlar.
"""
# Tüm katmanları dondur (sadece son sınıflandırıcıyı eğiteceğiz)
for param in modelResnet.parameters():
    param.requires_grad = False

# Modelin son katmanını değiştiriyoruz
num_ftrs = modelResnet.last_linear.in_features  # Son katmandaki input boyutunu alıyoruz

# Adaptive Pooling ekleyerek hatayı düzelt
modelResnet.avgpool_1a = nn.AdaptiveAvgPool2d((1, 1))

# Son katmanı değiştir
modelResnet.last_linear = nn.Sequential(
    nn.Linear(num_ftrs, 512),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(64, 2),
    nn.LogSoftmax(dim=1)
)

modelResnet = modelResnet.to(device)
# Loss function ve optimizer
loss_func_resnet = nn.NLLLoss(reduction="sum")  # NLLLoss kullandık, çünkü LogSoftmax kullanıyoruz
opt_resnet = torch.optim.Adam(modelResnet.parameters(), lr=1e-4)
lr_scheduler_resnet = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_resnet, mode='min', factor=0.5, patience=20,
                                                                 verbose=1)

"""
Inception-ResNetV2 Tercih Edildiğinde:

En yüksek doğruluk gerektiren kritik uygulamalarda
Yeterli hesaplama gücü ve bellek mevcutsa
Veri seti yeterince büyükse veya güçlü veri artırma (data augmentation) kullanılıyorsa
Farklı ölçeklerdeki özellikleri tanımanın önemli olduğu durumlarda (örn. çok küçük tümörler ve geniş beyin yapıları bir arada)

DenseNet169 Tercih Edildiğinde:

Orta düzeyde hesaplama kaynakları varsa
İyi bir doğruluk/hesaplama oranı gerektiğinde
Veri seti küçük-orta boyutluysa
Özellik yeniden kullanımının önemli olduğu durumlarda

Özel CNN Tercih Edildiğinde:

Sınırlı hesaplama kaynaklarında
Çok küçük veri setlerinde
Hızlı eğitim ve çıkarım (inference) gerektiğinde
Problem basitse ve karmaşık özellik çıkarımı gerektirmiyorsa
"""

# --------------------------------------------------------------------------------------

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, \
    classification_report


def calculate_metrics(y_true, y_pred):
    """
    Performans metriklerini hesaplayan yardımcı fonksiyon
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')  # hasta tahmin ettiklerinden kaçı gerçekte hasta
    recall = recall_score(y_true, y_pred, average='weighted')  # gerçekte hasta olanların kaçını doğru bildi
    f1 = f1_score(y_true, y_pred, average='weighted')

    return accuracy, precision, recall, f1


def train_epoch(model, train_loader, loss_func, optimizer, device):
    model.train()
    total_loss = 0
    predictions = []
    true_labels = []

    # For loop başlamadan önce, kaç batch olduğunu yazdır
    total_batches = len(train_loader)
    print(f"Training on {len(train_loader.dataset)} samples with {total_batches} batches")

    for batch_idx, (data, target) in enumerate(train_loader):
        # İlerleme bilgisini yazdır
        print(f"Processing batch {batch_idx + 1}/{total_batches} ({(batch_idx + 1) * 100 / total_batches:.1f}%)")

        # Veriyi GPU'ya taşı
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()  # Her adımda önceki gradyanları sıfırlamak gerekir    Gradyanı kullanarak “hangi ağırlığı ne kadar değiştirmeliyim ki hata azalsın” sorusuna cevap veririz.
        output = model(data)  # Modelin tahmini (log_softmax çıktısı).
        loss = loss_func(output, target)  # NLLLoss, tahmin ile gerçek değer arasındaki kaybı hesaplar.

        loss.backward()  # Gradyanları hesaplar.
        optimizer.step()  # Bu gradyanlara göre modelin ağırlıklarını günceller.

        total_loss += loss.item()  # Her batch’teki loss, toplam loss’a eklenir.

        pred = output.argmax(dim=1, keepdim=True)  # Hangi sınıfa ait olduğunu bulur (örneğin [0.2, 0.8] → sınıf 1).
        # CPU'ya taşıyıp numpy'a çevir
        predictions.extend(
            pred.cpu().detach().numpy().flatten())  # NumPy’ya çevirerek gradyan takibinden çıkarılır ve CPU’ya alınır.
        true_labels.extend(target.cpu().numpy().flatten())

    accuracy, precision, recall, f1 = calculate_metrics(true_labels, predictions)
    avg_loss = total_loss / len(train_loader.dataset)

    return avg_loss, accuracy, precision, recall, f1


def validate(model, val_loader, loss_func,
             device):  # validation eğitmez sadece ne kadar eğitildiğini test eder. ağırlıklar güncellenmez, gradyan hesaplanmaz
    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []

    with torch.no_grad():
        for data, target in val_loader:
            # Veriyi GPU'ya taşı
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = loss_func(output, target)
            total_loss += loss.item()

            pred = output.argmax(dim=1, keepdim=True)
            # CPU'ya taşıyıp numpy'a çevir
            predictions.extend(pred.cpu().numpy().flatten())
            true_labels.extend(target.cpu().numpy().flatten())

    accuracy, precision, recall, f1 = calculate_metrics(true_labels, predictions)
    avg_loss = total_loss / len(val_loader.dataset)

    return avg_loss, accuracy, precision, recall, f1


def early_stopping(history, patience=3):
    val_losses = history['val_losses']

    if len(val_losses) < patience + 1:
        return len(val_losses)

    # Son 'patience' sayısı kadar epochta kayıp düşmediyse durdur
    for i in range(len(val_losses) - patience, len(val_losses)):
        if val_losses[i] < val_losses[i - 1]:
            return len(val_losses)

    # Erken durdurma gerekli
    print(f"\n💡 Erken Durdurma: Son {patience} epochta kayıp düşmedi!")
    return len(val_losses) - patience


def train_model(model, model_name, train_loader, val_loader, loss_func, optimizer, lr_scheduler, device,
                num_epochs=100):
    """
    Modeli eğiten ana fonksiyon
    """
    print(f"\n{'=' * 20} {model_name} MODELI EĞITILIYOR {'=' * 20}")

    # Metrik geçmişini tutmak için listeler
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    best_val_loss = float('inf')  # İlk başta çok büyük bir değer ata
    model_filename = f"{model_name}.pth"  # Model dosya adı

    for epoch in range(num_epochs):
        train_loss, train_acc, train_prec, train_recall, train_f1 = train_epoch(
            model, train_loader, loss_func, optimizer, device
        )
        val_loss, val_acc, val_prec, val_recall, val_f1 = validate(
            model, val_loader, loss_func, device
        )

        # En iyi modeli kaydetme
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_filename)
            print(f"Epoch {epoch + 1}: Best model saved with validation loss {val_loss:.4f}")

        # Learning rate scheduler'ı güncelle
        lr_scheduler.step(val_loss)

        # Metrikleri kaydet
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # Sonuçları yazdır
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print(
            f'Training - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}, Precision: {train_prec:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}')
        print(
            f'Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, Precision: {val_prec:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}\n')

        # Early stopping kontrolü
        if epoch >= 3:  # En az 3 epoch sonra kontrol et
            stop_epoch = early_stopping({
                'train_losses': train_losses,
                'val_losses': val_losses
            }, patience=3)

            if stop_epoch != len(train_losses):
                print(f"\n Eğitim {stop_epoch}. epochta erken durduruldu!")
                break

    print(f"\n{'=' * 20} {model_name} MODELI EĞITIMI TAMAMLANDI {'=' * 20}\n")

    return {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies
    }


def plot_metrics(history, model_name):
    """
    Eğitim ve validation metriklerini çizdiren fonksiyon
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Başlık ekleyelim
    fig.suptitle(f"{model_name} - Eğitim Metrikleri", fontsize=16)

    # Loss grafiği
    ax1.plot(history['train_losses'], label='Training Loss')
    ax1.plot(history['val_losses'], label='Validation Loss')
    ax1.set_title('Loss over epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()  # çıktıda sağ üstteki gösterim

    # Accuracy grafiği
    ax2.plot(history['train_accuracies'], label='Training Accuracy')
    ax2.plot(history['val_accuracies'], label='Validation Accuracy')
    ax2.set_title('Accuracy over epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    plt.tight_layout()
    plt.show()


def test_model(model, model_name, test_loader, device):
    # Modeli yükle
    model_filename = f"{model_name}.pth"
    model.load_state_dict(torch.load(model_filename, map_location=device, weights_only=True))  # modeli yüklüyoruz
    model.eval()

    print(f"\n{'=' * 20} {model_name} MODELI TEST EDILIYOR {'=' * 20}")

    # Loss fonksiyonunu tanımla
    criterion = torch.nn.NLLLoss()

    # Test için hazırlık
    all_preds = []
    all_labels = []
    total_loss = 0.0  # Test loss hesaplamak için değişken

    with torch.no_grad():  # geriye yayılım ve gradyan hesaplama kapatılır.
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            # Model tahmini yap
            output = model(data)

            # Loss'u hesapla
            loss = criterion(output, target)
            total_loss += loss.item()  # Batch loss'unu topla

            # Tahminleri al
            pred = output.argmax(dim=1, keepdim=True)
            all_preds.extend(pred.cpu().numpy().flatten())
            all_labels.extend(target.cpu().numpy().flatten())

    # Ortalama test loss'u hesapla
    average_test_loss = total_loss / len(test_loader)
    print(f'\nTest Loss: {average_test_loss:.4f}')

    # Confusion Matrix hesapla
    cm = confusion_matrix(all_labels, all_preds)

    # Sınıf isimleri
    class_names = ['Brain Tumor', 'Healthy']

    # Performans raporu
    print("\nDetaylı Performans Raporu:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # Confusion Matrix görselleştirme
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{model_name} - Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()

    print(f"\n{'=' * 20} {model_name} MODELI TEST TAMAMLANDI {'=' * 20}\n")

    return cm, average_test_loss


def test_single_image(model, model_name, image_path, mean, std, device):
    print(f"\n{model_name} modeli ile görüntü test ediliyor: {os.path.basename(image_path)}")

    # Modeli yükle
    model_filename = f"{model_name}.pth"
    model.load_state_dict(torch.load(model_filename, map_location=device))

    # Ön işleme için dönüşümleri tanımla
    transformT = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)  # Test verisi için de aynı işlemler
    ])

    # Görüntüyü aç ve dönüştür
    image = Image.open(image_path).convert("RGB")
    image = transformT(image).unsqueeze(0).to(device)  # Batch boyutu ekleyerek modele uygun hale getir

    # Modeli değerlendir moduna al
    model.eval()
    with torch.no_grad():
        output = model(image)
        prediction = output.argmax(dim=1).item()  # En yüksek olasılığa sahip sınıfı al

    # Etiketleri sözlük ile eşleştir
    label_dict = {0: "🧠 Brain Tumor", 1: "✅ Healthy"}
    print(f"{model_name} Tahmini: {label_dict[prediction]}")


# Ana eğitim ve test döngüsü
def train_and_evaluate_models(models_dict, train_loader, val_loader, test_loader, device, mean, std, num_epochs=15):
    """
    Birden fazla modeli sırayla eğitip test eden fonksiyon

    Args:
        models_dict: {model_name: (model, loss_func, optimizer, scheduler)} şeklinde sözlük
        train_loader: Eğitim veri yükleyicisi
        val_loader: Doğrulama veri yükleyicisi
        test_loader: Test veri yükleyicisi
        device: Eğitim cihazı (cuda/cpu)
        mean: Normalizasyon için ortalama değerler
        std: Normalizasyon için standart sapma değerleri
        num_epochs: Maksimum epoch sayısı
    """
    model_results = {}  # sonuçlar buraya gelecek

    # Her modeli sırayla eğit ve test et
    for model_name, (
    model, loss_func, optimizer, scheduler) in models_dict.items():  # items key value değerini tupple şeklinde döndürür
        # Eğer model dosyası yoksa, modeli eğit
        if not os.path.exists(f"{model_name}.pth"):
            history = train_model(
                model=model,
                model_name=model_name,
                train_loader=train_loader,
                val_loader=val_loader,
                loss_func=loss_func,
                optimizer=optimizer,
                lr_scheduler=scheduler,
                device=device,
                num_epochs=num_epochs
            )

            # Eğitim metriklerini çizdir
            plot_metrics(history, model_name)

        # Modeli test et
        cm, avg_test_loss = test_model(model, model_name, test_loader, device)
        model_results[model_name] = {'confusion_matrix': cm, 'test_loss': avg_test_loss}

    # Tüm modelleri karşılaştır
    print("\n" + "=" * 60)
    print("MODELLERIN KARŞILAŞTIRMASI")
    print("=" * 60)

    for model_name, results in model_results.items():
        print(f"{model_name} - Test Loss: {results['test_loss']:.4f}")

    print("=" * 60)

    # Test görüntüleri üzerinde her modeli dene
    test_images = [
        r"...new output folder path...\brain\test\Brain Tumor\flip_Cancer (1841).jpg",
        r"...new output folder path...\brain\test\Brain Tumor\flip_Cancer (1745).jpg",
        r"...new output folder path...\brain\test\Brain Tumor\original_Cancer (819).jpg",
        r"...new output folder path...\brain\test\Healthy\flip_Not Cancer  (118).jpg"
    ]

    for image_path in test_images:
        print("\n" + "-" * 60)
        print(f"Test Image: {os.path.basename(image_path)}")  # yolun son ksımından resmin sedece ismi alınıyor
        print("-" * 60)

        for model_name, (model, _, _, _) in models_dict.items():
            test_single_image(model, model_name, image_path, mean, std, device)


# Modelleri bir sözlükte toplayın
models_dict = {
    'modelCNN': (modelCNN, loss_func_cnn, opt_cnn, lr_scheduler_cnn),
    'modelDense': (modelDense, loss_func_dense, opt_dense, lr_scheduler_dense),
    'modelResnet': (modelResnet, loss_func_resnet, opt_resnet, lr_scheduler_resnet)
}

# Tüm modelleri eğit ve test et
train_and_evaluate_models(
    models_dict,
    train_loader,
    val_loader,
    test_loader,
    device,
    mean,  # Normalizasyon için mean değeri
    std,  # Normalizasyon için std değeri
    num_epochs=15
)