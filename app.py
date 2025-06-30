from flask import Flask, request, jsonify, render_template
import torch
from torchvision import transforms
from PIL import Image
from modelcnn import BrainMRCNN  # Modeli içe aktar

app = Flask(__name__)  #Flask uygulamasını başlatır. Artık bu dosya bir web uygulaması haline geldi.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "modelCNN2.pth"  # Model dosyasının adı
model = BrainMRCNN().to(device)
model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
model.eval()

transformT = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.2448, 0.2450, 0.2451], std=[0.2313, 0.2313, 0.2314])
])

label_dict = {0: "🧠 Brain Tumor", 1: "✅ Healthy"}

#route kurallarının içeriği yapıldığı zaman fonksiyonlar çalışmaya başlar.

@app.route('/')   # Tarayıcıda siteye girince templates/index.html dosyasını render eder.
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])  #resmi tahmin etmek isterse postmesajına bakılır resim alınıp sonuç tahmin edilir.
def predict():  #Bu kısım POST ile resim yüklendiğinde çalışır.
    if 'image' not in request.files:  #post ile gelen dosya
        return jsonify({'error': 'Lütfen bir görüntü dosyası yükleyin!'}), 400

    file = request.files['image']
    image = Image.open(file).convert("RGB")
    image = transformT(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        prediction = output.argmax(dim=1).item()

    result = {"prediction": label_dict[prediction]}
    return jsonify(result)  #Sonucu JSON olarak döndürür.

if __name__ == '__main__':  #PROGRAMIN TERMİNALDE ÇALIŞMASINI SAĞLAR
    app.run(debug=True)
