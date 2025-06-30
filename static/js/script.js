

// bu kısım htmlde form  kısmında image-form elemanına bir dosya gönderme özelliği ekler ve event fonksiyonunu çalıştırır.
document.getElementById('image-form').addEventListener('submit', function(event) {
    event.preventDefault();  //sayfa yenilenmesini engelle
    let formData = new FormData(); //formu sunucuya göndermek için
    let imageFile = document.getElementById('image').files[0]; //birden fazla resim gelirse ilk resmi alır
    
    if (imageFile) {
        formData.append('image', imageFile);  //sunucuya image isimli dosya gidecek
        
        // Resim önizlemesi
        let reader = new FileReader();  //dosya okuyucu nesne
        reader.onload = function(e) {
            document.getElementById('image-preview').src = e.target.result;  //okunan görselin base64 kodunu src ye gönderir
            // Sonuç bölümünü görünür yap
            document.getElementById('result-section').style.display = 'block';
        }
        reader.readAsDataURL(imageFile);  //görseli base64 gibi okur
        
        // API isteği POST isteği ile form gönderiliyor.
        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json()) // sunucudan dönen cevabı json a çeviriyor
        .then(data => {
            let result = data.prediction; //burada data sunucuda gelen json cevabı, result brain tumor/healthy
            document.getElementById('result').innerHTML = result;
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Bir hata oluştu, lütfen tekrar deneyin!');
        });
    }
    else {
        alert('Lütfen bir görüntü yükleyin!');
    }
});