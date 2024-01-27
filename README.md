
# Churn Prediction

<img src="https://github.com/baynazoglu/Churn-Prediction/blob/main/customer%20churn.png" alt="Image" width="600" height="220">


## Business Problem

This project aims to develop a machine learning model that can predict which customers of a telecommunications company are likely to churn.

## Dataset Story

The TELXX churn dataset contains information about a hypothetical telecommunications company that provides home phone and internet services to 7043 customers in California during the third quarter. It shows which customers have churned, stayed, or signed up for services.

### Dataset

- 21 Variables
- 7043 Observations
- 977.5 KB

Below are the descriptions of the variables:

- CustomerId: Customer ID
- Gender: Gender
- SeniorCitizen: Whether the customer is a senior citizen (1: Yes, 0: No)
- Partner: Whether the customer has a partner (Yes, No)
- Dependents: Whether the customer has dependents (Yes, No)
- tenure: Number of months the customer has stayed with the company
- PhoneService: Whether the customer has phone service (Yes, No)
- MultipleLines: Whether the customer has multiple lines (Yes, No, No phone service)
- InternetService: Customer's internet service provider (DSL, Fiber optic, No)
- OnlineSecurity: Whether the customer has online security (Yes, No, No internet service)
- OnlineBackup: Whether the customer has online backup (Yes, No, No internet service)
- DeviceProtection: Whether the customer has device protection (Yes, No, No internet service)
- TechSupport: Whether the customer has tech support (Yes, No, No internet service)
- StreamingTV: Whether the customer has streaming TV (Yes, No, No internet service)
- StreamingMovies: Whether the customer has streaming movies (Yes, No, No internet service)
- Contract: The contract term of the customer (Month-to-month, One year, Two year)
- PaperlessBilling: Whether the customer has opted for paperless billing (Yes, No)
- PaymentMethod: The customer's payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic))
- MonthlyCharges: The amount charged to the customer monthly
- TotalCharges: The total amount charged to the customer
- Churn: Whether the customer has churned (Yes, No)
  
## Installation

1. Clone this project: `git clone https://github.com/YOUR_USERNAME/Feature-Engineering.git`
2. Navigate to the project directory: `cd Feature-Engineering`
3. Install the required dependencies: `pip install -r requirements.txt`
4. Run the project: `python main.py`

## Usage

1. Add the dataset to the "data" folder.
2. Run the project files.
3. Perform data analysis and feature engineering steps.
4. Develop a machine learning model to predict whether individuals have diabetes or not.
5. Evaluate the model's performance and analyze the results.

## Contributing

1. Fork this project.
2. Create a new branch: `git checkout -b feature/NewFeature`
3. Make your changes and commit them: `git commit -am 'Added a new feature'`
4. Push your branch to the forked repository: `git push origin feature/NewFeature`
5. Create a pull request.


-----------------------------------------------------------
# Churn Prediction

## İş Problemi

Bu proje, bir telekomünikasyon şirketinin müşterilerinden hangilerinin şirketi terk edeceğini tahminleyebilen bir makine öğrenmesi modeli geliştirmeyi amaçlamaktadır.

## Veri Seti Hikayesi

TELXX müşteri kaybı verileri, Kaliforniya'da hayali bir telekomünikasyon şirketi tarafından üçüncü çeyrekte sağlanan ev telefonu ve internet hizmetleriyle ilgili 7043 müşterinin bilgilerini içermektedir. Bu veri seti, hangi müşterilerin hizmetlerden ayrıldığını, hala müşteri olduklarını veya hizmete yeni kaydolduklarını göstermektedir.

### Veri Seti

- 21 Değişken
- 7043 Gözlem
- 977.5 KB

Aşağıda değişkenlerin açıklamaları verilmiştir:

- CustomerId: Müşteri ID'si
- Gender: Cinsiyet
- SeniorCitizen: Müşterinin yaşlı olup olmadığı (1: Evet, 0: Hayır)
- Partner: Müşterinin bir ortağı olup olmadığı (Evet, Hayır)
- Dependents: Müşterinin bakmakla yükümlü olduğu kişiler olup olmadığı (Evet, Hayır)
- tenure: Müşterinin şirkette kaldığı ay sayısı
- PhoneService: Müşterinin telefon hizmeti olup olmadığı (Evet, Hayır)
- MultipleLines: Müşterinin birden fazla hattı olup olmadığı (Evet, Hayır, Telefon hizmeti yok)
- InternetService: Müşterinin internet servis sağlayıcısı (DSL, Fiber optik, Yok)
- OnlineSecurity: Müşterinin çevrimiçi güvenliğinin olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
- OnlineBackup: Müşterinin online yedeğinin olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
- DeviceProtection: Müşterinin cihaz korumasına sahip olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
- TechSupport: Müşterinin teknik destek alıp almadığı (Evet, Hayır, İnternet hizmeti yok)
- StreamingTV: Müşterinin TV yayını hizmeti olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
- StreamingMovies: Müşterinin film akışı hizmeti olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
- Contract: Müşterinin sözleşme süresi (Aydan aya, Bir yıl, İki yıl)
- PaperlessBilling: Müşterinin kağıtsız fatura seçeneğini tercih edip etmediği (Evet, Hayır)
- PaymentMethod: Müşterinin ödeme yöntemi (Elektronik çek, Posta çeki, Banka havalesi (otomatik), Kredi kartı (otomatik))
- MonthlyCharges: Müşteriden aylık olarak tahsil edilen tutar
- TotalCharges: Müşteriden tahsil edilen toplam tutar
- Churn: Müşterinin hizmeti bırakıp bırakmadığı (Evet, Hayır)
## Kurulum

1. Bu projeyi klonlayın: `git clone https://github.com/YOUR_USERNAME/Feature-Engineering.git`
2. Proje dizinine gidin: `cd Feature-Engineering`
3. Gerekli bağımlılıkları yükleyin: `pip install -r requirements.txt`
4. Projeyi çalıştırın: `python main.py`

## Kullanım

1. Veri setini "data" klasörüne ekleyin.
2. Proje dosyalarını çalıştırın.
3. Veri analizi ve özellik mühendisliği adımlarını gerçekleştirin.
4. Makine öğrenmesi modeli geliştirerek, kişilerin diyabet hastası olup olmadığını tahmin edin.
5. Modelin performansını değerlendirin ve sonuçları analiz edin.

## Katkıda Bulunma

1. Bu projeyi fork edin.
2. Yeni bir dal oluşturun: `git checkout -b feature/YeniOzellik`
3. Değişikliklerinizi yapın ve bunları kaydedin: `git commit -am 'Yeni bir özellik eklendi'`
4. Dalınızı forked repository'e gönderin: `git push origin feature/YeniOzellik`
5. Bir pull isteği oluşturun.

