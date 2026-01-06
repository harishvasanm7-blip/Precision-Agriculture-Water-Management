import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta
import time

# ==========================================
# 1. APP CONFIGURATION
# ==========================================
st.set_page_config(page_title="Ecoverse | Pan-India System", page_icon="🇮🇳", layout="wide")

st.markdown("""
    <style>
    .stMetric { background-color: #f0f2f6; padding: 10px; border-radius: 10px; }
    .stButton>button { width: 100%; border-radius: 5px; background-color: #2e7d32; color: white; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. KNOWLEDGE BASE
# ==========================================
ALL_CROPS = [
    "Wheat", "Rice", "Corn", "Soybean", "Onion", "Sugarcane", "Cotton", "Sesame",
    "Groundnut", "Mustard", "Tea", "Coffee", "Rubber", "Coconut", "Jute", "Potato", 
    "Tomato", "Banana", "Pulse", "Millet", "Turmeric", "Ginger", "Garlic", "Chilli", 
    "Pepper", "Saffron", "Apple", "Mango", "Orange", "Grapes"
]

STATE_CROP_MAP = {
    "Andhra Pradesh": ["Rice", "Chilli", "Cotton", "Groundnut", "Turmeric"],
    "Arunachal Pradesh": ["Rice", "Corn", "Millet"],
    "Assam": ["Tea", "Rice", "Jute", "Banana"],
    "Bihar": ["Rice", "Wheat", "Corn", "Pulse"],
    "Chhattisgarh": ["Rice", "Pulse", "Soybean"],
    "Goa": ["Rice", "Coconut", "Cashew", "Mango"],
    "Gujarat": ["Cotton", "Groundnut", "Sesame", "Onion"],
    "Haryana": ["Wheat", "Rice", "Mustard", "Cotton"],
    "Himachal Pradesh": ["Apple", "Corn", "Wheat"],
    "Jharkhand": ["Rice", "Corn", "Pulse"],
    "Karnataka": ["Coffee", "Rice", "Sugarcane", "Coconut"],
    "Kerala": ["Rubber", "Coconut", "Pepper", "Tea", "Rice", "Banana"],
    "Madhya Pradesh": ["Soybean", "Wheat", "Pulse", "Garlic"],
    "Maharashtra": ["Sugarcane", "Cotton", "Soybean", "Onion", "Grapes", "Mango"],
    "Manipur": ["Rice", "Corn", "Chilli"],
    "Meghalaya": ["Rice", "Ginger", "Turmeric"],
    "Mizoram": ["Rice", "Ginger", "Turmeric"],
    "Nagaland": ["Rice", "Corn", "Millet"],
    "Odisha": ["Rice", "Pulse", "Jute", "Turmeric"],
    "Punjab": ["Wheat", "Rice", "Cotton", "Sugarcane"],
    "Rajasthan": ["Mustard", "Millet", "Wheat", "Corn"],
    "Sikkim": ["Rice", "Cardamom", "Ginger"],
    "Tamil Nadu": ["Rice", "Sugarcane", "Groundnut", "Coconut", "Banana", "Turmeric"],
    "Telangana": ["Rice", "Cotton", "Turmeric"],
    "Tripura": ["Rice", "Rubber", "Tea"],
    "Uttar Pradesh": ["Wheat", "Sugarcane", "Rice", "Potato"],
    "Uttarakhand": ["Rice", "Wheat", "Sugarcane"],
    "West Bengal": ["Rice", "Jute", "Potato", "Tea"],
    "Jammu & Kashmir": ["Saffron", "Apple", "Rice"],
    "Andaman & Nicobar": ["Coconut", "Rice", "Banana"],
    "Delhi": ["Wheat", "Rice"],
    "Puducherry": ["Rice", "Coconut"]
}

SEASONS = ["Kharif (Monsoon)", "Rabi (Winter)", "Zaid (Summer)"]

# ==========================================
# 3. AI ENGINE
# ==========================================
@st.cache_resource
def get_trained_model():
    np.random.seed(42)
    n = 5000
    df = pd.DataFrame({
        'soil': np.random.uniform(10, 90, n),
        'temp': np.random.uniform(15, 45, n),
        'humid': np.random.uniform(20, 90, n),
        'crop': np.random.randint(0, len(ALL_CROPS), n)
    })
    
    high_water = [1, 5, 13, 14, 17, 20] 
    low_water = [6, 7, 9, 19] 
    
    conds = [
        (df['crop'].isin(high_water)) & (df['soil'] < 60),
        (df['crop'].isin(low_water)) & (df['soil'] < 25),
        (~df['crop'].isin(high_water + low_water)) & (df['soil'] < 40),
        (df['temp'] > 35) & (df['soil'] < 50)
    ]
    df['needed'] = np.select(conds, [1, 1, 1, 1], default=0)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(df[['soil', 'temp', 'humid', 'crop']], df['needed'])
    return model

model = get_trained_model()
CROP_MAP = {name: i for i, name in enumerate(ALL_CROPS)}

# ==========================================
# 4. MASTER TRANSLATION DATABASE
# ==========================================
TRANSLATIONS = {
    "English": {
        "title": "💧 Smart Irrigation System", "lbl_state": "Select State", "lbl_season": "Select Season",
        "lbl_soil": "Soil Moisture", "lbl_temp": "Temperature", "lbl_humid": "Humidity", "lbl_crop": "Crop",
        "btn_analyze": "Analyze", "alert_irrigate": "🚨 IRRIGATION REQUIRED", "alert_safe": "✅ OPTIMAL",
        "rec": "Recommendation: Pump ON", "modes": ["Manual Input", "Batch Upload", "History Tracker"],
        "hist_head": "📊 Environmental History", "log_head": "💧 Irrigation Tracker (Last 30 Days)",
        "crops": {c: c for c in ALL_CROPS},
        "states": {s: s for s in STATE_CROP_MAP.keys()},
        "seasons": {s: s for s in SEASONS}
    },
    "தமிழ் (Tamil)": {
        "title": "💧 ஸ்மார்ட் நீர்ப்பாசனம்", "lbl_state": "மாநிலம்", "lbl_season": "பருவம்",
        "lbl_soil": "மண் ஈரம்", "lbl_temp": "வெப்பநிலை", "lbl_humid": "ஈரப்பதம்", "lbl_crop": "பயிர்",
        "btn_analyze": "பகுப்பாய்வு", "alert_irrigate": "🚨 நீர்ப்பாசனம் தேவை", "alert_safe": "✅ சரியಾಗಿದೆ",
        "rec": "பரிந்துரை: பம்ப் ஆன்", "modes": ["கையேடு", "பதிவேற்றம்", "வரலாறு"],
        "hist_head": "📊 சுற்றுச்சூழல் வரலாறு", "log_head": "💧 கண்காணிப்பு",
        "crops": {"Rice": "அரிசி", "Coconut": "தேங்காய்", "Banana": "வாழை", "Sugarcane": "கரும்பு", "Cotton": "பருத்தி", "Tea": "தேயிலை", "Turmeric": "மஞ்சள்", "Groundnut": "நிலக்கடலை", "Rubber": "ரப்பர்", "Mango": "மாம்பழம்", "Onion": "வெங்காயம்", "Tomato": "தக்காளி", "Pepper": "மிளகு", "Chilli": "மிளகாய்"},
        "states": {
            "Tamil Nadu": "தமிழ்நாடு", "Kerala": "கேரளா", "Karnataka": "கர்நாடகா", "Andhra Pradesh": "ஆந்திரப் பிரதேசம்", 
            "Telangana": "தெலுங்கானா", "Maharashtra": "மகாராஷ்டிரா", "Delhi": "டெல்லி", "Punjab": "பஞ்சாப்",
            "Gujarat": "குஜராத்", "Rajasthan": "ராஜஸ்தான்", "West Bengal": "மேற்கு வங்கம்", "Odisha": "ஒடிசா",
            "Uttar Pradesh": "உத்தரப் பிரதேசம்", "Bihar": "பீகார்", "Assam": "அசாம்", "Jammu & Kashmir": "ஜம்மு காஷ்மீர்",
            "Andaman & Nicobar": "அந்தமான் நிக்கோபார்", "Puducherry": "புதுச்சேரி"
        },
        "seasons": {"Kharif (Monsoon)": "காரிஃப் (மழை)", "Rabi (Winter)": "ராபி (குளிர்காலம்)", "Zaid (Summer)": "சையத் (கோடை)"}
    },
    "हिन्दी (Hindi)": {
        "title": "💧 स्मार्ट सिंचाई प्रणाली", "lbl_state": "राज्य", "lbl_season": "मौसम",
        "lbl_soil": "मिट्टी की नमी", "lbl_temp": "तापमान", "lbl_humid": "नमी", "lbl_crop": "फसल",
        "btn_analyze": "विश्लेषण", "alert_irrigate": "🚨 सिंचाई आवश्यक", "alert_safe": "✅ अनुकूल",
        "rec": "सुझाव: पंप चालू करें", "modes": ["मैनुअल", "बैच अपलोड", "इतिहास ट्रैकर"],
        "hist_head": "📊 पर्यावरण इतिहास", "log_head": "💧 सिंचाई ट्रैकर (30 दिन)",
        "crops": {"Rice": "चावल", "Wheat": "गेहूं", "Corn": "मक्का", "Sugarcane": "गन्ना", "Cotton": "कपास", "Mango": "आम", "Potato": "आलू", "Tomato": "टमाटर", "Banana": "केला", "Onion": "प्याज", "Tea": "चाय"},
        "states": {
            "Punjab": "पंजाब", "Kerala": "केरल", "Maharashtra": "महाराष्ट्र", "Tamil Nadu": "तमिलनाडु", 
            "Uttar Pradesh": "उत्तर प्रदेश", "Gujarat": "गुजरात", "Rajasthan": "राजस्थान", "Karnataka": "कर्नाटक", 
            "West Bengal": "पश्चिम बंगाल", "Bihar": "बिहार", "Andhra Pradesh": "आंध्र प्रदेश", "Telangana": "तेलंगाना", 
            "Madhya Pradesh": "मध्य प्रदेश", "Odisha": "ओडिशा", "Haryana": "हरियाणा", "Assam": "असम", 
            "Delhi": "दिल्ली", "Andaman & Nicobar": "अंडमान और निकोबार"
        },
        "seasons": {"Kharif (Monsoon)": "खरीफ (मानसून)", "Rabi (Winter)": "रबी (सर्दी)", "Zaid (Summer)": "जायद (गर्मी)"}
    },
    "తెలుగు (Telugu)": {
        "title": "💧 స్మార్ట్ నీటిపారుదల", "lbl_state": "రాష్ట్రం", "lbl_season": "సీజన్", "lbl_soil": "నేల తేమ", "lbl_temp": "ఉష్ణోగ్రత", "lbl_humid": "తేమ", "lbl_crop": "పంట", "btn_analyze": "విశ్లేషించండి", "alert_irrigate": "🚨 నీరు అవసరం", "alert_safe": "✅ బాగుంది", "rec": "సలహా: మోటార్ ఆన్", "modes": ["మాన్యువల్", "అప్‌లోడ్", "చరిత్ర"], "hist_head": "📊 చరిత్ర", "log_head": "💧 లాగ్",
        "crops": {"Rice": "బియ్యం", "Chilli": "మిరప", "Turmeric": "పసుపు", "Cotton": "పత్తి", "Corn": "మొక్కజొన్న"},
        "states": {"Andhra Pradesh": "ఆంధ్రప్రదేశ్", "Telangana": "తెలంగాణ", "Karnataka": "కర్ణాటక", "Tamil Nadu": "తమిళనాడు"},
        "seasons": {"Kharif (Monsoon)": "ఖరీఫ్", "Rabi (Winter)": "రబీ", "Zaid (Summer)": "జైద్"}
    },
    "ಕನ್ನಡ (Kannada)": {
        "title": "💧 ಸ್ಮಾರ್ಟ್ ನೀರಾವರಿ", "lbl_state": "ರಾಜ್ಯ", "lbl_season": "ಋತು", "lbl_soil": "ಮಣ್ಣಿನ ತೇವಾಂಶ", "lbl_temp": "ತಾಪಮಾನ", "lbl_humid": "ಆರ್ದ್ರತೆ", "lbl_crop": "ಬೆಳೆ", "btn_analyze": "ವಿಶ್ಲೇಷಿಸಿ", "alert_irrigate": "🚨 ನೀರಾವರಿ ಅಗತ್ಯ", "alert_safe": "✅ ಉತ್ತಮ", "rec": "ಸಲಹೆ: ಪಂಪ್ ಆನ್", "modes": ["ಮ್ಯಾನುಯಲ್", "ಅಪ್‌ಲೋಡ್", "ಇತಿಹಾಸ"], "hist_head": "📊 ಇತಿಹಾಸ", "log_head": "💧 ದಾಖಲೆ",
        "crops": {"Rice": "ಅಕ್ಕಿ", "Coconut": "ತೆಂಗಿನಕಾಯಿ", "Sugarcane": "ಕಬ್ಬು", "Coffee": "ಕಾಫಿ"},
        "states": {"Karnataka": "ಕರ್ನಾಟಕ", "Kerala": "ಕೇರಳ", "Maharashtra": "ಮಹಾರಾಷ್ಟ್ರ"},
        "seasons": {"Kharif (Monsoon)": "ಮುಂಗಾರು", "Rabi (Winter)": "ಹಿಂಗಾರು", "Zaid (Summer)": "ಬೇಸಿಗೆ"}
    },
    "മലയാളം (Malayalam)": {
        "title": "💧 സ്മാർട്ട് ജലസേചനം", "lbl_state": "സംസ്ഥാനം", "lbl_season": "സീസൺ", "lbl_soil": "ഈർപ്പം", "lbl_temp": "താപനില", "lbl_humid": "അന്തരീക്ഷം", "lbl_crop": "വിള", "btn_analyze": "പരിശോധിക്കുക", "alert_irrigate": "🚨 നനയ്ക്കണം", "alert_safe": "✅ കുഴപ്പമില്ല", "rec": "നിർദ്ദേശം: പമ്പ് ഓൺ", "modes": ["മാനുവൽ", "അപ്‌ലോഡ്", "ചരിത്രം"], "hist_head": "📊 ചരിത്രം", "log_head": "💧 രേഖകൾ",
        "crops": {"Rice": "അരി", "Coconut": "തേങ്ങ", "Rubber": "റബ്ബർ", "Banana": "വാഴ", "Pepper": "കുരുമുളക്"},
        "states": {"Kerala": "കേരളം", "Tamil Nadu": "തമിഴ്നാട്"},
        "seasons": {"Kharif (Monsoon)": "വർഷകാലം", "Rabi (Winter)": "ശൈത്യകാലം", "Zaid (Summer)": "വേനൽക്കാലം"}
    },
    "বাংলা (Bengali)": {
        "title": "💧 স্মার্ট সেচ", "lbl_state": "রাজ্য", "lbl_season": "ঋতু", "lbl_soil": "মাটির আর্দ্রতা", "lbl_temp": "তাপমাত্রা", "lbl_humid": "আর্দ্রতা", "lbl_crop": "ফসল", "btn_analyze": "বিশ্লেষণ", "alert_irrigate": "🚨 সেচ প্রয়োজন", "alert_safe": "✅ ঠিক আছে", "rec": "পরামর্শ: পাম্প চালান", "modes": ["ম্যানুয়াল", "আপলোড", "ইতিহাস"], "hist_head": "📊 ইতিহাস", "log_head": "💧 সেচ লগ",
        "crops": {"Rice": "চাল", "Jute": "পাট", "Potato": "আলু", "Tea": "চা"},
        "states": {"West Bengal": "পশ্চিমবঙ্গ", "Assam": "আসাম"},
        "seasons": {"Kharif (Monsoon)": "খারিফ", "Rabi (Winter)": "রবি", "Zaid (Summer)": "জায়েদ"}
    },
    "ગુજરાતી (Gujarati)": {
        "title": "💧 સ્માર્ટ સિંચાઈ", "lbl_state": "રાજ્ય", "lbl_season": "મોસમ", "lbl_soil": "જમીન ભેજ", "lbl_temp": "તાપમાન", "lbl_humid": "ભેજ", "lbl_crop": "પાક", "btn_analyze": "વિશ્લેષણ", "alert_irrigate": "🚨 સિંચાઈ જરૂરી", "alert_safe": "✅ બરાબર છે", "rec": "ભલામણ: પંપ ચાલુ", "modes": ["મેન્યુઅલ", "અપલોડ", "ઇતિહાસ"], "hist_head": "📊 ઇતિહાસ", "log_head": "💧 સિંચાઈ લોગ",
        "crops": {"Cotton": "કપાસ", "Groundnut": "મગફળી", "Mango": "કેરી", "Onion": "ડુંગળી"},
        "states": {"Gujarat": "ગુજરાત", "Maharashtra": "મહારાષ્ટ્ર"},
        "seasons": {"Kharif (Monsoon)": "ખરીફ", "Rabi (Winter)": "રવિ", "Zaid (Summer)": "ઉનાળુ"}
    },
    "मराठी (Marathi)": {
        "title": "💧 स्मार्ट सिंचन", "lbl_state": "राज्य", "lbl_season": "हंगाम", "lbl_soil": "मातीची आर्द्रता", "lbl_temp": "तापमान", "lbl_humid": "आर्द्रता", "lbl_crop": "पीक", "btn_analyze": "विश्लेषण", "alert_irrigate": "🚨 पाणी देणे गरजेचे", "alert_safe": "✅ उत्तम", "rec": "सल्ला: पंप चालू करा", "modes": ["मॅन्युअल", "अपलोड", "इतिहास"], "hist_head": "📊 इतिहास", "log_head": "💧 सिंचन लॉग",
        "crops": {"Sugarcane": "ऊस", "Cotton": "कापूस", "Onion": "कांदा", "Grapes": "द्राक्षे", "Soybean": "सोयाबीन"},
        "states": {"Maharashtra": "महाराष्ट्र", "Goa": "गोवा"},
        "seasons": {"Kharif (Monsoon)": "खरीप", "Rabi (Winter)": "रब्बी", "Zaid (Summer)": "उन्हाळी"}
    },
    "ਪੰਜਾਬੀ (Punjabi)": {
        "title": "💧 ਸਮਾਰਟ ਸਿੰਚਾਈ", "lbl_state": "ਰਾਜ", "lbl_season": "ਮੌਸਮ", "lbl_soil": "ਮਿੱਟੀ ਦੀ ਨਮੀ", "lbl_temp": "ਤਾਪਮਾਨ", "lbl_humid": "ਨਮੀ", "lbl_crop": "ਫਸਲ", "btn_analyze": "ਵਿਸ਼ਲੇਸ਼ਣ", "alert_irrigate": "🚨 ਸਿੰਚਾਈ ਦੀ ਲੋੜ", "alert_safe": "✅ ਠੀਕ ਹੈ", "rec": "ਸਲਾਹ: ਪੰਪ ਚਲਾਓ", "modes": ["ਮੈਨੂਅਲ", "ਅਪਲੋਡ", "ਇਤਿਹਾਸ"], "hist_head": "📊 ਇਤਿਹਾਸ", "log_head": "💧 ਸਿੰਚਾਈ ਲੌਗ",
        "crops": {"Wheat": "ਕਣਕ", "Rice": "ਚਾਵਲ", "Cotton": "ਕਪਾਹ", "Sugarcane": "ਗੰਨਾ"},
        "states": {"Punjab": "ਪੰਜਾਬ", "Haryana": "ਹਰਿਆਣਾ"},
        "seasons": {"Kharif (Monsoon)": "ਸਾਉਣੀ", "Rabi (Winter)": "ਹਾੜੀ", "Zaid (Summer)": "ਜ਼ੈਦ"}
    },
    "ଓଡ଼ିଆ (Odia)": {
        "title": "💧 ସ୍ମାର୍ଟ ଜଳସେଚନ", "lbl_state": "ରାଜ୍ୟ", "lbl_season": "ଋତୁ", "lbl_soil": "ମାଟିର ଆର୍ଦ୍ରତା", "lbl_temp": "ତାପମାତ୍ରା", "lbl_humid": "ଆର୍ଦ୍ରତା", "lbl_crop": "ଫସଲ", "btn_analyze": "ବିଶ୍ଳେଷଣ", "alert_irrigate": "🚨 ଜଳସେଚନ ଆବଶ୍ୟକ", "alert_safe": "✅ ଠିକ୍ ଅଛି", "rec": "ପରାମର୍ଶ: ପମ୍ପ ଅନ୍ କରନ୍ତୁ", "modes": ["ମାନୁଆଲ", "ଅପଲୋଡ୍", "ଇତିହାସ"], "hist_head": "📊 ଇତିହାସ", "log_head": "💧 ଟ୍ରାକର୍",
        "crops": {"Rice": "ଚାଉଳ", "Pulse": "ଡାଲି", "Turmeric": "ହଳଦୀ"},
        "states": {"Odisha": "ଓଡ଼ିଶା"},
        "seasons": {"Kharif (Monsoon)": "ଖରିଫ", "Rabi (Winter)": "ରବି", "Zaid (Summer)": "ଗ୍ରୀଷ୍ମ"}
    },
    "অসমীয়া (Assamese)": {
        "title": "💧 স্মাৰ্ট জলসিঞ্চন", "lbl_state": "ৰাজ্য", "lbl_season": "ঋতু", "lbl_soil": "মাটিৰ আৰ্দ্ৰতা", "lbl_temp": "উষ্ণতা", "lbl_humid": "আৰ্দ্ৰতা", "lbl_crop": "শস্য", "btn_analyze": "বিশ্লেষণ", "alert_irrigate": "🚨 জলসিঞ্চনৰ প্ৰয়োজন", "alert_safe": "✅ ঠিক আছে", "rec": "পৰামৰ্শ: পাম্প চলাওক", "modes": ["মেনুৱেল", "আপলোড", "ইতিহাস"], "hist_head": "📊 ইতিহাস", "log_head": "💧 লগ্",
        "crops": {"Tea": "চাহ", "Rice": "চাউল", "Jute": "মৰাপাত"},
        "states": {"Assam": "অসম"},
        "seasons": {"Kharif (Monsoon)": "খাৰিফ", "Rabi (Winter)": "ৰবি", "Zaid (Summer)": "গ্ৰীষ্ম"}
    },
    "اردو (Urdu)": {
        "title": "💧 اسمارٹ آبپاشی", "lbl_state": "ریاست", "lbl_season": "موسم", "lbl_soil": "مٹی کی نمی", "lbl_temp": "درجہ حرارت", "lbl_humid": "نمی", "lbl_crop": "فصل", "btn_analyze": "تجزیہ", "alert_irrigate": "🚨 آبپاشی کی ضرورت", "alert_safe": "✅ بہترین", "rec": "تجویز: پمپ چلائیں", "modes": ["دستی", "اپ لوڈ", "تاریخ"], "hist_head": "📊 تاریخ", "log_head": "💧 ٹریکر",
        "crops": {"Wheat": "گندم", "Rice": "چاول", "Cotton": "کپاس"},
        "states": {"Jammu & Kashmir": "جموں و کشمیر"},
        "seasons": {"Kharif (Monsoon)": "خریف", "Rabi (Winter)": "ربیع", "Zaid (Summer)": "زید"}
    },
    "संस्कृतम् (Sanskrit)": {
        "title": "💧 चतुर-सेचनम्", "lbl_state": "राज्यम्", "lbl_season": "ऋतुः", "lbl_soil": "मृदा-आर्द्रता", "lbl_temp": "तापमानम्", "lbl_humid": "आर्द्रता", "lbl_crop": "सस्यम्", "btn_analyze": "विश्लेषणं कुरु", "alert_irrigate": "🚨 सेचनम् आवश्यकम्", "alert_safe": "✅ उत्तमम्", "rec": "परामर्शः: जलयन्त्रं चालयतु", "modes": ["हस्तेन", "सञ्चिका", "इतिहास"], "hist_head": "📊 इतिहास", "log_head": "💧 सेचन-वृत्तम्",
        "crops": {"Rice": "तण्डुलः", "Wheat": "गोधूमः", "Sugarcane": "इक्षुः"},
        "states": {"Uttarakhand": "उत्तराखण्ड", "Himachal Pradesh": "हिमाचल प्रदेशः"},
        "seasons": {"Kharif (Monsoon)": "वर्षा", "Rabi (Winter)": "हेमन्त", "Zaid (Summer)": "ग्रीष्म"}
    },
    "नेपाली (Nepali)": {
        "title": "💧 स्मार्ट सिँचाइ", "lbl_state": "राज्य", "lbl_season": "मौसम", "lbl_soil": "माटोको चिस्यान", "lbl_temp": "तापक्रम", "lbl_humid": "आर्द्रता", "lbl_crop": "बाली", "btn_analyze": "विश्लेषण", "alert_irrigate": "🚨 सिँचाइ आवश्यक", "alert_safe": "✅ ठीक छ", "rec": "सुझाव: पम्प चलाउनुहोस्", "modes": ["म्यानुअल", "अपलोड", "इतिहास"], "hist_head": "📊 इतिहास", "log_head": "💧 सिँचाइ लग",
        "crops": {"Rice": "धान", "Corn": "मकै", "Ginger": "अदुवा"},
        "states": {"Sikkim": "सिक्किम"},
        "seasons": {"Kharif (Monsoon)": "वर्षा", "Rabi (Winter)": "हिउँद", "Zaid (Summer)": "गर्मी"}
    },
    "कोङ्कणी (Konkani)": {
        "title": "💧 स्मार्ट शिंपणे", "lbl_state": "राज्य", "lbl_season": "मोसम", "lbl_soil": "मातयेची ओलसाण", "lbl_temp": "तापमान", "lbl_humid": "ओलसाण", "lbl_crop": "पीक", "btn_analyze": "विश्र्लेषण", "alert_irrigate": "🚨 उदक जाय", "alert_safe": "✅ बरे आसा", "rec": "सल्लो: पंप चालू करा", "modes": ["मॅन्युअल", "अपलोड", "इतिहास"], "hist_head": "📊 इतिहास", "log_head": "💧 शिंपणे लग",
        "crops": {"Coconut": "नाल्ल", "Rice": "तांदूळ", "Cashew": "काजू"},
        "states": {"Goa": "गोंय"},
        "seasons": {"Kharif (Monsoon)": "पावसाळी", "Rabi (Winter)": "शिवाळी", "Zaid (Summer)": "गिम्हाळी"}
    },
    "মণিপুরী (Manipuri)": {
        "title": "💧 স্মার্ট ইরিগেশন", "lbl_state": "রাজ্য", "lbl_season": "ঋতু", "lbl_soil": "লৈবাক্কী ঈশিং", "lbl_temp": "অশা-অইং", "lbl_humid": "ঈশিং", "lbl_crop": "ফসল", "btn_analyze": "এনালাইজ", "alert_irrigate": "🚨 ঈশিং থাইগদবনি", "alert_safe": "✅ ফৈ", "rec": "পাম্প অন তৌ", "modes": ["মেনুয়েল", "আপলোড", "হিস্ট্রি"], "hist_head": "📊 হিস্ট্রি", "log_head": "💧 ইরিগেশন লগ",
        "crops": {"Rice": "চেং", "Corn": "চুজাক"},
        "states": {"Manipur": "মণিপুর"},
        "seasons": {"Kharif (Monsoon)": "কালেন", "Rabi (Winter)": "নিঙথাম", "Zaid (Summer)": "ইয়েল"}
    },
    "सिन्धी (Sindhi)": {
        "title": "💧 سمارٽ آبپاشي", "lbl_state": "راڄ", "lbl_season": "موسم", "lbl_soil": "مٽي جي نمي", "lbl_temp": "گرمي پد", "lbl_humid": "نمي", "lbl_crop": "فصل", "btn_analyze": "تجزيو", "alert_irrigate": "🚨 پاڻي جي ضرورت", "alert_safe": "✅ ٺيڪ آهي", "rec": "صلاح: پمپ هلايو", "modes": ["دستي", "اپ لوڊ", "تاريخ"], "hist_head": "📊 تاريخ", "log_head": "💧 آبپاشي لاگ",
        "crops": {"Wheat": "ڪڻڪ", "Rice": "چاول", "Cotton": "ڦٽي"},
        "states": {"Gujarat": "گجرات"},
        "seasons": {"Kharif (Monsoon)": "خريف", "Rabi (Winter)": "ربي", "Zaid (Summer)": "زيد"}
    }
}

def get_txt(lang, key, subkey=None):
    base = TRANSLATIONS.get(lang, TRANSLATIONS["English"])
    if subkey:
        cat = base.get(key, TRANSLATIONS["English"].get(key, {}))
        val = cat.get(subkey, TRANSLATIONS["English"][key].get(subkey, subkey))
        return val
    return base.get(key, TRANSLATIONS["English"].get(key, key))

# ==========================================
# 5. USER INTERFACE
# ==========================================
st.sidebar.header("Language / भाषा")
lang_options = list(TRANSLATIONS.keys())
selected_lang = st.sidebar.selectbox("", lang_options)

st.sidebar.title("🌱 Ecoverse")

# Mode Selection
modes = get_txt(selected_lang, "modes")
mode = st.sidebar.radio("Menu", modes)

# --- MODE 1: MANUAL INPUT (PREDICTION) ---
if mode == modes[0]:
    st.header(get_txt(selected_lang, "title"))
    
    # State & Season Selection (Translated)
    eng_states = sorted(list(STATE_CROP_MAP.keys()))
    disp_states = [get_txt(selected_lang, "states", s) for s in eng_states]
    selected_disp_state = st.sidebar.selectbox(get_txt(selected_lang, "lbl_state"), disp_states)
    
    # Map back to English for Logic
    # Safety Check: If index error happens (rare), default to 0
    try:
        idx = disp_states.index(selected_disp_state)
    except:
        idx = 0
    selected_eng_state = eng_states[idx]

    disp_seasons = [get_txt(selected_lang, "seasons", s) for s in SEASONS]
    st.sidebar.selectbox(get_txt(selected_lang, "lbl_season"), disp_seasons)
    
    col1, col2, col3 = st.columns(3)
    with col1: soil = st.slider(get_txt(selected_lang, "lbl_soil"), 0, 100, 40)
    with col2: temp = st.slider(get_txt(selected_lang, "lbl_temp"), 10, 50, 30)
    with col3: humid = st.slider(get_txt(selected_lang, "lbl_humid"), 0, 100, 50)
    
    # Filter Crops by State
    state_crops_eng = STATE_CROP_MAP.get(selected_eng_state, ["Wheat", "Rice"])
    state_crops_disp = [get_txt(selected_lang, "crops", c) for c in state_crops_eng]
    selected_disp_crop = st.selectbox(get_txt(selected_lang, "lbl_crop"), state_crops_disp)
    
    # Find ID
    try:
        idx_crop = state_crops_disp.index(selected_disp_crop)
        orig_crop_name = state_crops_eng[idx_crop]
    except:
        orig_crop_name = "Wheat" # Fallback

    crop_id = CROP_MAP.get(orig_crop_name, 0)
    
    if st.button(get_txt(selected_lang, "btn_analyze")):
        # ML Prediction
        pred = model.predict([[soil, temp, humid, crop_id]])[0]
        
        if pred == 1:
            msg = get_txt(selected_lang, "alert_irrigate")
            st.toast(msg, icon="🚨")
            st.error(f"**{msg}**")
            st.info(get_txt(selected_lang, "rec"))
        else:
            msg = get_txt(selected_lang, "alert_safe")
            st.toast(msg, icon="✅")
            st.success(f"**{msg}**")

# --- MODE 2: BATCH UPLOAD ---
elif mode == modes[1]:
    st.header("📂 " + modes[1])
    uploaded_file = st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"])
    if uploaded_file and st.button(get_txt(selected_lang, "btn_analyze")):
        try:
            if uploaded_file.name.endswith('.csv'): df = pd.read_csv(uploaded_file)
            else: df = pd.read_excel(uploaded_file)
            
            results = []
            for _, row in df.iterrows():
                row = {k.lower(): v for k, v in row.items()} # Case insensitive
                s, t, h = row.get('soil', 40), row.get('temp', 30), row.get('humidity', 50)
                c_name = row.get('crop', 'Wheat')
                
                # Robust ID Lookup
                c_id = CROP_MAP.get(c_name, 0)
                
                pred = model.predict([[s, t, h, c_id]])[0]
                results.append(get_txt(selected_lang, "alert_irrigate") if pred == 1 else get_txt(selected_lang, "alert_safe"))
            
            df['AI Status'] = results
            st.dataframe(df)
            st.success("✅ Analysis Complete")
        except Exception as e: st.error(f"Error: {e}")

# --- MODE 3: HISTORY TRACKER (NEW!) ---
elif mode == modes[2]:
    st.header(get_txt(selected_lang, "hist_head"))
    
    # 1. Generate Fake History Data
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    hist_soil = np.random.uniform(20, 80, 30)
    hist_temp = np.random.uniform(25, 40, 30)
    hist_humid = np.random.uniform(40, 90, 30)
    
    # Irrigation happened if Soil < 35%
    actions = ["✅ Irrigated" if s < 35 else "-" for s in hist_soil]
    
    hist_df = pd.DataFrame({
        "Date": dates.strftime("%Y-%m-%d"),
        "Soil Moisture (%)": hist_soil,
        "Temperature (°C)": hist_temp,
        "Humidity (%)": hist_humid,
        "Status": actions
    })
    
    # 2. Charts (Tracker for Env Conditions)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Soil Moisture Trend")
        st.line_chart(hist_df.set_index("Date")["Soil Moisture (%)"])
    with col2:
        st.subheader("Temperature Trend")
        st.line_chart(hist_df.set_index("Date")["Temperature (°C)"])
        
    st.subheader("Humidity Trend")
    st.line_chart(hist_df.set_index("Date")["Humidity (%)"])
    
    # 3. Irrigation Log Tracker
    st.markdown("---")
    st.subheader(get_txt(selected_lang, "log_head"))
    
    log_df = hist_df[hist_df["Status"] == "✅ Irrigated"]
    if not log_df.empty:
        st.dataframe(log_df[["Date", "Soil Moisture (%)", "Status"]], use_container_width=True)
    else:
        st.info("No irrigation required in the last 30 days.")
