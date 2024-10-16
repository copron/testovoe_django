from django.shortcuts import render
import joblib
import os
import gdown  # Убедитесь, что эта библиотека установлена
from django.conf import settings

# Определяем путь к директории с моделями
models_dir = os.path.join(settings.BASE_DIR, 'review_classifier', 'models')

os.makedirs(models_dir, exist_ok=True)

voting_clf_url = 'https://drive.google.com/uc?id=1JbW5yl70JaohTxHpRJ50rBit_YQpMu31'
output_voting_clf = os.path.join(models_dir, 'voting_classifier_model.joblib')

if not os.path.exists(output_voting_clf):
    gdown.download(voting_clf_url, output_voting_clf, quiet=False)

sentiment_clf = joblib.load(os.path.join(models_dir, 'sgd_classifier_model.joblib'))
voting_clf = joblib.load(output_voting_clf)
tfidf_vectorizer = joblib.load(os.path.join(models_dir, 'tfidf_vectorizer.joblib'))

def predict_status(request):
    sentiment_prediction = None
    rating_prediction = None
    error_message = None  
    
    if request.method == 'POST':
        review_text = request.POST.get('review')

        # Проверка, что отзыв не пустой
        if review_text:
            # Преобразуем текст отзыва в нужный формат
            review_tfidf = tfidf_vectorizer.transform([review_text])
            
            # Предсказание настроения
            sentiment_prediction = sentiment_clf.predict(review_tfidf)[0]
            
            # Предсказание рейтинга
            rating_prediction = voting_clf.predict([review_text])[0]
        else:
            error_message = "Пожалуйста, введите текст отзыва."  

    return render(request, 'review_classifier/predict.html', {
        'sentiment_prediction': sentiment_prediction,
        'rating_prediction': rating_prediction,
        'error_message': error_message  
    })
