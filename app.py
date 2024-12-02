from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import spacy
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = FastAPI()

# Инициализация NLP
nlp = spacy.load("ru_core_news_md")
stop_words = set(stopwords.words("russian"))

# Индекс из текстов с отзывами
documents = [
    "Этот телефон работает очень хорошо и быстро.",
    "Качество наушников оставляет желать лучшего.",
    "Я очень доволен покупкой, товар отличный!",
    "Клавиатура удобная, но батарея садится быстро.",
    "Монитор имеет отличное качество изображения.",
    "Наушники удобно сидят, но звук слегка приглушённый.",
    "Доставка была быстрой, товар приехал в целости и сохранности.",
    "Я ожидал большего от этой мыши, она не так удобна, как казалось.",
    "Эта модель ноутбука имеет хорошее соотношение цены и качества.",
    "Пылесос тихий, но иногда оставляет немного мусора.",
    "Смарт-часы выглядят стильно и имеют долгий срок службы батареи.",
    "Планшет хорошо справляется с задачами, но иногда слегка тормозит.",
    "Фен мощный, но кнопки управления расположены неудобно.",
    "Колонка выдает чистый звук даже на максимальной громкости.",
    "Принтер печатает быстро, но иногда зажевывает бумагу.",
    "Камера делает отличные снимки даже в условиях плохого освещения.",
    "Стиральная машина тихая, но отжим оставляет одежду слегка влажной.",
    "Чехол для телефона удобен и хорошо защищает от падений.",
    "Процессор мощный, но иногда сильно греется под нагрузкой.",
    "Микроволновка разогревает равномерно и быстро.",
    "Электросамокат быстрый, но запас хода меньше, чем указано.",
    "Кофемашина варит вкусный кофе, но требует частой очистки.",
    "Электрическая зубная щетка отлично очищает зубы и долго держит заряд.",
    "Видеокарта справляется с играми на высоких настройках без проблем.",
    "Блок питания шумный, но обеспечивает стабильное напряжение."
]




# Инициализация TF-IDF
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)


class TextRequest(BaseModel):
    text: str


class SearchRequest(BaseModel):
    query: str


@app.post("/nlp_pipeline/", response_model=List[str])
async def nlp_pipeline(request: TextRequest):
    """Обработка текста через NLP пайплайн."""
    doc = nlp(request.text.lower())
    tokens = [token.lemma_ for token in doc if token.is_alpha and token.lemma_ not in stop_words]
    return tokens


@app.post("/search/", response_model=List[str])
async def search_text(request: SearchRequest):
    """Поиск релевантных текстов по запросу."""
    query = request.query.lower()
    query_tfidf = tfidf_vectorizer.transform([query])
    similarity_scores = cosine_similarity(query_tfidf, tfidf_matrix)

    top_indices = np.argsort(similarity_scores.flatten())[-3:][::-1]
    top_documents = [documents[idx] for idx in top_indices]

    return top_documents



