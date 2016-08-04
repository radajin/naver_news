# naver_news
news article classification and recommend article using comments

### crawling naver news articles
- https://github.com/radajin/naver_news/blob/master/naver_news_crawling.ipynb
- function : auto_cralwing(download_date)

### news classification 
- https://github.com/radajin/naver_news/blob/master/news_classification.ipynb
- Using TfidfVectorizer and MultinomialNB
- category : 정치, 경제, 사회, 생활/문화, 세계, IT/과학
- max precision : 88% ( the number of training data : 2327 )

### recomend news article
- https://github.com/radajin/naver_news/blob/master/recommend_article.ipynb
- Using Evalidation MAE(mean absolute error)

### flask web service
- http://radajin.ml
