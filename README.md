# naver_news
news article classification and recommend article using comments

### crawling naver news articles
- https://github.com/radajin/naver_news/blob/master/naver_news_crawling.ipynb
- function : auto_cralwing(download_date)

### news classification 
- https://github.com/radajin/naver_news/blob/master/news_classification.ipynb
- Using TfidfVectorizer and MultinomialNB
- category : 정치, 경제, 사회, 생활/문화, 세계, IT/과학
- max precision : 90% ( the number of training data : 103910 )

### recomend news article
- https://github.com/radajin/naver_news/blob/master/recomend_article_1.ipynb
- Using TfidfVectorizer and cosin distance
- https://github.com/radajin/naver_news/blob/master/recomend_article_2.ipynb
- Using Evalidation MAE(mean absolute error)

### spam filtering
- https://github.com/radajin/naver_news/blob/master/spam_filtering.ipynb
- Using TfidfVectorizer and MultinomialNB
- precision: 89% ( the number of tarining data : 58075 )

