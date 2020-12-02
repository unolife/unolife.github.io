---
title: "추천 시스템 탐구"
date: 2020-12-02 13:50:00 -0400
categories: Recommendation
---
내술노트 개발하다가 추천 시스템 공부 해야해서 만든 포스트

추천 시스템을 공부하려면 우선 "유저"와 "콘텐츠" 데이터가 있어야 추천을 할 수 있을거 같은데, 그 데이터를 만들 수는 없으니 캐글을 뒤지기 시작했다.<br>
그러다가 <a href="https://www.kaggle.com/kanncaa1/recommendation-systems-tutorial">Recommendation Systems Tutorial notebook</a>을 발견하고, 여기서부터 시작하기로 했다.

추천시스템을 검색하면 가장 기초적인? 방법이 collborative filter인것 같다. 해당 노트북에서는 프로그래밍을 위해서 네 단계로 나눠났지만,<br>
간단히 말해서 나랑 가장 비슷한 성향의 유저를 찾고, 그 유저가 본 영화중에 내가 아직 안 본 영화를 추천하는 방식이다.<br>
단점도 언급되어 있는데, 유저기반이다보니 유저가 늘어나면 computing 부하가 걸리고, 유저의 취향 변화에 빠르게 대응하지 못한다고 한다.

그래서 다른 방법으로 item based CF(Collaborative Filter)방식을 사용할 수도 있다.<br>
여기서는 유저x영화 매트릭스(테이블,행렬)에서 column(영화)간의 유사도를 계산해서, 비슷한 영화를 찾는다. <br>

- 예제코드<br>
  ```python
    import pandas as pd
    movie = pd.read_csv('./movie.csv')
    print(movie.columns)
    
    movie = movie.loc[:,["movieId","title"]]
    
    rating = pd.read_csv("./rating.csv")
    print(rating.columns)
    
    rating = rating.loc[:,["userId", "movieId", "rating"]]
    
    data = pd.merge(movie, rating)
    print(data.shape, data.head(10))
    # 영화별로 유저가 남긴 평가가 있는거라서 행이 엄청 많아진다.(2천만행)
    data = data.iloc[:1000000,:]
    
    pivot_table = data.pivot_table(index = ["userId"], columns = ["title"], values="rating")
    # userId를 인덱스 삼았으니 중복분이 제거되고, 유저별로 집계됨, columns에 title을 넣어서 영화를 컬럼으로 만들고, value에 rating을 넣어서 위에서 원하던 유저x영화 매트릭스가 만들어짐
    # 영화 하나를 고르고, 그것과 유사도를 측정해서 유사한 영화들을 찾음, 그걸 추천
    movie_watched = pivot_table["Bad Boys (1995)"]
    similarity_with_other_movies = pivot_table.corrwith(movie_watched)
    similarity_wit_other_movies = similarity_with_other_movies.sort_valus(ascending=False)
    similarity_with_other_movies.head() # notebook에서 할경우
  ```
corrwith은 사용해본적이 없어서 찾아봤는데, <a href="https://rfriend.tistory.com/405">블로그</a>를 통해 새로 알게된 사실은 <br> corr은 두 변수간의 상관관계이고, corrwith은 하나의 변수와 나머지 모든 변수간의 상관관계를 측정한다고 한다.

--------------------------------------------------------------------------------------<br>

우선 기본적인거는 살펴 봤으니, 좀 더 전문적인 버전을 찾아봤다. <a href="https://www.kaggle.com/rajmehra03/cf-based-recsys-by-low-rank-matrix-factorization">CF Based RecSys by Low Rank Matrix Factorization</a>이라는 노트북을 찾았는데,<br> 
문제는 제목부터 어렵다... Low Rank Matrix Factorizaion, 사전을 찾아봤더니 factorizaiont은 인수분해인데 문제는 앞에 rank가 붙는순간 이게 선형대수로 넘어간다...<br>
여기저기 찾아봤더니 Andrew Ng교수님의 강의중에 Recommender System강의 내용을 다루는 블로그들이 있었다.(3학년때 잠깐 들었던 기억이 있는데... 다시 봐야겠다 ㅠㅠ)<br>
우선 <a href="https://www.youtube.com/playlist?list=PLLssT5z_DsK-h9vYZkQkYNWcItqhlRJLN">전체 강의</a> 중에 Recommender System을 다루는 16강만 들어야겠다.

노트북에 나온 코드부터 살펴보는걸 먼저 해야겠다.
- 라이브러리 import
  ```python
  # Ignore the warnings
  import warnings
  warnings.filterwarnings('always')
  warnings.filterwarnings('ignore')
  
  # data visualisation and manipulation
  import numpy as np
  import pandas as pd
  import matplotlib.pyplot as plt
  from matplotlib import style
  import seaborn as sns
  
  # configure
  # sets matplotlib to incline and displays graphs below the corresponding cell.
  # just for ipnb or colab
  %matplotlib inline
  style.use('fivethirtyeight')
  sns.set(style='whitegrid',color_codes=True) 
  
  # model selection
  from sklearn.model_selection import train_test_split
  from sklearn.model_selection import KFold
  from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, roc_auc_score
  from sklearn.metrics import mean_absolute_error
  from sklearn.model_selection import GridSearchCV
  from sklearn.preprocesing import LabelEncoder
  
  # preprocess
  keras.preprocessing.image import ImageDataGenerator
  
  # dl libraries
  import keras
  
