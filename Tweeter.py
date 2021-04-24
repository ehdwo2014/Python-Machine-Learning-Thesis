# 예제 코드1: 검색 키워드를 활용한 트위터 게시글 단순 검색

import tweepy

# 트위터 Consumer Key (API Key)
consumer_key = "J3tuHAcANXUdPS6GGAa9OX91D"
# 트위터 Consumer Secret (API Secret)
consumer_secret = "Z16v0wY6Exp3j4f5HdrZNWeTRk5wKeVWA0sR7wTRqMCyUuZ4Sj"

# 1차 인증: 개인 앱 정보
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)

# 트위터 Access Token
access_token = "1171186180341346305-M4OXS4d9EpAFjxMuIGDoGRvQM0HVht"
# 트위터 Access Token Secret
access_token_secret= "3SVuJeoEIB1xWQHHJMokLcCCF9mI7UbYwTs4QsaSkaEKg"

# 2차 인증: 토큰 정보
auth.set_access_token(access_token, access_token_secret)

# 3. twitter API 생성
api = tweepy.API(auth)

keyword = "leoni";     # OR 로 검색어 묶어줌, 검색어 5개까지 가능 (반드시 OR 는 대문자로 작성해야 함)
location = "%s,%s,%s" % ("37.00", "100", "1000km")  # 트윗 지역 기준 검색, 대한민국 중심 좌표, 반지름

# tweepy.Cursor 메서드를 사용하면, pagination 이 자동 관리되어 100개 이상의 검색 결과를 가져올 수 있음
# 모든 트위터 데이터가 검색되는 것은 아님(트위터사에서 이를 명시하고 있음)
cursor = tweepy.Cursor(api.search,
                       q=keyword,
                       since='2019-09-06',  # 2015-01-01 이후에 작성된 트윗들로 가져옴
                       count=100,           # 페이지당 반환할 트위터 수 최대 100
                       include_entities=True)
for i, tweet in enumerate(cursor.items()):
    print("{}: {}".format(i, tweet.text))