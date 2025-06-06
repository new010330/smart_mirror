from icrawler.builtin import GoogleImageCrawler
import os

# 퍼스널컬러 시즌별 대표 키워드
season_keywords = {
    "spring": [
        "아이유 봄 퍼스널컬러", "장원영 봄 웜톤", "엔믹스 해원 봄 웜톤", "에스파 카리나 봄 웜톤",
        "spring warm tone face", "봄웜 여자 얼굴", "봄웜 연예인", "봄웜 아이돌"
    ],
    "summer": [
        "태연 여름 퍼스널컬러", "김채원 여름 쿨톤", "트와이스 다현 여름 쿨톤",
        "summer cool tone face", "여름쿨 여자 얼굴", "여름쿨 연예인", "여름쿨 아이돌"
    ],
    "autumn": [
        "이효리 가을 퍼스널컬러", "선미 가을 웜톤", "히토미 가을 웜톤",
        "autumn warm tone face", "가을웜 여자 얼굴", "가을웜 연예인", "가을웜 아이돌"
    ],
    "winter": [
        "카리나 겨울 퍼스널컬러", "제니 겨울 쿨톤", "윈터 겨울 쿨톤",
        "winter cool tone face", "겨울쿨 여자 얼굴", "겨울쿨 연예인", "겨울쿨 아이돌"
    ]
}

# 저장 경로 설정
base_dir = "data/personal_color"

# 키워드별 이미지 수집
for season, keywords in season_keywords.items():
    save_dir = os.path.join(base_dir, season)
    os.makedirs(save_dir, exist_ok=True)

    for idx, keyword in enumerate(keywords):
        print(f"[{season.upper()}] ({idx+1}/{len(keywords)}): '{keyword}' 다운로드 중...")
        crawler = GoogleImageCrawler(storage={"root_dir": save_dir})
        crawler.crawl(keyword=keyword, max_num=10)
