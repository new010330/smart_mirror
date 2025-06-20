from pathlib import Path
import os

# BASE 디렉토리
BASE_DIR = Path(__file__).resolve().parent.parent

# 보안 관련 키 (배포 시엔 환경변수로 분리 권장)
SECRET_KEY = 'django-insecure-$-l1%1#^qglv-+hlwgqz*@n375)z&pquv1h2om-bin6a%sldpk'

DEBUG = True
ALLOWED_HOSTS = []  # 개발 시엔 빈 배열, 배포 시엔 도메인 추가

# 앱 등록
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
   

    'rest_framework',
    'corsheaders',
    'api',
    'skin_status.face_analysis',
]

# 미들웨어
MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

# URL 설정
ROOT_URLCONF = 'personal_color_server.urls'

# 템플릿 설정
TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],  # 템플릿 디렉토리 추가 시 여기에 넣으면 됨
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',  # ✅ 추가 추천
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

# WSGI
WSGI_APPLICATION = 'personal_color_server.wsgi.application'

# DB 설정 (기본 SQLite)
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# 비밀번호 검증
AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

# 언어 및 시간대
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

# 정적파일
STATIC_URL = 'static/'

# ✅ 미디어 업로드 설정
MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')

# 기본 PK 필드 타입
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# ✅ CORS 설정 (프론트와 연동 위해 개발용 전체 허용)
CORS_ALLOW_ALL_ORIGINS = True
