# import os
# import json
# import shutil

# # 경로 세팅 (아래 경로를 본인 환경에 맞게 조정)
# json_root = r'C:\junho\web\data\skin_condition\open_data\data\Training\labeled_data\여드름\정면'
# img_root  = r'C:\junho\web\data\skin_condition\open_data\data\Training\source_data\여드름\정면'
# output_root = r'C:\junho\web\data\skin_dataset_split\train'  # 결과물 저장 폴더(예시)

# # 여드름 레벨 기준 직접 조정 가능
# def get_level(n):
#     if n == 0:
#         return '정상'
#     elif n <= 3:
#         return '여드름1'
#     elif n <= 7:
#         return '여드름2'
#     elif n <= 12:
#         return '여드름3'
#     elif n <= 20:
#         return '여드름4'
#     else:
#         return '여드름5'

# os.makedirs(output_root, exist_ok=True)

# for json_name in os.listdir(json_root):
#     if not json_name.endswith('.json'):
#         continue
#     json_path = os.path.join(json_root, json_name)
#     with open(json_path, 'r', encoding='utf-8') as f:
#         data = json.load(f)
#     annotations = data.get('annotations', [])
#     if not annotations:
#         level = '정상'
#         img_file = None
#     else:
#         lesions = annotations[0].get('bbox', {}).get('lesions', [])
#         level = get_level(len(lesions))
#         img_file = annotations[0]['photograph']['file_path'].split('/')[-1]
#     if img_file is None:
#         continue
#     src_img_path = os.path.join(img_root, img_file)
#     out_dir = os.path.join(output_root, level)
#     os.makedirs(out_dir, exist_ok=True)
#     dst_img_path = os.path.join(out_dir, img_file)
#     if os.path.exists(src_img_path):
#         shutil.copy2(src_img_path, dst_img_path)
#         print(f"이미지 {img_file} -> {level}로 복사됨")
#     else:
#         print(f"이미지 파일 없음: {src_img_path}")

# print("폴더 분류 작업 완료")













############################################################################################33

# split_val_by_level.py
import os
import json
import shutil

# 원본 경로(검증 데이터)
JSON_DIR = r"C:\junho\web\data\skin_condition\open_data\data\Validation\labeled_data\여드름\정면"
IMG_DIR = r"C:\junho\web\data\skin_condition\open_data\data\Validation\source_data\여드름\정면"

# 결과(분리) 경로
OUT_ROOT = r"C:\junho\web\data\skin_dataset_split\val"
LEVELS = ["여드름1", "여드름2", "여드름3", "여드름4", "여드름5"]

# 레벨 기준: json의 lesions 개수로 분류
def level_by_count(n):
    if n <= 2: return 0  # 여드름1
    elif n <= 5: return 1
    elif n <= 8: return 2
    elif n <= 12: return 3
    else: return 4

os.makedirs(OUT_ROOT, exist_ok=True)
for lv in LEVELS:
    os.makedirs(os.path.join(OUT_ROOT, lv), exist_ok=True)

for file in os.listdir(JSON_DIR):
    if not file.endswith(".json"): continue
    json_path = os.path.join(JSON_DIR, file)
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
        try:
            lesions = data["annotations"][0]["bbox"]["lesions"]
            n_lesions = len(lesions)
            level_idx = level_by_count(n_lesions)
        except Exception as e:
            print(f"json 오류: {file} - {e}")
            continue
        img_name = data["annotations"][0]["photograph"]["file_path"].split("/")[-1]
        img_path = os.path.join(IMG_DIR, img_name)
        if os.path.exists(img_path):
            shutil.copy(img_path, os.path.join(OUT_ROOT, LEVELS[level_idx], img_name))
        else:
            print(f"이미지 누락: {img_path}")

print("분류 완료.")
