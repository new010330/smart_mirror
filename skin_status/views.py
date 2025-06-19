# skin_status/views.py
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser
from PIL import Image
from .predict_skin_status import predict_skin_status

class SkinStatusView(APIView):
    parser_classes = [MultiPartParser]

    def post(self, request):
        image = request.FILES.get("file")
        if not image:
            return Response({"error": "이미지를 보내주세요"}, status=400)
        img = Image.open(image)
        result = predict_skin_status(img)
        return Response(result)

