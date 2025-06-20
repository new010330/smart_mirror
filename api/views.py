from rest_framework.response import Response
from rest_framework.views import APIView

class PredictColorView(APIView):
    def post(self, request):
        return Response({"detail": "이 API는 사용되지 않음. /api/analyze/로 요청하세요."}, status=410)
