from django.http import HttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
import json
import base64
from digits_recognition.process import recNum



def index(request):
	return render(request, 'index.html')

@csrf_exempt
def process(request):
	if (request.method == "POST") and (request.POST.get('id') == "1"):
		imgStr = request.POST.get('txt')
		#imgStr.replace(" ", "+")
		imgStr = base64.b64decode(imgStr)
		#识别
		res = str(recNum(imgStr))
		return HttpResponse(json.dumps({"status": 1, "result": res}))

