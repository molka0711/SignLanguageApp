from django.shortcuts import render
import json
import subprocess
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from chat import get_response

def index(request):
    return render(request, 'index.html')

def about(request):
    return render(request, 'about.html')

@csrf_exempt
def chatbot_endpoints(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            message = data.get('message')

            # Call your chatbot logic here and get the response
            chatbot_response = get_response(message)

            # Send chatbot response back to the frontend
            return JsonResponse({'answer': chatbot_response})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
