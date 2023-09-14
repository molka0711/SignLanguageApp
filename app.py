import json
from chat import get_response  # Import the chatbot logic function
from django.http import JsonResponse  # Import JsonResponse for sending JSON responses


def chatbot_response(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)  # Parse the JSON data from the request body
            msg = data.get("message")  # Extract the 'message' field from the JSON data

            if msg is None:
                return JsonResponse(
                    {"error": "Missing 'message' field in JSON data"}, status=400
                )

            bot_response = get_response(msg)  # Call the chatbot logic function
            response_data = {"answer": bot_response}
            return JsonResponse(response_data)

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
