"""DjangoApp URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.views.generic import TemplateView
from . import views
from . import views1
from . import input_chat

urlpatterns = [
    path("", TemplateView.as_view(template_name="index.html"), name="home"),
    path("index.html", TemplateView.as_view(template_name="index.html"), name="home2"),
    path("about.html", TemplateView.as_view(template_name="about.html"), name="about"),
    path(
        "contact.html",
        TemplateView.as_view(template_name="contact.html"),
        name="contact",
    ),
    path(
        "courses.html",
        TemplateView.as_view(template_name="courses.html"),
        name="courses",
    ),
    path("team.html", TemplateView.as_view(template_name="team.html"), name="team"),
    path(
        "testimonial.html",
        TemplateView.as_view(template_name="testimonial.html"),
        name="testimonial",
    ),
    path(
        "404.html",
        TemplateView.as_view(template_name="404.html"),
        name="404",
    ),
    path("chatbot/", views.chatbot_endpoints, name="chatbot_endpoint"),
    path(
        "get_gesture_result/",
        views1.gesture_recognition_stream,
        name="get_gesture_result",
    ),
    path(
        "gesture_translator.html",
        TemplateView.as_view(template_name="gesture_translator.html"),
        name="gesture_translator",
    ),
    path("execute-processCam/", input_chat.execute_processCam, name="execute_processCam"),
    path("send_gesture/", input_chat.send_gesture_to_url, name="send_gesture_to_url"),
        path(
        "tools.html",TemplateView.as_view(template_name="tools.html"), name="tools"),
   path(
        "shapes.html",TemplateView.as_view(template_name="shapes.html"), name="shapes"),
    path("allsigns.html",TemplateView.as_view(template_name="allsigns.html"),name="allsigns"), 
]
