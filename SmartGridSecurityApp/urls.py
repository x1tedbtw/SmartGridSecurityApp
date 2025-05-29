from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name="home"),
    path('dataset/', views.dataset, name="dataset"),
    path('training/', views.training, name="training"),
    path('detection/', views.detection, name="detection"),
]
