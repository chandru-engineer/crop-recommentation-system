from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('admin/', admin.site.urls, name='admin'),
    path('', views.home, name='home-crop'),
    path('predict/', views.predict, name='predict'),
    path('analysis/', views.analysis, name='analysis'),
    path('usecase/', views.usecase, name='usecase'),
]
