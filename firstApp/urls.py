from django.urls import path
from . import views
# from django.conf.urls import static
# from django.conf import settings

urlpatterns = [
    path('', views.index, name='index'),
    path('predictImage', views.predictImage, name='predictImage'),
    path('viewDataBase', views.viewDataBase, name='viewDataBase'),
]

