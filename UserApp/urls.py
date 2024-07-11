from django.urls import path
from UserApp import views
urlpatterns=[
    path('', views.index),
    path('index', views.index),
    path('login', views.login),
    path('logaction', views.loglction),
    path('userhome', views.userhome),
    path('register', views.register),
    path('regaction', views.regaction),
    path('viewprofile', views.viewprofile),
    path('uploadImage', views.uploadImage),
    path('imageAction',views.imageAction),
    path('brainstrokepredict', views.brainstrokepredict),
    ]
