from django.contrib import admin
from django.urls import path, re_path
from django.views.generic import RedirectView
from . import views 

urlpatterns = [
    path('admin/', admin.site.urls),
    path('predict/', views.predict_status, name='predict_status'),
    re_path(r'^$', RedirectView.as_view(url='/predict/', permanent=False)),  # Перенаправление с пустого пути
]
