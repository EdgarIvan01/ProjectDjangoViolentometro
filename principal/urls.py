from django.urls import path
from . import views

urlpatterns = [
    path('', views.hello, name='hello'),
    path('index/', views.main, name='main'),
    path('secuencial',views.sec, name='secuencial'),
    path('hilos',views.hilos_func, name='hilos'),
    path('cuda',views.cuda_func, name='cuda'),
    path('consulta',views.consulta, name='consulta'),
    path('resultados',views.resultados, name='resultados'),
] 