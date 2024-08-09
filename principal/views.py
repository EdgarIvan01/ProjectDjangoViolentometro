from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse, HttpRequest
from django.views.decorators.csrf import csrf_exempt
from .forms import Data
from . import secuencial
from . import uso
from . import hilos
from . import cuda
# Create your views here.
def main(request):
    num = 4
    return render(request,'index.html',{
        'num': num
    })

def hello(request: HttpRequest):
    return render(request, 'preentreno.html')

def sec(request: HttpRequest):
    execution_time = secuencial.sec_uencial()
    return render(request, 'secuencial.html',{
        'execution_time': str(execution_time)
    })

def hilos_func(request: HttpRequest):
    numero = request.POST['select']
    execution_time = hilos.main(int(numero))
    return render(request, 'hilos.html',{
        'execution_time': str(execution_time)
    })

def cuda_func(request: HttpRequest):
    execution_time = cuda.main()
    return render(request, 'cuda.html',{
        'execution_time': str(execution_time)
    })

def consulta(request):
    return render(request,'consulta.html')

def resultados(request: HttpRequest):
    texto_1 = request.POST['input1']
    predi = uso.uso(texto_1)
    return render(request, 'resultados.html',{
        'prediccion': str(predi)
    })