from django.shortcuts import render
from django.http import HttpResponse


def home(request):

	return render(request, 'game/home.html')
	#return HttpResponse("Kanarya Home Screen!")
