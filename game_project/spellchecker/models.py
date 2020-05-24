from django.db import models
from django.contrib.auth.models import User

class Query(models.Model):

	sentence = models.TextField()
	isHappy = models.BooleanField( blank=True, null=True)
	user = models.ForeignKey(User, on_delete=models.CASCADE)