from django.db import models
from django.contrib.auth.models import User


class Query(models.Model):

	sentence = models.TextField(default="")
	user = models.ForeignKey(User, blank=True, null=True, on_delete=models.CASCADE)
	tagged_sentence = models.TextField(default="")


class Feedback(models.Model):

	query = models.ForeignKey(Query, on_delete=models.DO_NOTHING)
	report = models.TextField(null=True)
