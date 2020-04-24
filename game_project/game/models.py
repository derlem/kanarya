from django.db import models
from django.utils import timezone
from django.contrib.auth.models import User

class Session(models.Model):
	user = models.ForeignKey(User, on_delete=models.CASCADE)


class Test(models.Model):
	session = models.ForeignKey(Session, on_delete=models.CASCADE)


class Question(models.Model):
	test = models.ForeignKey(Test, on_delete=models.CASCADE)
	sentence = models.ForeignKey(Sentence, on_delete=models.CASCADE)


class Activity(models.Model):
	sentence = models.ForeignKey(Sentence, on_delete=models.CASCADE)
	created_at = models.DateTimeField(default=timezone.now())
	name = models.CharField(max_length=10) # HINT, SEPARATE, ADJACENT, SKIP, REPORT


class Decision(models.Model):
	sentence = models.ForeignKey(Sentence, on_delete=models.CASCADE)
	name = models.CharField(max_length=10) # SEPARATE, ADJACENT, SKIP


class Report(models.Model):
	sentence = models.ForeignKey(Sentence, on_delete=models.CASCADE)
	text = models.TextField()

# All the sentences in the system. Prepopulated, static table
class Sentence(models.Model):
	index = models.IntegerField()
	text = models.TextField()
	pos = models.IntegerField()
	status = models.TextField()

