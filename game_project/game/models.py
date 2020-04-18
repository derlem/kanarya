from django.db import models
from django.utils import timezone
from django.contrib.auth.models import User

class Decision(models.Model):
	username = models.ForeignKey(User, on_delete=models.CASCADE) # if user is deleted, delete the datum
	sentence_idx = models.TextField()
	decision = models.IntegerField() # Store the decision as an integer for now.
	date = models.DateTimeField(default=timezone.now)

