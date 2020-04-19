from django.db import models
from django.utils import timezone
from django.contrib.auth.models import User

class Decision(models.Model):
	user = models.ForeignKey(User, on_delete=models.CASCADE) # if user is deleted, delete the datum
	sentence_idx = models.TextField()
	decision = models.CharField(max_length=10) # Store the decision as an integer for now.
	#date = models.DateTimeField(default=timezone.now)

