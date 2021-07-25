import uuid

from django.db import models
from django.contrib.auth.models import User


class Query(models.Model):

	sentence = models.TextField(default="")
	user = models.ForeignKey(User, blank=True, null=True, on_delete=models.CASCADE)
	tagged_sentence = models.TextField(default="")

	random_id = models.UUIDField(db_index=True, default=uuid.uuid4, editable=False)


class Feedback(models.Model):

	query = models.ForeignKey(Query, on_delete=models.DO_NOTHING)
	report = models.TextField(null=True)
